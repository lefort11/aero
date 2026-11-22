#include "../aero/partition_operator.h"
#include "../parallel/parallel_for.h"
#include "convection.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#include "../parallel/device_check.h"
#endif

namespace aero {
using turbulence::Norm;
using turbulence::Operation;
void PartitionOperator::prepare_sa(const MeshPartition &part, const PeriodicPartition *periodic,
                                   const BoundaryPatches &patches, const std::vector<PeriodicPair> &pairs,
                                   const turbulence::Settings &settings, bool gpu, bool implicit, bool steady, bool has_stencil) {
    sa_implicit_time_ = implicit;
    sa_high_order_ = settings.high_order && has_stencil;
    aes_->turbulence_field.distance =
        turbulence::wall_distances(part, periodic, patches, pairs, world_, state_nodes());
    // Для RANS на стационарном шаге можно один раз построить линии по геометрии сетки.
    const bool use_lines = steady && !settings.ddes;
    world_.local([&] {
        if (sa_high_order_ && !gpu) {
            host_sa_correction_.resize(grid_.edges);
            sa_correction_ = host_sa_correction_.data();
        }
        aes_->prepare_sa_faces(grid_, !gpu);
        aes_->prepare_turbulence(settings, !gpu, implicit, use_lines ? &owned : nullptr);
    });
    sa_lines_ = use_lines && world_.reduce_double(aes_->turbulence_field.line_segments.empty() ? 0. : 1., true) > 0.;
}
turbulence::Step PartitionOperator::sa_step() const {
    turbulence::Step s{};
    auto &t = s.transport;
    auto aes = aes_->view();
    t.flux = flux_.data();
    t.convection_correction = first_order_start ? nullptr : sa_correction_;
    t.profiles = grid_.patches.profiles;
    t.volume = volumes_;
    t.owned = world_.size == 1 ? nullptr : owned.data();
    t.count = owned.size();
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        const auto &g = gpu_->active_grid();
        aes = gpu_->viscosity_view();
        t.flux = g.flux;
        t.profiles = g.geometry.patches.profiles;
        t.volume = g.volumes;
        t.owned = g.owned;
    }
#endif
    t.sa = aes.sa;
    t.filter = aes.nodal_filter();
    t.gas_nodes = aes.gas;
    t.cells = aes.cells;
    t.cell_count = aes.cell_count;
    t.dim = grid_.dim;
    t.edges = grid_.edges;
    const double ratio = sa_bdf2_ && sa_previous_dt_ > 0. ? sa_dt_ / sa_previous_dt_ : 0.;
    const double fraction = ratio / (1. + ratio);
    s.keep_gas = sa_implicit_time_;
    s.dt = sa_dt_;
    s.local_steps = SteadyNodeSteps::active(steady_steps_.get());
    s.leading = 1. + fraction;
    s.history = ratio * fraction;
    return s;
}
namespace {
template <unsigned dim, bool same_gas> void host_sa_rows(const turbulence::Step &step) {
    const auto &transport = step.transport;
    // На CPU строки лежат подряд. Размерность и повторное использование газа
    // выбираем до параллельного цикла, чтобы внутренние циклы имели постоянные границы.
    const auto row = [&](MeshIndex k) { transport.residual_row<dim, same_gas, true>(k, step.keep_gas); };
    if (transport.count < 1024) {
        for (MeshIndex k = 0; k < transport.count; ++k)
            row(k);
    } else {
        host_parallel::for_indices("Aero: SA residual", transport.count, row);
    }
}

template <bool same_gas> void host_sa_residual(const turbulence::Step &step) {
    switch (step.transport.dim) {
    case 1:
        host_sa_rows<1, same_gas>(step);
        break;
    case 2:
        host_sa_rows<2, same_gas>(step);
        break;
    default:
        host_sa_rows<3, same_gas>(step);
        break;
    }
}
}

void PartitionOperator::sa_apply(const turbulence::Step &s, Operation op) const {
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        if (const char *error = device_work::aero_device_sa_step(s, op))
            throw std::runtime_error(error);
        return;
    }
#endif
    if (op==Operation::LineRelax && s.transport.sa.lines.count) {
        parallel_for(s.transport.sa.lines.count,[&](MeshIndex k){s.relax_line(k);},64);
        return;
    }
    if (op == Operation::Residual) {
        if (s.transport.sa.diffusion.vertex_contributions)
            parallel_for(s.transport.cell_count,[&](MeshIndex e){s.transport.diffuse_element(e);});
        host_sa_residual<false>(s);
        return;
    }
    parallel_for(s.transport.count, [&](MeshIndex k) { s.apply(op, k); });
}
namespace {
template<unsigned dim,bool refresh>
double host_sa_norm(const turbulence::Step& s,Norm which) {
    const auto magnitude=[&](MeshIndex k) {
        return refresh ? s.updated_defect<dim,true>(k) : s.magnitude(which,k);
    };
    double value=0.;
    if (s.transport.count<1024) {
        for (MeshIndex i=0;i<s.transport.count;++i) value=std::max(value,magnitude(i));
    } else {
        host_parallel::ensure_runtime();
        Kokkos::parallel_reduce("Aero: SA maximum",host_parallel::Range(0,s.transport.count),
            [&](MeshIndex i,double& best) { best=std::max(best,magnitude(i)); },Kokkos::Max<double>(value));
    }
    return value;
}
}
double PartitionOperator::sa_max(const turbulence::Step &s, Norm which,bool refresh) const {
    double value = 0.;
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            if (const char *error = device_work::aero_device_sa_norm(s, which, refresh, &value))
                throw std::runtime_error(error);
            return;
        }
#endif
        if (!refresh) { value=host_sa_norm<0,false>(s,which); return; }
        parallel_for(s.transport.cell_count,[&](MeshIndex e){s.transport.refresh_diffusivity<true>(e);});
        switch (s.transport.dim) {
        case 1: value=host_sa_norm<1,true>(s,which); break;
        case 2: value=host_sa_norm<2,true>(s,which); break;
        default: value=host_sa_norm<3,true>(s,which); break;
        }
    });
    return world_.reduce_double(value, true);
}
// Замыкание пересчитываем из текущего поля на каждой стадии и итерации Ньютона.
void PartitionOperator::update_les(const std::vector<double>& state, SolverTimings& timing,
                                   viscosity::Evaluation phase) {
    auto* seconds=timing.counter(SolverTimings::LesClosure);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds) PartitionGpu::synchronize();
#endif
    WallTimer timer(seconds);
    double* nu=nullptr;
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            const auto op=gpu_->viscosity_view();
            device_work::check(device_work::aero_device_viscosity(op,gpu_->active_grid().state,
                gpu_->current_gas(),gpu_->stage_material_current() && phase==viscosity::Evaluation::LesClosure
                                     ? viscosity::Evaluation::FilterOnly : phase));
            nu=op.eddy_nu;
            return;
        }
#endif
        aes_->evaluate(state.data(),nullptr,gas_current_ && grid_.packed.gas,phase);
        nu=aes_->view().eddy_nu;
    });
    if (world_.size>1 && !aes_->view().c_des) turbulence_exchange_->transfer(world_,{1,nu,nullptr},device());
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds) PartitionGpu::synchronize();
#endif
}
void PartitionOperator::sa_exchange() {
    if (turbulence_exchange_)
        turbulence_exchange_->transfer(world_, {1, sa_step().transport.sa.w, nullptr}, device());
}

void PartitionOperator::prepare_sa_convection(const turbulence::Step& step) const {
    if (!sa_high_order_ || first_order_start) return;
    // Список начинается со своих внутренних рёбер. Физические границы в конце
    // списка сохраняют заданный входной профиль и узловое значение на выходе.
    const auto count = compute_.size() - (grid_.faces - grid_.edges);
    world_.local([&] {
        const auto* geometry = &grid_;
        const auto* computed = world_.size == 1 ? nullptr : compute_.data();
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            geometry = &gpu_->active_grid().geometry;
            computed = gpu_->active_grid().computed;
        }
#endif
        const turbulence::SaEdgeConvection edge{geometry->stencil,geometry->surfaces,computed,
            step.transport.flux,step.transport.sa.w,sa_correction_,geometry->options.upw_coeff,geometry->options.mp6_hybrid};
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            device_work::check(device_work::aero_device_sa_convection(edge,count));
            return;
        }
#endif
        parallel_for(count,edge);
    });
    // Все ранги участвуют, в том числе ранг без собственных рёбер.
    if (sa_edge_exchange_) sa_edge_exchange_->transfer(world_,{1,sa_correction_,nullptr},device());
}
// При включённых замерах отделяем ядра SA от асинхронной сборки газового потока.
void PartitionOperator::evaluate_sa(const std::vector<double> &state, SolverTimings &timing, bool rates_ready, bool rows_ready) {
    auto *seconds = timing.counter(SolverTimings::SaTransport);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds)
        PartitionGpu::synchronize();
#endif
    WallTimer timer(seconds);
    solve_sa_stage(state, rates_ready, rows_ready);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds)
        PartitionGpu::synchronize();
#endif
}
void PartitionOperator::solve_sa_stage(const std::vector<double> &state, bool rates_ready, bool rows_ready) {
    // Для явной стадии достаточно пространственной невязки. В BDF уточняем рабочую
    // вязкость при фиксированном газе до проверки полной невязки нового слоя.
    auto s = sa_step();
    if (!rates_ready) {
        prepare_sa_convection(s);
        world_.local([&] { sa_apply(s, Operation::Residual); });
    }
    if (!sa_dt_)
        return;
    const auto residual_norm = [&](bool refresh) {
        if (refresh) prepare_sa_convection(s);
        const double value = sa_max(s, Norm::Bdf, refresh);
        if (!s.extrapolate && !std::isfinite(value))
            throw std::runtime_error("SA: nonfinite implicit transport residual");
        return value;
    };
    double norm = residual_norm(false);
    if (sa_target_ == 0.)
        sa_target_ =
            std::max(sa_tol_ * norm, 32. * std::numeric_limits<double>::epsilon() * s.transport.sa.ambient);
    unsigned count = 0;
    // При достижении допуска или лимита используем текущее поле рабочей вязкости.
    // Ниже по этому полю обновляются напряжения и тепловой поток газа.
    while (norm > sa_target_ && count < sa_iterations_) {
        const double previous_norm=norm;
        // На небольшом псевдошаге нескольких точечных поправок обычно достаточно.
        // Прогонку запускаем, только если после четырёх итераций перенос ещё не сошёлся.
        const bool line_trial=sa_lines_ && count>=4;
        s.extrapolate=line_trial || (count%2==1 && count+1<sa_iterations_);
        world_.local([&] { sa_apply(s,line_trial ? Operation::LineRelax : Operation::Relax); });
        sa_exchange();
        norm = residual_norm(true);
        // Проверяем настоящую нелинейную невязку по всем процессам. Если
        // ускоренная поправка её не уменьшила, принимаем обычную итерацию.
        if (s.extrapolate && !(norm<previous_norm)) {
            s.extrapolate=false;
            world_.local([&] { sa_apply(s,Operation::RestorePlain); });
            sa_exchange();
            norm=residual_norm(true);
        }
        ++count;
    }
    // При нуле итераций скалярное поле осталось тем же. Газовая сборка на входе
    // уже использовала эту вязкость и повторять её не требуется.
    // Пробная поправка Ньютона тоже уже содержит полную вязкую невязку
    // данного газа. Если SA приняла старое nu_tilde, эти строки остаются точными.
    if (count == 0 && (rows_ready || rates_ready))
        return;
    const bool same_gas=rows_ready || rates_ready;
    const auto phase = !count ? viscosity::Evaluation::RowsOnly :
        (same_gas ? viscosity::Evaluation::AfterSaFixedGas : viscosity::Evaluation::AfterSaUpdate);
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            const auto &g = gpu_->active_grid();
            if (const char *error = device_work::aero_device_viscosity(gpu_->viscosity_view(), g.state,
                                                                       g.geometry.packed.gas, phase))
                throw std::runtime_error(error);
            return;
        }
#endif
        aes_->evaluate(state.data(), grid_.packed.gas, true, phase);
    });
}
std::vector<double> PartitionOperator::sa_output(const std::vector<double> &state) {
    if (les_active()) {
        std::vector<double> columns(2*state_nodes(),0.);
        // При плоском течении поле ν_t известно точно, отдельного массива на GPU нет.
        if (!aes_->view().computes_les()) return columns;
        std::vector<double> values(state_nodes());
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            if (!values.empty()) gpu::checked(gpu::copy(values.data(),gpu_->viscosity_view().eddy_nu,
                values.size()*sizeof(double),gpu::Copy::ToHost));
        } else
#endif
        if (!values.empty()) std::copy_n(aes_->view().eddy_nu,values.size(),values.data());
        for (size_t i=0;i<values.size();++i) columns[2*i+1]=values[i];
        return columns;
    }
    std::vector<double> w, previous;
    sa_checkpoint(w, previous, false, 0.);
    std::vector<double> columns(2 * w.size());
    const auto aes = aes_->view();
    parallel_for(w.size(), [&](MeshIndex n) {
        EulerState gas;
        if (euler_state(grid_.dim, state.data() + n * width_, grid_.options.gamma, &gas) !=
            EulerStatus::Success)
            throw std::runtime_error("SA: invalid state in field output");
        const double nu = aes.law.dynamic_viscosity(gas.pressure / gas.density) / gas.density;
        columns[2 * n] = w[n];
        columns[2 * n + 1] = w[n] * turbulence::fv1(w[n] / nu);
    });
    return columns;
}
void PartitionOperator::sa_checkpoint(std::vector<double> &current, std::vector<double> &previous, bool restore,
                                      double previous_dt) {
    // На диск пишем поле с активного исполнителя. Соседние значения после чтения заполнит MPI.
    const auto s = sa_step().transport.sa;
    const size_t n = state_nodes();
    const auto transfer = [&](double *field, std::vector<double> &host) {
        host.resize(n);
        if (!n)
            return;
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            const auto status = restore ? gpu::copy(field, host.data(), n * sizeof(double), gpu::Copy::ToDevice)
                                        : gpu::copy(host.data(), field, n * sizeof(double), gpu::Copy::ToHost);
            if (status != gpu::Status::Success)
                throw std::runtime_error(gpu::message(status));
            return;
        }
#endif
        if (restore)
            std::copy_n(host.data(), n, field);
        else
            std::copy_n(field, n, host.data());
    };
    transfer(s.w, current);
    if (previous_dt > 0.)
        transfer(s.previous, previous);
    else
        previous.clear();
    if (restore)
        sa_previous_dt_ = previous_dt;
}
}
