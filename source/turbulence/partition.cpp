#include "step_execution.h"
#include "scalar_solver.h"
#include <cstdio>
#include <cstdlib>
#include "../aero/partition_operator.h"
#include "../parallel/parallel_for.h"
#include "convection.h"
#include "../timestep/bdf_weights.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#include "../parallel/device_check.h"
#endif

namespace aero {
using turbulence::Norm;
using turbulence::Operation;
// Готовит поле SA, расстояния до стенок, границы и рабочую память.
void PartitionOperator::prepare_sa(const MeshPartition &part, const PeriodicPartition *periodic,
                                   const BoundaryPatches &patches, const std::vector<PeriodicPair> &pairs,
                                   const turbulence::Settings &settings, bool gpu, bool implicit, bool has_stencil) {
    sa_implicit_time_ = implicit;
    sa_high_order_ = settings.high_order && has_stencil;
    aes_->turbulence_field.distance =
        turbulence::wall_distances(part, periodic, patches, pairs, world_, state_nodes());
    world_.local([&] {
        if (sa_high_order_ && !gpu) {
            host_sa_correction_.resize(grid_.edges);
            sa_correction_ = host_sa_correction_.data();
        }
        aes_->prepare_sa_faces(grid_, !gpu);
        aes_->prepare_turbulence(settings, !gpu, implicit, nullptr);
    });
}
// Собирает представление текущей стадии переноса SA.
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
        t.gas_faces={g.geometry.surfaces,g.geometry.incidence,g.geometry.offsets};
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
    t.shared_geometry = aes.stencil;
    t.heat_to_diffusion = turbulence::diffusion_scale(aes.law.gamma,aes.law.prandtl);
    t.cell_count = aes.cell_count;
    t.dim = grid_.dim;
    t.edges = grid_.edges;
    const auto weights = BdfWeights::for_step(sa_bdf2_, sa_dt_, sa_previous_dt_);
    s.keep_gas = sa_implicit_time_;
    s.dt = sa_dt_;
    s.local_steps = SteadyNodeSteps::active(steady_steps_.get());
    s.leading = weights.leading;
    s.history = weights.history;
    return s;
}
// Выполняет выбранную операцию SA на CPU или GPU.
void PartitionOperator::sa_apply(const turbulence::Step &s, Operation op) const {
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        device_work::check(device_work::aero_device_sa_step(s, op));
        return;
    }
#endif
    host_parallel::ensure_runtime();
    turbulence::execution::apply_step(host_parallel::Execution(),s,op);
}
// Вычисляет глобальную норму SA с редукцией между процессами.
double PartitionOperator::sa_max(const turbulence::Step &s, Norm which,bool refresh) const {
    double value = 0.;
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            device_work::check(device_work::aero_device_sa_norm(s, which, refresh, &value));
            return;
        }
#endif
        host_parallel::ensure_runtime();
        value=turbulence::execution::measure_norm(host_parallel::Execution(),s,which,refresh);
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
// Неявный шаг фиксирует длину DDES внутри Ньютона. Перед первым использованием
// принятого поля пересчитываем фильтр по готовому газу. Меняется источник SA,
// поэтому обновляем rate и diagonal; межузловые потоки и конвекция SA уже готовы.
void PartitionOperator::refresh_accepted_ddes(const std::vector<double>& state, SolverTimings& timing) {
    if (!ddes_filter_pending_) return;
    WallTimer timer(timing.counter(SolverTimings::LesClosure));
    world_.local([&] {
        // На подобласти с постоянной длиной без экранирования фильтр не участвует в источнике.
        if (aes_->view().skip_ddes_filter) return;
        const auto phase = viscosity::Evaluation::FilterOnly;
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            device_work::check(device_work::aero_device_viscosity(gpu_->viscosity_view(),
                gpu_->active_grid().state,gpu_->current_gas(),phase));
        } else
#endif
            aes_->evaluate(state.data(),grid_.packed.gas,gas_current_,phase);
        sa_apply(sa_step(),Operation::Residual);
    });
    ddes_filter_pending_ = false;
}
// Обновляет значения рабочей вязкости в halo-узлах.
void PartitionOperator::sa_exchange() {
    if (turbulence_exchange_)
        turbulence_exchange_->transfer(world_, {1, sa_step().transport.sa.w, nullptr}, device());
}

// Готовит геометрию и веса конвективных потоков SA.
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
            device_work::check(device_work::aero_device_sa_convection(edge,count,geometry->packed.tiles,geometry->packed.order));
            return;
        }
#endif
        host_parallel::ensure_runtime();
        turbulence::execution::convection(host_parallel::Execution(),edge,count,geometry->packed.tiles,geometry->packed.order);
    });
    // Все ранги участвуют, в том числе ранг без собственных рёбер.
    if (sa_edge_exchange_) sa_edge_exchange_->transfer(world_,{1,sa_correction_,nullptr},device());
}
// При включённых замерах отделяем ядра SA от асинхронной сборки газового потока.
double PartitionOperator::evaluate_sa(const std::vector<double> &state, SolverTimings &timing, bool rates_ready, bool rows_ready,
                                    bool reference_only) {
    auto *seconds = timing.counter(SolverTimings::SaTransport);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds)
        PartitionGpu::synchronize();
#endif
    WallTimer timer(seconds);
    const double defect = solve_sa_stage(state, rates_ready, rows_ready, reference_only);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds)
        PartitionGpu::synchronize();
#endif
    return defect;
}
// Настройки берутся из общего неявного интегратора ровно один раз на слой.
void PartitionOperator::begin_sa_linear_step(bool history_allowed) {
    sa_history_allowed_=history_allowed && !(steady_steps_ && steady_steps_->local) &&
        !sa_settings_.without_predictor && sa_settings_.prediction_order;
    if (sa_linear_) sa_linear_->begin_step(sa_settings_.jacobian_update_every);
}
// Полином экстраполяции использует только принятые слои.
void PartitionOperator::commit_sa_linear_step(double dt) {
    if (sa_linear_) sa_linear_->commit(dt,sa_history_allowed_);
}
// Решает текущую стадию SA с согласованием потоков и диффузии.
double PartitionOperator::solve_sa_stage(const std::vector<double> &state, bool rates_ready, bool rows_ready, bool reference_only) {
    // Для явной стадии достаточно пространственной невязки. В BDF уточняем рабочую
    // вязкость при фиксированном газе до проверки полной невязки нового слоя.
    auto s = sa_step();
    if (!rates_ready) {
        prepare_sa_convection(s);
        world_.local([&] { sa_apply(s, Operation::Residual); });
    }
    if (!sa_dt_) return 0.;
    (void)reference_only;
    double norm = sa_max(s,Norm::Bdf);
    const double initial_norm = norm;
    if (!std::isfinite(norm)) throw std::domain_error("SA: nonfinite transport residual");
    if (norm <= sa_target_) return norm;
    // Для переноса используется тот же выбор графа, предобусловливателя и линейного допуска.
    auto faces=turbulence::GasFaces{grid_.surfaces,grid_.incidence,grid_.offsets};
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) faces=s.transport.gas_faces;
#endif
    if (!sa_linear_) {
        const auto graph=bdf_->scalar_topology(sa_settings_.csgs);
        const auto linear=linear_grid();
        const auto exchange=[this](double* field) {
            turbulence_exchange_->transfer(world_,{1,field,nullptr},device());
        };
        world_.local([&] {
            sa_linear_=std::make_shared<turbulence::ScalarSolver>(graph,linear.partition,linear.device,
                s.transport.count,state_nodes(),bdf_->report_workspace(),width_,sa_settings_.jacobian_update_every>1,exchange);
        });
    }
    const auto refresh=[&] {
        sa_exchange();
        prepare_sa_convection(s);
        return sa_max(s,Norm::Bdf,true);
    };
    const bool spatial=steady_steps_ && !s.transport.sa.ddes && sa_settings_.simplified_newton();
    sa_linear_->prepare_stage(s,faces,sa_settings_);
    const double reference=sa_target_/sa_settings_.newton_relative;
    norm=sa_linear_->initial_guess(norm,reference,sa_settings_,spatial,steady_steps_ && steady_steps_->local,sa_history_allowed_,refresh);
    bool filtered=false, shifted=false;
    for (unsigned iteration=0;iteration<sa_settings_.max_iterations && norm>sa_target_;++iteration) {
        if (!sa_settings_.simplified_newton()) sa_linear_->prepare(s,faces,sa_settings_.simplified_jacobian,true);
        const auto solved=sa_linear_->correct(sa_settings_,norm,filtered,shifted);
        if (solved.status!=implicit::LinearStatus::Converged)
            throw implicit::LinearSolveFailure("SA BiCGSTAB failed above linear tolerance");
        shifted=filtered;
        const double before=norm;
        norm=refresh();
        // При профилировании сохраняем полную нелинейную невязку отдельно от линейной.
        if (world_.rank == 0 && std::getenv("AERO_PROFILE_BDF"))
            std::printf("SA Newton %u: initial=%.12e before=%.12e after=%.12e target=%.12e BiCGSTAB=%u linear_residual=%.12e\n",
                        iteration+1,initial_norm,before,norm,sa_target_,solved.iterations,solved.residual_norm);
        if (!std::isfinite(norm)) throw std::domain_error("SA: nonfinite Newton residual");
        if (sa_settings_.damp_correction && CorrectionFilter::slow(norm,before)) filtered=true;
    }
    const bool same_gas=rows_ready || rates_ready;
    const auto phase = same_gas ? viscosity::Evaluation::AfterSaFixedGas : viscosity::Evaluation::AfterSaUpdate;
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            const auto &g = gpu_->active_grid();
            device_work::check(device_work::aero_device_viscosity(gpu_->viscosity_view(), g.state,
                                                                       g.geometry.packed.gas, phase));
            return;
        }
#endif
        aes_->evaluate(state.data(), grid_.packed.gas, true, phase);
    });
    return norm;
}
// Формирует пары рабочей и физической вихревой вязкости для вывода.
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
// Сохраняет или восстанавливает текущее и предыдущее поля SA.
void PartitionOperator::sa_checkpoint(std::vector<double> &current, std::vector<double> &previous, bool restore,
                                      double previous_dt) {
    // На диск пишем поле с активного исполнителя. Соседние значения после чтения заполнит MPI.
    const auto s = sa_step().transport.sa;
    const size_t n = state_nodes();
    // Переносит рабочую вязкость между основным полем и буфером пробной стадии.
    const auto transfer = [&](double *field, std::vector<double> &host) {
        host.resize(n);
        if (!n)
            return;
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            const auto status = restore ? gpu::copy(field, host.data(), n * sizeof(double), gpu::Copy::ToDevice)
                                        : gpu::copy(host.data(), field, n * sizeof(double), gpu::Copy::ToHost);
            gpu::checked(status);
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
