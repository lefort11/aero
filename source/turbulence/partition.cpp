#include "../aero/partition_operator.h"
#include "../parallel/parallel_for.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#endif

namespace aero {
using turbulence::Norm;
using turbulence::Operation;
void PartitionOperator::prepare_sa(const MeshPartition &part, const PeriodicPartition *periodic,
                                   const BoundaryPatches &patches, const std::vector<PeriodicPair> &pairs,
                                   const turbulence::Settings &settings, bool gpu, bool implicit) {
    sa_implicit_time_ = implicit;
    aes_->turbulence_field.distance =
        turbulence::wall_distances(part, periodic, patches, pairs, world_, state_nodes());
    world_.local([&] {
        aes_->prepare_sa_faces(grid_, !gpu);
        aes_->prepare_turbulence(settings, !gpu);
    });
}
turbulence::Step PartitionOperator::sa_step() const {
    turbulence::Step s{};
    auto &t = s.transport;
    auto aes = aes_->view();
    t.flux = flux_.data();
    t.volume = volumes_;
    t.owned = world_.size == 1 ? nullptr : owned.data();
    t.count = owned.size();
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        const auto &g = gpu_->active_grid();
        aes = gpu_->viscosity_view();
        t.flux = g.flux;
        t.volume = g.volumes;
        t.owned = g.owned;
    }
#endif
    t.sa = aes.sa;
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
    if (op == Operation::RepeatResidual) {
        parallel_for(s.transport.cell_count, [&](MeshIndex e) { s.transport.refresh_diffusivity(e); });
        host_sa_residual<true>(s);
        return;
    }
    if (op == Operation::Residual) {
        host_sa_residual<false>(s);
        return;
    }
    parallel_for(s.transport.count, [&](MeshIndex k) { s.apply(op, k); });
}
double PartitionOperator::sa_max(const turbulence::Step &s, Norm which) const {
    double value = 0.;
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            if (const char *error = device_work::aero_device_sa_norm(s, which, &value))
                throw std::runtime_error(error);
            return;
        }
#endif
        if (s.transport.count < 1024) {
            for (MeshIndex i = 0; i < s.transport.count; ++i)
                value = std::max(value, s.magnitude(which, i));
        } else {
            // Каждый поток хранит собственный максимум. В MPI наружу передаём одно число.
            host_parallel::ensure_runtime();
            Kokkos::parallel_reduce(
                "Aero: SA maximum", host_parallel::Range(0, s.transport.count),
                [&](MeshIndex i, double &best) { best = std::max(best, s.magnitude(which, i)); },
                Kokkos::Max<double>(value));
        }
    });
    return world_.reduce_double(value, true);
}
void PartitionOperator::sa_exchange() {
    if (sa_nodes_)
        sa_nodes_->transfer(world_, {1, sa_step().transport.sa.w, nullptr}, device());
}
// При включённых замерах отделяем ядра SA от асинхронной сборки газового потока.
void PartitionOperator::evaluate_sa(const std::vector<double> &state, SolverTimings &timing, bool rates_ready) {
    auto *seconds = timing.counter(SolverTimings::SaTransport);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds)
        PartitionGpu::synchronize();
#endif
    WallTimer timer(seconds);
    solve_sa_stage(state, rates_ready);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_ && seconds)
        PartitionGpu::synchronize();
#endif
}
void PartitionOperator::solve_sa_stage(const std::vector<double> &state, bool rates_ready) {
    // Для явной стадии достаточно пространственной невязки. В BDF уточняем рабочую
    // вязкость при фиксированном газе до проверки полной невязки нового слоя.
    const auto s = sa_step();
    if (!rates_ready)
        world_.local([&] { sa_apply(s, Operation::Residual); });
    if (!sa_dt_)
        return;
    const auto residual_norm = [&] {
        const double value = sa_max(s, Norm::Bdf);
        if (!std::isfinite(value))
            throw std::runtime_error("SA: nonfinite implicit transport residual");
        return value;
    };
    double norm = residual_norm();
    if (sa_target_ == 0.)
        sa_target_ =
            std::max(sa_tol_ * norm, 32. * std::numeric_limits<double>::epsilon() * s.transport.sa.ambient);
    unsigned count = 0;
    // При достижении допуска или лимита используем текущее поле рабочей вязкости.
    // Ниже по этому полю обновляются напряжения и тепловой поток газа.
    while (norm > sa_target_ && count < sa_iterations_) {
        world_.local([&] { sa_apply(s, Operation::Relax); });
        sa_exchange();
        world_.local([&] { sa_apply(s, Operation::RepeatResidual); });
        norm = residual_norm();
        ++count;
    }
    // При нуле итераций скалярное поле осталось тем же. Газовая сборка на входе
    // уже использовала эту вязкость и повторять её не требуется.
    if (count == 0 && rates_ready)
        return;
    const auto phase = count ? viscosity::Evaluation::AfterSaUpdate : viscosity::Evaluation::RowsOnly;
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
