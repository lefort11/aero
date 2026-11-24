#include "solver_timings.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_check.h"
#endif
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>

namespace aero {
// Настраивает сбор времён фаз и профилирование неявного решателя.
SolverTimings::SolverTimings(const Processes& world, bool enabled, bool cuda, std::uint64_t first_step)
    : world_(world), enabled_(enabled), cuda_(cuda), first_(first_step) {
    if (!enabled_)
        return;
    const char* profile = std::getenv("AERO_PROFILE_BDF");
    profile_bdf_ = profile && std::strcmp(profile,"1")==0;
#ifdef AERO_WITH_MPI
    // Синхронизируем старт единственного общего интервала замера.
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    start_ = WallTimer::Clock::now();
}
// Связывает таймеры фаз BDF с общей статистикой решателя.
implicit::PhaseTiming SolverTimings::bdf_phases() {
    if (!profile_bdf_) return {};
    implicit::PhaseTiming timing{seconds_.data()+ImplicitFirst,calls_.data()+ImplicitFirst,nullptr};
#ifdef AERO_WITH_KOKKOS_GPU
    if (cuda_) timing.wait=[] { gpu::checked(gpu::wait(nullptr)); };
#endif
    return timing;
}
// После последнего шага собирает минимум, среднее и максимум времени по процессам.
void SolverTimings::report(std::uint64_t last_step, std::ostream& out) {
    if (!enabled_)
        return;
    seconds_[Total] = std::chrono::duration<double>(WallTimer::Clock::now() - start_).count();
    calls_[Total] = 1;
    // При продолжении считаем только шаги текущего запуска.
    const auto steps = last_step - first_;
    auto phases=seconds_;
    // Общий таймер сборки захватывает также вязкие ядра. Вычитаем их
    // на каждом ранге до MPI_MIN/MAX, чтобы строки отчёта можно было складывать.
    phases[ImplicitFirst+implicit::PhaseTiming::Matrix]-=phases[ImplicitFirst+implicit::PhaseTiming::ViscousMatrix];
    auto low = phases, sum = phases, high = phases;
#ifdef AERO_WITH_MPI
    MPI_Reduce(phases.data(), low.data(), Count, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(phases.data(), sum.data(), Count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(phases.data(), high.data(), Count, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#endif
    // Редукции завершены на всех процессах. Таблицу печатает процесс 0.
    if (world_.rank)
        return;
    const char* names[] = {"flux_compute",    "stage_update", "halo_exchange",  "flux_exchange", "cfl",
                           "output_readback", "fields_write", "recovery_write", "sa_transport", "les_closure",
                           "matrix_convective_and_graph", "matrix_viscous", "precondition_setup", "bicgstab",
                           "preconditioned_predictor", "integration"};
    std::ostringstream table;
    table.imbue(std::locale::classic());
    table << (cuda_ ? "GPU" : "CPU") << " timings: ranks=" << world_.size << " start_step=" << first_
          << " end_step=" << last_step << " steps=" << steps
          << "\nphase\tcalls_per_rank\tmin_s\tmean_s\tmax_s\tmax_s_per_step\n"
          << std::scientific << std::setprecision(9);
    for (int i = 0; i < Count; ++i) {
        if (i>=ImplicitFirst && i<Total && !profile_bdf_) continue;
        table << names[i] << '\t' << calls_[i] << '\t' << low[i] << '\t' << sum[i] / world_.size << '\t'
              << high[i] << '\t';
        if (steps)
            table << high[i] / static_cast<double>(steps);
        else
            table << "n/a";
        table << '\n';
    }
    out << table.str();
}
}
