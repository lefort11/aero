#ifndef AERO_SOLVER_TIMINGS_H
#define AERO_SOLVER_TIMINGS_H
#include "../parallel/processes.h"
#include "../parallel/wall_timer.h"
#include "../implicit/phase_timing.h"
#include <array>
#include <iosfwd>

namespace aero {
// Накапливает время и число вызовов фаз решателя с учётом MPI и GPU.
class SolverTimings {
public:
    // Для каждой операции храним отдельно число вызовов и суммарное время на данном процессе.
    enum Phase { Flux, Update, Halo, FluxExchange, Cfl, Readback, Fields, Recovery, SaTransport, LesClosure,
                 ImplicitFirst, Total = ImplicitFirst + implicit::PhaseTiming::Count, Count };
    // Настраивает сбор времён фаз и профилирование неявного решателя.
    SolverTimings(const Processes& world, bool enabled, bool cuda, std::uint64_t first_step);
    // Связывает таймеры фаз BDF с общей статистикой решателя.
    implicit::PhaseTiming bdf_phases();
    // При включённых замерах регистрирует вызов и отдаёт накопитель таймеру данной фазы.
    double* counter(Phase phase) {
        if (!enabled_)
            return nullptr;
        ++calls_[phase];
        return &seconds_[phase];
    }
    // После последнего шага собирает минимум, среднее и максимум времени по процессам.
    void report(std::uint64_t last_step, std::ostream& out);

private:
    // Группа процессов, измеряющих одни и те же фазы расчёта.
    const Processes& world_;
    // Флаги хронометража, устройства и подробного учёта BDF.
    bool enabled_, cuda_, profile_bdf_ = false;
    // Номер уже принятого шага в момент запуска замера, в том числе после восстановления.
    std::uint64_t first_;
    // Начало интервала измерения.
    WallTimer::Clock::time_point start_;
    // Число вызовов каждой измеряемой операции.
    std::array<std::uint64_t, Count> calls_{};
    // Накопленное время операций в секундах.
    std::array<double, Count> seconds_{};
};
}
#endif
