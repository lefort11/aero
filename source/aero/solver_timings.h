#ifndef AERO_SOLVER_TIMINGS_H
#define AERO_SOLVER_TIMINGS_H
#include "../parallel/processes.h"
#include "../parallel/wall_timer.h"
#include <array>
#include <iosfwd>

namespace aero {
class SolverTimings {
public:
    // Для каждой операции храним отдельно число вызовов и суммарное время на данном процессе.
    enum Phase { Flux, Update, Halo, FluxExchange, Cfl, Readback, Fields, Recovery, SaTransport, LesClosure, Total, Count };
    SolverTimings(const Processes& world, bool enabled, bool cuda, std::uint64_t first_step);
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
    bool enabled_, cuda_;
    // Номер уже принятого шага в момент запуска замера, в том числе после восстановления.
    std::uint64_t first_;
    WallTimer::Clock::time_point start_;
    std::array<std::uint64_t, Count> calls_{};
    std::array<double, Count> seconds_{};
};
}
#endif
