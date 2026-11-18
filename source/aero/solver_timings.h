#ifndef AERO_SOLVER_TIMINGS_H
#define AERO_SOLVER_TIMINGS_H
#include "../parallel/processes.h"
#include "../parallel/wall_timer.h"
#include <array>
#include <iosfwd>

namespace aero {
class SolverTimings {
public:
    enum Phase { Flux,Update,Halo,FluxExchange,Cfl,Readback,Fields,Recovery,Total,Count };
    SolverTimings(const Processes& world,bool enabled,bool cuda,std::uint64_t first_step);
    double* counter(Phase phase) {
        if (!enabled_) return nullptr;
        ++calls_[phase];
        return &seconds_[phase];
    }
    void report(std::uint64_t last_step,std::ostream& out);
private:
    const Processes& world_;
    bool enabled_,cuda_;
    std::uint64_t first_;
    WallTimer::Clock::time_point start_;
    std::array<std::uint64_t,Count> calls_{};
    std::array<double,Count> seconds_{};
};
}
#endif
