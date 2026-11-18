#ifndef AERO_TIME_LOOP_H
#define AERO_TIME_LOOP_H
#include "solver_timings.h"
#include <functional>

namespace aero {
class OutputSchedule;
class Recovery;
struct RunSettings;

struct TimeStep {
    std::uint64_t number; // one-based number of the step being taken
    double start,end,dt;
    bool reporting;
};

struct StepOperations {
    // Compute the first-stage fluxes once and reuse them in advance.
    std::function<double(SolverTimings&)> cfl_step;
    std::function<void(const TimeStep&,SolverTimings&)> advance;
    // Read back the accepted state only when a field or recovery file is due.
    std::function<const std::vector<double>&()> output_state;
    std::function<void(const TimeStep&)> report;
    std::function<void(const TimeStep&,const std::vector<double>&)> write_fields;
};

void integrate_time(const RunSettings& run,const Processes& world,OutputSchedule& output,Recovery* recovery,
                    const std::vector<double>& initial,const StepOperations& operations);
}
#endif
