#include "time_loop.h"
#include "recovery.h"
#include "../io/flowreport.h"
#include "../io/scheduled_output.h"
#include <algorithm>
#include <iostream>
#include <limits>

namespace aero {
void integrate_time(const RunSettings& run,const Processes& world,OutputSchedule& output,Recovery* recovery,
                    const std::vector<double>& initial,const StepOperations& operations) {
    std::uint64_t number=recovery?recovery->step:0;
    double time=recovery?recovery->time:0;
    if (!world.rank) std::cout<<"Starting time integration\n";
    if (number) world.local([&] { output.resume(number,time); });
    SolverTimings timing(world,run.mpi_timing,run.use_cuda,number);
    while (time<run.final_time) {
        if (number==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Step number overflow");
        const double cfl_dt=operations.cfl_step(timing);
        const double remaining=run.final_time-time;
        const double dt=output.limit_dt(time,std::min(cfl_dt,remaining));
        const double next=dt==remaining?run.final_time:time+dt;
        if (!(next>time)) throw std::runtime_error("Time step too small to advance time");
        const TimeStep step={number+1,time,next,dt,FlowReport::due(number+1,run.flowreport_every)};
        operations.advance(step,timing);
        number=step.number;
        time=step.end;

        const bool writing=output.due(number,time);
        const bool saving=recovery && recovery->due(number,time);
        const std::vector<double>* state=nullptr;
        if (writing || saving) {
            WallTimer timer(run.use_cuda?timing.counter(SolverTimings::Readback):nullptr);
            state=&operations.output_state();
        }
        if (step.reporting) operations.report(step);
        if (writing) {
            WallTimer timer(timing.counter(SolverTimings::Fields));
            operations.write_fields(step,*state);
        }
        if (saving) {
            WallTimer timer(timing.counter(SolverTimings::Recovery));
            recovery->save(number,time,*state,initial);
        }
    }
    timing.report(number,std::cout);
    if (!world.rank) std::cout<<"Completed "<<number<<" steps. Final time = "<<time<<'\n';
}
}
