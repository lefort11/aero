#include "time_loop.h"
#include "recovery.h"
#include "../io/flowreport.h"
#include "../io/scheduled_output.h"
#include <algorithm>
#include <iostream>
#include <limits>

namespace aero {
namespace {
// Готовит шаг для интегратора по уже вычисленному ограничению CFL.
// Расписание только ограничивает длину шага. Запись полей происходит после его выполнения.
TimeStep choose_time_step(std::uint64_t next_number, double current_time, double cfl_dt,
                          const RunSettings& run, const OutputSchedule& output) {
    // Сначала ограничиваем шаг концом расчёта, затем ближайшим сроком записи полей.
    const double remaining = run.final_time - current_time;
    const double candidate_dt = std::min(cfl_dt, remaining);
    const double dt = output.limit_dt(current_time, candidate_dt);

    // На последнем шаге берём заданное конечное время напрямую, чтобы округление
    // при сложении не оставило ещё один шаг до конца расчёта.
    const double end_time = (dt == remaining) ? run.final_time : current_time + dt;
    // Даже положительный dt может быть слишком мал, чтобы изменить current_time.
    if (!(end_time > current_time)) {
        throw std::runtime_error("Time step too small to advance time");
    }

    const bool reporting = FlowReport::due(next_number, run.flowreport_every);
    return {next_number, current_time, end_time, dt, reporting};
}

// Выполняет все виды вывода, назначенные на уже завершённый шаг.
void write_step_results(const TimeStep& step, OutputSchedule& output, Recovery* recovery,
                        const std::vector<double>& initial, const StepOperations& operations,
                        SolverTimings& timing, bool use_cuda) {
    const bool fields_due = output.due(step.number, step.end);
    const bool checkpoint_due = recovery && recovery->due(step.number, step.end);

    // Снимок полей и контрольная точка используют один и тот же буфер.
    // На GPU это одна выгрузка на оба файла. Для одного отчёта она не нужна.
    const std::vector<double>* state = nullptr;
    if (fields_due || checkpoint_due) {
        WallTimer timer(use_cuda ? timing.counter(SolverTimings::Readback) : nullptr);
        state = &operations.output_state();
    }
    if (step.reporting) {
        operations.report(step);
    }
    if (fields_due) {
        WallTimer timer(timing.counter(SolverTimings::Fields));
        operations.write_fields(step, *state);
    }
    if (checkpoint_due) {
        WallTimer timer(timing.counter(SolverTimings::Recovery));
        recovery->save(step.number, step.end, *state, initial);
    }
}
}

void integrate_time(const RunSettings& run, const Processes& world, OutputSchedule& output,
                    Recovery* recovery, const std::vector<double>& initial,
                    const StepOperations& operations) {
    std::uint64_t number = recovery ? recovery->step : 0;
    double time = recovery ? recovery->time : 0;
    if (!world.rank)
        std::cout << "Starting time integration\n";
    // При продолжении расчёта пропускаем уже записанные снимки.
    if (number)
        world.local([&] { output.resume(number, time); });

    SolverTimings timing(world, run.mpi_timing, run.use_cuda, number);
    while (time < run.final_time) {
        if (number == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Step number overflow");
        }
        const double cfl_dt = operations.cfl_step(timing);
        const TimeStep step = choose_time_step(number + 1, time, cfl_dt, run, output);
        operations.advance(step, timing);
        number = step.number;
        time = step.end;
        write_step_results(step, output, recovery, initial, operations, timing, run.use_cuda);
    }
    timing.report(number, std::cout);
    if (!world.rank)
        std::cout << "Completed " << number << " steps. Final time = " << time << '\n';
}
}
