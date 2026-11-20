#include "time_loop.h"
#include "recovery.h"
#include "../io/flowreport.h"
#include "../io/scheduled_output.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace aero {
namespace {
// Готовит шаг для интегратора по уже вычисленному ограничению CFL.
// Расписание только ограничивает длину шага. Запись полей происходит после его выполнения.
TimeStep choose_time_step(std::uint64_t next_number, double current_time, double cfl_dt,
                          const RunSettings& run, const OutputSchedule& output) {
    // Сначала ограничиваем шаг концом расчёта, затем ближайшим сроком записи полей.
    const double remaining = run.steady() ? std::numeric_limits<double>::infinity() : run.final_time - current_time;
    const double candidate_dt = std::min(cfl_dt, remaining);
    const double dt = output.limit_dt(current_time, candidate_dt);

    // На последнем шаге берём заданное конечное время напрямую, чтобы округление
    // при сложении не оставило ещё один шаг до конца расчёта.
    const double end_time = (dt == remaining) ? run.final_time : current_time + dt;
    // Даже положительный dt может быть слишком мал, чтобы изменить current_time.
    if (!std::isfinite(end_time) || !(end_time > current_time)) {
        throw std::runtime_error("Time step too small to advance time");
    }

    const bool reporting = FlowReport::due(next_number, run.flowreport_every);
    return {next_number, current_time, end_time, dt, reporting};
}

// Выполняет все виды вывода, назначенные на уже завершённый шаг.
void write_step_results(const TimeStep& step, OutputSchedule& output, Recovery* recovery,
                        const std::vector<double>& initial, const StepOperations& operations,
                        SolverTimings& timing, bool use_cuda, bool final_field = false) {
    const bool fields_due = final_field ? output.has_zones() : output.due(step.number, step.end);
    const bool checkpoint_due = recovery && recovery->due(step.number, step.end, final_field);

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
        operations.write_fields(step, *state, final_field);
    }
    if (checkpoint_due) {
        WallTimer timer(timing.counter(SolverTimings::Recovery));
        if (operations.checkpoint_history) operations.checkpoint_history(recovery->history);
        recovery->save(step.number, step.end, *state, initial);
        recovery->history = implicit::BdfHistory{};
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
    const auto first_step = number;
    double reference = 0., criterion = 0.;
    // Абсолютный критерий измеряется в единицах |R|/V, относительный —
    // в долях начальной нормы. Нулевой исходный отток означает стационар.
    const auto steady_measure = [&](double norm) {
        return run.steady_absolute ? norm : (reference > 0. ? norm / reference : 0.);
    };
    const char* criterion_name = run.steady_absolute ? "absolute" : "relative";
    const bool steady = run.steady();
    while (steady || time < run.final_time) {
        if (number == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Step number overflow");
        }
        const double cfl_dt = operations.cfl_step(timing);
        // CFL уже собрал потоки начального поля. При продолжении отсчитываем
        // снижение невязки от состояния, загруженного из контрольной точки.
        if (steady && number == first_step) {
            reference = operations.steady_residual();
            criterion = steady_measure(reference);
            if (criterion <= run.steady_tol) {
                const TimeStep initial_step{number, time, time, 0., false};
                write_step_results(initial_step, output, recovery, initial, operations, timing, run.use_cuda, true);
                break;
            }
        }
        const TimeStep step = choose_time_step(number + 1, time, cfl_dt, run, output);
        operations.advance(step, timing);
        number = step.number;
        time = step.end;
        bool converged = false, exhausted = false;
        if (steady) {
            // Внутренняя сходимость Ньютона относится к одному псевдошагу.
            // Для стационара отдельно проверяем полный отток из каждого объёма.
            criterion = steady_measure(operations.steady_residual());
            converged = criterion <= run.steady_tol;
            exhausted = number - first_step >= run.steady_max_steps;
        }
        write_step_results(step, output, recovery, initial, operations, timing, run.use_cuda, converged || exhausted);
        if (steady && !world.rank && (step.reporting || converged || exhausted))
            std::cout << "Steady " << criterion_name << " residual, step " << number << ": " << criterion << '\n';
        if (converged) break;
        if (exhausted) throw std::runtime_error("Steady: pseudo-step limit reached before convergence");
    }
    timing.report(number, std::cout);
    if (!world.rank) {
        if (steady)
            std::cout << "Steady state converged, " << criterion_name << " spatial residual = " << criterion << '\n';
        std::cout << "Completed " << number << (steady ? " steps. Pseudo-time = " : " steps. Final time = ")
                  << time << '\n';
    }
}
}
