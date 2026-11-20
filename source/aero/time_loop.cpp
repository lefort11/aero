#include "time_loop.h"
#include "recovery.h"
#include "../io/report.h"
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
    double deadline = run.steady() ? std::numeric_limits<double>::infinity() : run.final_time;
    // Все стадии шага используют один порядок. Попадаем точно в t_fo,
    // даже если CFL или расписание вывода допускают более длинный шаг.
    if (run.first_order_start(current_time)) deadline = std::min(deadline, run.t_fo);
    const double remaining = deadline - current_time;
    const double candidate_dt = std::min(cfl_dt, remaining);
    const double dt = output.limit_dt(current_time, candidate_dt);

    // На последнем шаге берём заданное конечное время напрямую, чтобы округление
    // при сложении не оставило ещё один шаг до конца расчёта.
    const double end_time = (dt == remaining) ? deadline : current_time + dt;
    // Даже положительный dt может быть слишком мал, чтобы изменить current_time.
    if (!std::isfinite(end_time) || !(end_time > current_time)) {
        throw std::runtime_error("Time step too small to advance time");
    }

    const bool reporting = Report::due(next_number, run.report_every);
    return {next_number, current_time, end_time, dt, reporting};
}

// Выполняет все виды вывода, назначенные на уже завершённый шаг.
void write_step_results(const TimeStep& step, OutputSchedule& output, Recovery& recovery,
                        const std::vector<double>& initial, const StepOperations& operations,
                        SolverTimings& timing, bool use_cuda, bool final_step) {
    const bool fields_due = final_step ? output.has_zones() : output.due(step.number, step.end);
    const bool checkpoint_due = final_step || recovery.due(step.number);

    // Снимок полей и контрольная точка используют один и тот же буфер.
    // На GPU это одна выгрузка на оба файла. Для одного отчёта она не нужна.
    const std::vector<double>* state = nullptr;
    const bool error_due = operations.exact_error && (step.reporting || final_step);
    if (fields_due || checkpoint_due || error_due) {
        WallTimer timer(use_cuda ? timing.counter(SolverTimings::Readback) : nullptr);
        state = &operations.output_state();
    }
    if (error_due) operations.exact_error(step, *state);
    if (fields_due) {
        WallTimer timer(timing.counter(SolverTimings::Fields));
        operations.write_fields(step, *state, final_step);
    }
    if (checkpoint_due) {
        WallTimer timer(timing.counter(SolverTimings::Recovery));
        if (operations.checkpoint_history) operations.checkpoint_history(recovery.history);
        recovery.save(step.number, step.end, *state, initial);
        recovery.history = implicit::BdfHistory{};
    }
}
}

void integrate_time(const RunSettings& run, const Processes& world, OutputSchedule& output,
                    Recovery& recovery, const std::vector<double>& initial,
                    const StepOperations& operations) {
    std::uint64_t number = recovery.step;
    double time = recovery.time;
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
    bool stopped = false;
    bool previous_first_order = run.first_order_start(time);
    // Возможен запуск с final_time = 0 или продолжение с уже достигнутого времени.
    // В обоих случаях итоговые файлы относятся к загруженному состоянию.
    if (!steady && time >= run.final_time)
        write_step_results({number,time,time,0.,false}, output,recovery,initial,operations,timing,run.use_cuda,true);
    while (steady || time < run.final_time) {
        if (number == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Step number overflow");
        }
        const bool first_order_start = run.first_order_start(time);
        const double cfl_dt = operations.cfl_step(first_order_start, timing);
        // CFL уже собрал потоки начального поля. При продолжении отсчитываем
        // снижение невязки от состояния, загруженного из контрольной точки.
        // При смене схемы пространственная невязка меняется даже при том же поле.
        // Для относительного критерия берём начальную норму соответствующей фазы.
        if (steady && (number == first_step || previous_first_order != first_order_start)) {
            reference = operations.steady_residual();
            criterion = steady_measure(reference);
            if (number == first_step && !first_order_start && criterion <= run.steady_tol) {
                const TimeStep initial_step{number, time, time, 0., false};
                write_step_results(initial_step, output, recovery, initial, operations, timing, run.use_cuda, true);
                break;
            }
        }
        const TimeStep step = choose_time_step(number + 1, time, cfl_dt, run, output);
        if (!operations.advance(step, timing)) continue; // Время и расписание вывода остаются прежними.
        previous_first_order = first_order_start;
        number = step.number;
        time = step.end;
        bool converged = false, exhausted = false;
        if (steady) {
            // Внутренняя сходимость Ньютона относится к одному псевдошагу.
            // Для стационара отдельно проверяем полный отток из каждого объёма.
            criterion = steady_measure(operations.steady_residual());
            // Выход на стационар проверяем по невязке выбранной схемы.
            converged = !first_order_start && criterion <= run.steady_tol;
            exhausted = number - first_step >= run.steady_max_steps;
        }
        // Отчёт относится к только что выполненному шагу. Затем выбираем CFL следующего шага
        // и применяем ручные команды. В контрольную точку попадёт уже готовое к продолжению состояние.
        if (step.reporting) operations.report(step);
        if (operations.accepted) operations.accepted();
        if (run.control_every && (number == first_step + 1 || number % run.control_every == 0) && operations.control)
            stopped = operations.control(number);
        const bool final_step = stopped || converged || exhausted || (!steady && time == run.final_time);
        // Один вызов обслуживает и расписание, и завершение: у зоны multi=1
        // будет ровно один снимок с номером последнего принятого шага.
        write_step_results(step, output, recovery, initial, operations, timing, run.use_cuda, final_step);
        if (steady && !world.rank && (step.reporting || converged || exhausted))
            std::cout << "Steady " << criterion_name << " residual, step " << number << ": " << criterion << '\n';
        if (stopped || converged) break;
        if (exhausted) throw std::runtime_error("Steady: pseudo-step limit reached before convergence");
    }
    timing.report(number, std::cout);
    if (!world.rank) {
        if (stopped) std::cout << "STOP: final fields and recovery saved at step " << number << std::endl;
        else if (steady)
            std::cout << "Steady state converged, " << criterion_name << " spatial residual = " << criterion << '\n';
        std::cout << "Completed " << number << (steady ? " steps. Pseudo-time = " : " steps. Final time = ")
                  << time << '\n';
    }
}
}
