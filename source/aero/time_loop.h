#ifndef AERO_TIME_LOOP_H
#define AERO_TIME_LOOP_H
#include "solver_timings.h"
#include <functional>

namespace aero {
class OutputSchedule;
class Recovery;
namespace implicit { struct BdfHistory; }
struct RunSettings;

// Параметры одного шага, общие для всех стадий интегрирования.
struct TimeStep {
    std::uint64_t number; // Номер выполняемого шага, начиная с единицы.
    double start;         // Физическое время в начале шага.
    double end;           // Физическое время, которое примет решатель после шага.
    double dt;            // Длина шага для обновления полей на стадиях интегратора.
    bool reporting;       // Нужно ли собрать данные для отчёта за этот шаг.
};

// Операции решателя, которыми пользуется общий цикл по времени.
struct StepOperations {
    // Вычисляет ограничение CFL и оставляет потоки готовыми для первой стадии.
    std::function<double(bool first_order_start, SolverTimings&)> cfl_step;
    // Использует готовые физические потоки для проверки выхода на стационар.
    std::function<double()> steady_residual;
    // Выполняет все стадии шага и обмен значениями в граничных узлах подобластей.
    std::function<bool(const TimeStep&, SolverTimings&)> advance;
    std::function<bool(std::uint64_t)> control; // Применяет команды после принятого шага. STOP запрашивает итоговый вывод.
    std::function<Recovery*()> stop_recovery; // Создаёт контрольную точку даже при отключённом расписании.
    std::function<void()> accepted; // После отчёта можно повысить CFL следующего шага.
    // Даёт текущее состояние в памяти CPU для вывода полей и контрольной точки.
    std::function<const std::vector<double>&()> output_state;
    std::function<void(const TimeStep&, const std::vector<double>&)> exact_error;
    // Выгружает предыдущий принятый слой только при записи контрольной точки BDF2.
    std::function<void(implicit::BdfHistory&)> checkpoint_history;
    // Печатает сводку, собранную при выполнении шага.
    std::function<void(const TimeStep&)> report;
    // Записывает из одного снимка все зоны, которым подошёл срок вывода.
    // Последний аргумент запрашивает запись всех заданных зон при выходе на стационар.
    std::function<void(const TimeStep&, const std::vector<double>&, bool)> write_fields;
};

// Ведёт расчёт до конечного времени от начального состояния или контрольной точки.
void integrate_time(const RunSettings& run, const Processes& world, OutputSchedule& output,
                    Recovery* recovery, const std::vector<double>& initial, const StepOperations& operations);
}
#endif
