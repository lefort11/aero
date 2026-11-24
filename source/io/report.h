#ifndef AERO_IO_REPORT_H
#define AERO_IO_REPORT_H

#include "../eulersolver/linear_euler.h"
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>
#include <string>

namespace aero {
// Единый формат календарного времени для отчёта и прогноза завершения расчёта.
std::string local_date_time(std::time_t timestamp);
class ExplicitIntegrator;
// Минимум, максимум и объёмный интеграл физической величины.
struct FlowRange {
    double low = std::numeric_limits<double>::max(), high = std::numeric_limits<double>::lowest();
    long double integral = 0;
};
// Сводные диапазоны полей и масштабированная норма их изменения.
struct FlowSummary {
    FlowRange field[6];
    // Сумма квадратов хранится в единицах rate_scale² с весами объёмов.
    long double rate_scale = 0, scaled_squares = 0, volume = 0;
    // Накапливает взвешенные квадраты приращений в масштабе наибольшего из них, чтобы избежать переполнения.
    void add_squares(long double scale, long double weight);
    // Объединяет диапазоны и масштабированные нормы локальных отчётных данных.
    void merge(const FlowSummary& other, unsigned fields);
};
// Сведения о принятом шаге. Нулевой предел Ньютона означает явную схему.
struct ReportDetails {
    std::string scheme, stabilization, steady_criterion;
    double max_local_step = 0.; // Ненулевой при локальном установлении; dt отчёта хранит минимум.
    // Наибольшее на сетке отношение шагов соседей до и после согласования.
    bool local_rate_control = false;
    double step_spread_limit = 0.; // Общий предел dt_max/dt_min, включая заданный вручную.
    std::uint64_t rate_limited = 0, rate_capped = 0, rate_probe = ~std::uint64_t(0);
    double probe[4] = {}; // Шаг, невязки газа и SA, разрешённая доля местного CFL.
    std::uint64_t growth_nodes[2] = {~std::uint64_t(0),~std::uint64_t(0)}; // Один узел с ростом газа и один с ростом SA в шаблоне наблюдения.
    double edge_ratio_before = 0., edge_ratio_after = 0., edge_ratio_limit = 0.;
    bool linear = false;
    std::vector<unsigned> correction_iterations; // В порядке поправок текущего шага.
    unsigned solver_max_iter = 0;
    unsigned iterations = 0, max_iterations = 0;
    double relative_steady_residual = 0.;
    double newton_residual = 0., newton_tol = 0., steady_residual = 0., steady_tol = 0.;
    std::uint64_t failed = 0, first_order = 0;
};
// Печатает диапазоны величин и норму изменения поля.
void write_flow_summary(const FlowSummary&, std::uint64_t step, double time, double cfl, double dt,
                        unsigned dimension, std::ostream& out, const ReportDetails& details = {});

// Один объект обслуживает один расчёт. Объёмы относятся к независимым расчётным узлам.
class Report {
public:
    // Настраивает период отчётов, параметры газа и объёмные веса узлов.
    Report(std::uint64_t every, double gamma, unsigned dimension, std::size_t nodes,
               const double* volumes, const LinearBackground& background = {});
    // Проверяет наступление отчётного шага или срока.
    static bool due(std::uint64_t step, std::uint64_t every);
    // Проверяет наступление отчётного шага или срока.
    bool due(std::uint64_t step) const;

    // До и после шага передавайте один номер предстоящего шага, начиная с единицы.
    // При повторной попытке шага исходное поле можно сохранить ещё раз.
    void capture(std::uint64_t step, const std::vector<double>& previous);
    // Вычисляет отчётные показатели принятого поля.
    FlowSummary measure(std::uint64_t step, const std::vector<double>& current);
    // Измеряет и печатает отчёт о текущем состоянии расчёта.
    void write(std::uint64_t step, double time, double cfl, double dt, const std::vector<double>& current,
               std::ostream& out = std::cout);
    // Обёртки интегратора проверяют срок отчёта до обращения к полю на CPU или GPU.
    void capture(std::uint64_t step, const ExplicitIntegrator& solver);
    // Измеряет и печатает отчёт о текущем состоянии расчёта.
    void write(std::uint64_t step, double time, double cfl, double dt, const ExplicitIntegrator& solver,
               std::ostream& out = std::cout);

private:
    // Вычисляет диапазоны и объёмные нормы по текущему полю.
    FlowSummary measure_values(std::uint64_t step, const double* u);
    // Сохраняет значения начала отчётного интервала для измерения изменений.
    void capture_values(std::uint64_t step, const double* u);
    // Печатает отчётные показатели переданного поля.
    void write_values(std::uint64_t step, double time, double cfl, double dt, const double* u,
                      std::ostream& out);
    // Период отчётов и номер шага, для которого сохранена энергия перед обновлением.
    std::uint64_t every_, captured_step_;
    LinearBackground background_; // Задаёт смысл текущего поля и его последней компоненты.
    double gamma_;
    unsigned dimension_;
    // Объёмы постоянны; энергию запоминаем перед шагом, для которого запрошен отчёт.
    std::vector<double> volumes_, previous_energy_;
    // Суммарный объём для нормировки среднеквадратичного изменения энергии.
    long double total_volume_;
};
}
#endif
