#include "report.h"
#include "../implicit/linear_iteration_report.h"
#include "flow_values.h"
#include <ctime>
#include "../eulersolver/euler.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace aero {
// Единый формат календарного времени для отчёта и прогноза завершения расчёта.
std::string local_date_time(std::time_t timestamp) {
    std::tm local{};
#ifdef _WIN32
    if (localtime_s(&local, &timestamp) == 0)
#else
    if (localtime_r(&timestamp, &local))
#endif
    {
        std::ostringstream out;
        out << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
        return out.str();
    }
    return "unavailable";
}

// Накапливает взвешенные квадраты приращений в масштабе наибольшего из них, чтобы избежать переполнения.
void FlowSummary::add_squares(long double scale, long double weight) {
    if (scale != 0)
        diagnostic::add_scaled_square(scale, weight, rate_scale, scaled_squares);
}
// Объединяет диапазоны и масштабированные нормы локальных отчётных данных.
void FlowSummary::merge(const FlowSummary& other, unsigned fields) {
    add_squares(other.rate_scale, other.scaled_squares);
    volume += other.volume;
    for (unsigned k = 0; k < fields; ++k) {
        field[k].low = std::min(field[k].low, other.field[k].low);
        field[k].high = std::max(field[k].high, other.field[k].high);
        field[k].integral += other.field[k].integral;
    }
}
// Настраивает период отчётов, параметры газа и объёмные веса узлов.
Report::Report(std::uint64_t every, double gamma, unsigned dimension, std::size_t nodes,
                       const double* volumes, const LinearBackground& background)
    : every_(every), captured_step_(0), background_(background), gamma_(gamma), dimension_(dimension), total_volume_(0) {
    if (!every)
        return;
    if (dimension < 1 || dimension > 3 || !(gamma > 1 && std::isfinite(gamma)) || !nodes || !volumes ||
        nodes > volumes_.max_size() / (dimension + 2))
        throw std::invalid_argument("report: invalid grid or gas");
    volumes_.assign(volumes, volumes + nodes);
    for (double v : volumes_) {
        if (!(v > 0 && std::isfinite(v)))
            throw std::invalid_argument("report: invalid control volume");
        total_volume_ += v;
    }
    if (!std::isfinite(total_volume_))
        throw std::overflow_error("report: total volume overflow");
}

// Проверяет наступление отчётного шага или срока.
bool Report::due(std::uint64_t step) const {
    return due(step, every_);
}
// Проверяет наступление отчётного шага или срока.
bool Report::due(std::uint64_t step, std::uint64_t every) {
    return every && step && step % every == 0;
}
// Сохраняет начальное поле интервала для последующей оценки изменения.
void Report::capture(std::uint64_t step, const std::vector<double>& previous) {
    if (!due(step))
        return;
    if (previous.size() != volumes_.size() * (dimension_ + 2))
        throw std::invalid_argument("report: previous field size");
    capture_values(step, previous.data());
}
// Сохраняет значения начала отчётного интервала для измерения изменений.
void Report::capture_values(std::uint64_t step, const double* u) {
    captured_step_ = 0;
    // На одной GPU steady берёт энергию из U_n интегратора. Буфер хоста
    // нужен только тем режимам, которые действительно вызывают capture.
    if (previous_energy_.empty()) previous_energy_.resize(volumes_.size());
    const unsigned width = dimension_ + 2;
    parallel_for(volumes_.size(), [&](std::size_t i) {
        const auto* q = u+i*width;
        const double e = background_.active() ? background_.energy(dimension_,q,gamma_) : q[width-1];
        if (!std::isfinite(e))
            throw std::domain_error("report: invalid previous energy at node " + std::to_string(i));
        previous_energy_[i] = e;
    });
    captured_step_ = step;
}
// Измеряет и печатает отчёт о текущем состоянии расчёта.
void Report::write(std::uint64_t step, double time, double cfl, double dt,
                       const std::vector<double>& current, std::ostream& out) {
    if (!due(step))
        return;
    if (current.size() != volumes_.size() * (dimension_ + 2))
        throw std::invalid_argument("report: current field size");
    write_values(step, time, cfl, dt, current.data(), out);
}
// Печатает отчётные показатели переданного поля.
void Report::write_values(std::uint64_t step, double time, double cfl, double dt, const double* u,
                              std::ostream& out) {
    const auto total = measure_values(step, u);
    ReportDetails details;
    details.linear = background_.active();
    write_flow_summary(total, step, time, cfl, dt, dimension_, out, details);
    captured_step_ = 0;
}
// Вычисляет отчётные показатели принятого поля.
FlowSummary Report::measure(std::uint64_t step, const std::vector<double>& current) {
    if (!due(step) || current.size() != volumes_.size() * (dimension_ + 2))
        throw std::invalid_argument("report: unexpected step or field size");
    const auto result = measure_values(step, current.data());
    captured_step_ = 0;
    return result;
}
// Вычисляет диапазоны и объёмные нормы по текущему полю.
FlowSummary Report::measure_values(std::uint64_t step, const double* u) {
    if (captured_step_ != step)
        throw std::logic_error("report: capture the state before this step");
    const std::size_t chunk = 256;
    const unsigned width = dimension_ + 2, fields = width + 1;
    std::vector<FlowSummary> blocks((volumes_.size() + chunk - 1) / chunk);
    parallel_for(
        blocks.size(),
        [&](std::size_t b) {
            FlowSummary& block = blocks[b];
            const auto end = std::min((b + 1) * chunk, volumes_.size());
            for (std::size_t i = b * chunk; i < end; ++i) {
                const double* q = u + i * width;
                // Одинаковое представление для таблицы и снимков на диске.
                const auto sampled = output_values(dimension_, q, gamma_, i, background_);
                for (unsigned k = 0; k < fields; ++k) {
                    const double value = sampled[diagnostic::report_column(dimension_, k)];
                    FlowRange& s = block.field[k];
                    diagnostic::accumulate_column(value,volumes_[i],s.low,s.high,s.integral);
                }
                // Сравниваем энергию с началом того же принятого шага. Вес узла равен его контрольному объёму.
                const long double delta = static_cast<long double>(sampled[diagnostic::Energy]) - previous_energy_[i];
                block.add_squares(std::abs(delta), volumes_[i]);
            }
        },
        2);
    FlowSummary total;
    for (const auto& block : blocks)
        total.merge(block, fields);
    total.volume = total_volume_;
    return total;
}
// Печатает диапазоны величин и норму изменения поля.
void write_flow_summary(const FlowSummary& total, std::uint64_t step, double time, double cfl, double dt,
                        unsigned dimension, std::ostream& out, const ReportDetails& details) {
    if (!(dt > 0 && std::isfinite(dt) && cfl > 0 && std::isfinite(cfl) && total.volume > 0))
        throw std::invalid_argument("report: expected positive finite CFL, dt and volume");
    const unsigned width = dimension + 2, fields = width + 1;
    const long double rms = total.rate_scale * std::sqrt(total.scaled_squares / total.volume) / (details.max_local_step ? 1. : dt);
    if (!std::isfinite(rms))
        throw std::overflow_error("report: energy residual overflow");
    std::ostringstream table;
    table.imbue(std::locale::classic());
    // Время компьютера в момент формирования отчёта, отдельно от физического времени t.
    const std::time_t now = std::time(nullptr);
    std::ostringstream title;
    title << "STEP " << step;
    if (!details.scheme.empty()) title << "  |  " << details.scheme;
    table << std::left << std::setw(53) << title.str();
    table << local_date_time(now);
    table << '\n' << std::string(72, '-') << '\n';
    if (details.max_local_step)
        table << "dt_min = " << std::scientific << std::setprecision(6) << dt
              << "    dt_max = " << details.max_local_step;
    else
        table << (details.steady_criterion.empty() ? "t = " : "Pseudo-time = ") << std::setprecision(12) << time
          << "    dt = " << std::scientific << std::setprecision(6) << dt;
    table << "    CFL = " << std::defaultfloat << cfl << "\n\n" << std::scientific;
    if (details.local_rate_control)
        table << "LTS residual limit  nodes = " << details.rate_limited
              << "    nodes with stored cap = " << details.rate_capped << '\n';
    if (details.step_spread_limit > 0.)
        table << "LTS step ratio limit = " << details.step_spread_limit << '\n';
    if (details.rate_probe != ~std::uint64_t(0))
        table << "LTS probe       node = " << details.rate_probe << "    dt = " << details.probe[0]
              << "    gas rate = " << details.probe[1] << "    SA rate = " << details.probe[2]
              << "    cap/CFL step = " << details.probe[3] << '\n';
    for (unsigned k=0; k<2; ++k)
        if (details.growth_nodes[k] != ~std::uint64_t(0))
            table << "LTS probe stencil  " << (k ? "SA" : "gas") << " response at node = " << details.growth_nodes[k] << '\n';
    if (details.edge_ratio_limit > 0.)
        table << "LTS edge ratio  " << details.edge_ratio_before << " -> " << details.edge_ratio_after
              << "    limit = " << details.edge_ratio_limit << '\n';
    if (details.max_iterations)
        table << "Newton         " << details.iterations << '/' << details.max_iterations
              << " iterations    relative residual = " << details.newton_residual
              << (details.newton_residual <= details.newton_tol ? "  [OK]\n" : "  [ITERATION LIMIT]\n");
    if (details.solver_max_iter) {
        table << "BiCGSTAB       ";
        implicit::write_linear_iterations(table, details.correction_iterations, details.solver_max_iter);
        table << '\n';
    }
    if (!details.steady_criterion.empty())
        table << "Steady         " << details.steady_criterion << " residual = " << details.steady_residual
              << "    target = " << details.steady_tol << '\n';
    if (details.steady_criterion == "absolute")
        table << "               relative residual = " << details.relative_steady_residual << '\n';
    table << "Energy         " << (details.max_local_step ? "RMS(delta E" : "RMS(dE")
          << (details.linear ? "'" : "") << (details.max_local_step ? ") = " : "/dt) = ") << rms << '\n';
    if (!details.stabilization.empty())
        table << "Stabilization  " << details.stabilization << "    failed since last report = " << details.failed
              << "    first-order now = " << details.first_order << '\n';
    table << '\n' << std::left << std::setw(9) << "Field" << std::right << std::setw(15) << "Minimum"
          << std::setw(15) << "Maximum" << std::setw(20) << "Domain integral" << '\n';
    const char* names[] = {"rho", "ux", "uy", "uz", "p", "E"};
    for (unsigned k = 0; k < fields; ++k) {
        const FlowRange& s = total.field[k];
        if (!std::isfinite(s.integral))
            throw std::overflow_error("report: domain integral overflow");
        const char* name = k <= dimension ? names[k] : names[k == width ? 5 : 4];
        table << std::left << std::setw(9) << (std::string(name) + (details.linear ? "'" : "")) << std::right << std::setw(15) << s.low
              << std::setw(15) << s.high << std::setw(20) << s.integral << '\n';
    }
    table << '\n';
    const auto text = table.str();
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.flush();
    if (!out)
        throw std::runtime_error("report: output failed");
}
}
