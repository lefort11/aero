#include "flowreport.h"
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
// Накапливает взвешенные квадраты приращений в масштабе наибольшего из них, чтобы избежать переполнения.
void FlowSummary::add_squares(long double scale, long double weight) {
    if (scale == 0)
        return;
    if (scale > rate_scale) {
        // При появлении большего масштаба сначала пересчитываем накопленную сумму в его единицы.
        const long double ratio = rate_scale / scale;
        scaled_squares = weight + scaled_squares * ratio * ratio;
        rate_scale = scale;
    } else {
        const long double ratio = scale / rate_scale;
        scaled_squares += weight * ratio * ratio;
    }
}
void FlowSummary::merge(const FlowSummary& other, unsigned fields) {
    add_squares(other.rate_scale, other.scaled_squares);
    volume += other.volume;
    for (unsigned k = 0; k < fields; ++k) {
        field[k].low = std::min(field[k].low, other.field[k].low);
        field[k].high = std::max(field[k].high, other.field[k].high);
        field[k].integral += other.field[k].integral;
    }
}
namespace {
std::string cell(long double value) {
    std::ostringstream s;
    s.imbue(std::locale::classic());
    const auto magnitude = std::abs(value);
    if (magnitude >= 1e8L || (magnitude > 0 && magnitude < 1e-4L))
        s << std::scientific;
    else
        s << std::fixed;
    s << std::setprecision(6) << value;
    return s.str();
}
}

FlowReport::FlowReport(std::uint64_t every, double gamma, unsigned dimension, std::size_t nodes,
                       const double* volumes)
    : every_(every), captured_step_(0), gamma_(gamma), dimension_(dimension), total_volume_(0) {
    if (!every)
        return;
    if (dimension < 1 || dimension > 3 || !(gamma > 1 && std::isfinite(gamma)) || !nodes || !volumes ||
        nodes > volumes_.max_size() / (dimension + 2))
        throw std::invalid_argument("flowreport: invalid grid or gas");
    volumes_.assign(volumes, volumes + nodes);
    for (double v : volumes_) {
        if (!(v > 0 && std::isfinite(v)))
            throw std::invalid_argument("flowreport: invalid control volume");
        total_volume_ += v;
    }
    if (!std::isfinite(total_volume_))
        throw std::overflow_error("flowreport: total volume overflow");
    previous_energy_.resize(nodes);
}

bool FlowReport::due(std::uint64_t step) const {
    return due(step, every_);
}
bool FlowReport::due(std::uint64_t step, std::uint64_t every) {
    return every && step && step % every == 0;
}
void FlowReport::capture(std::uint64_t step, const std::vector<double>& previous) {
    if (!due(step))
        return;
    if (previous.size() != volumes_.size() * (dimension_ + 2))
        throw std::invalid_argument("flowreport: previous field size");
    capture_values(step, previous.data());
}
void FlowReport::capture_values(std::uint64_t step, const double* u) {
    captured_step_ = 0;
    const unsigned width = dimension_ + 2;
    parallel_for(volumes_.size(), [&](std::size_t i) {
        const double e = u[i * width + width - 1];
        if (!std::isfinite(e))
            throw std::domain_error("flowreport: invalid previous energy at node " + std::to_string(i));
        previous_energy_[i] = e;
    });
    captured_step_ = step;
}
void FlowReport::write(std::uint64_t step, double time, double cfl, double dt,
                       const std::vector<double>& current, std::ostream& out) {
    if (!due(step))
        return;
    if (current.size() != volumes_.size() * (dimension_ + 2))
        throw std::invalid_argument("flowreport: current field size");
    write_values(step, time, cfl, dt, current.data(), out);
}
void FlowReport::write_values(std::uint64_t step, double time, double cfl, double dt, const double* u,
                              std::ostream& out) {
    const auto total = measure_values(step, u);
    write_flow_summary(total, step, time, cfl, dt, dimension_, out);
    captured_step_ = 0;
}
FlowSummary FlowReport::measure(std::uint64_t step, const std::vector<double>& current) {
    if (!due(step) || current.size() != volumes_.size() * (dimension_ + 2))
        throw std::invalid_argument("flowreport: unexpected step or field size");
    const auto result = measure_values(step, current.data());
    captured_step_ = 0;
    return result;
}
FlowSummary FlowReport::measure_values(std::uint64_t step, const double* u) {
    if (captured_step_ != step)
        throw std::logic_error("flowreport: capture the state before this step");
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
                EulerState state;
                if (euler_state(dimension_, q, gamma_, &state) != EulerStatus::Success)
                    throw std::domain_error("flowreport: invalid state at node " + std::to_string(i));
                double values[6] = {state.density};
                std::copy(state.velocity, state.velocity + dimension_, values + 1);
                values[width - 1] = state.pressure;
                values[width] = q[width - 1];
                for (unsigned k = 0; k < fields; ++k) {
                    FlowRange& s = block.field[k];
                    s.low = std::min(s.low, values[k]);
                    s.high = std::max(s.high, values[k]);
                    s.integral += static_cast<long double>(volumes_[i]) * values[k];
                }
                // Сравниваем энергию с началом того же принятого шага. Вес узла равен его контрольному объёму.
                const long double delta = static_cast<long double>(q[width - 1]) - previous_energy_[i];
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
void write_flow_summary(const FlowSummary& total, std::uint64_t step, double time, double cfl, double dt,
                        unsigned dimension, std::ostream& out, const double* newton_relative) {
    if (!(dt > 0 && std::isfinite(dt) && cfl > 0 && std::isfinite(cfl) && total.volume > 0))
        throw std::invalid_argument("flowreport: expected positive finite CFL, dt and volume");
    const unsigned width = dimension + 2, fields = width + 1;
    const long double rms = total.rate_scale * std::sqrt(total.scaled_squares / total.volume) / dt;
    if (!std::isfinite(rms))
        throw std::overflow_error("flowreport: energy residual overflow");
    std::ostringstream table;
    table.imbue(std::locale::classic());
    // Время компьютера в момент формирования отчёта, отдельно от физического времени t.
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    if (localtime_s(&local, &now) == 0)
#else
    if (localtime_r(&now, &local))
#endif
        table << "Local time = " << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
    table << '\n';
    table << "Flow report    Step = " << step << "    t = " << std::setprecision(12) << time
          << "    CFL = " << std::setprecision(6) << cfl << "    dt = " << std::scientific << dt << '\n';
    if (newton_relative)
        table << "Newton relative residual, step " << step << ": " << *newton_relative << '\n';
    table << std::left << std::setw(9) << "Variable" << std::right << std::setw(15) << "Minimum"
          << std::setw(15) << "Maximum" << std::setw(20) << "Domain integral" << '\n';
    const char* names[] = {"rho", "ux", "uy", "uz", "p", "E"};
    for (unsigned k = 0; k < fields; ++k) {
        const FlowRange& s = total.field[k];
        if (!std::isfinite(s.integral))
            throw std::overflow_error("flowreport: domain integral overflow");
        const char* name = k <= dimension ? names[k] : names[k == width ? 5 : 4];
        table << std::left << std::setw(9) << name << std::right << std::setw(15) << cell(s.low)
              << std::setw(15) << cell(s.high) << std::setw(20) << cell(s.integral) << '\n';
    }
    table << "\nEnergy residual (volume-weighted RMS of dE/dt) = " << rms << "\n";
    const auto text = table.str();
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.flush();
    if (!out)
        throw std::runtime_error("flowreport: output failed");
}
}
