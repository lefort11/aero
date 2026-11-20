#ifndef AERO_STABILIZATION_H
#define AERO_STABILIZATION_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace aero {
// Настройки повторного шага. Файл cfl.txt нужен только для активного режима.
struct Stabilization {
    std::string mode = "off";
    double factor = 2., minimum = 0., maximum = 1.;
    std::uint64_t interval = 1;
    // Одни и те же ограничения действуют при старте, ручной команде и восстановлении.
    void validate(double cfl, const std::string& where) const {
        if (!std::isfinite(cfl) || cfl <= 0.)
            throw std::invalid_argument(where + ": cfl must be finite and positive");
        if (!active()) return;
        if (!interval || !std::isfinite(factor) || factor <= 1.)
            throw std::invalid_argument(where + ": require cfl_num_steps >= 1 and cfl_coeff > 1");
        if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum <= 0. ||
            cfl < minimum || cfl > maximum)
            throw std::invalid_argument(where + ": require 0 < cfl_min <= cfl <= cfl_max");
    }
    bool active() const { return mode != "off"; }
    bool local_order() const { return mode == "low" || mode == "on"; }
};

// Счётчик ведётся по принятым шагам. Отказ на любой стадии начинает отсчёт заново.
class AdaptiveCourant {
    Stabilization options_;
    double value_;
    std::uint64_t successes_ = 0;
public:
    AdaptiveCourant(const Stabilization& options, double initial) : options_(options), value_(initial) {}
    double value() const { return value_; }
    std::uint64_t successes() const { return successes_; }
    // После ручной команды отсчёт начинается заново. Из recovery возвращается сохранённый счётчик.
    void configure(const Stabilization& options, double value, std::uint64_t successes = 0) {
        options.validate(value, "CFL");
        if (successes >= options.interval)
            throw std::invalid_argument("CFL success count must be less than cfl_num_steps");
        options_ = options;
        value_ = value;
        successes_ = successes;
    }
    void reject() {
        const double next = value_ / options_.factor;
        if (next < options_.minimum)
            throw std::runtime_error("Nonphysical Euler state: next CFL is below cfl_min");
        if (!(next > 0. && next < value_)) throw std::runtime_error("Cannot reduce CFL further");
        value_ = next;
        successes_ = 0;
    }
    bool accept() {
        if (value_ >= options_.maximum || ++successes_ < options_.interval) return false;
        value_ = std::min(options_.maximum, value_ * options_.factor);
        successes_ = 0;
        return true;
    }
};
}
#endif
