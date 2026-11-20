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
    void configure(const Stabilization& options, double value) {
        options_ = options;
        restore(value, 0);
    }
    void restore(double value, std::uint64_t successes) {
        if (!(value > 0.) || !std::isfinite(value) || (options_.active() && (value < options_.minimum || value > options_.maximum)) || successes >= options_.interval)
            throw std::runtime_error("Checkpoint CFL outside configured limits");
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
