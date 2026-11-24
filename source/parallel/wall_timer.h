#ifndef AERO_WALL_TIMER_H
#define AERO_WALL_TIMER_H
#include <chrono>

namespace aero {
// Добавляет в счётчик время жизни объекта. При отключённом счётчике часы не опрашиваются.
class WallTimer {
public:
    using Clock = std::chrono::steady_clock;
    // Начинает измерение интервала для добавления к указанному счётчику времени.
    explicit WallTimer(double* seconds) : seconds_(seconds) {
        if (seconds_)
            start_ = Clock::now();
    }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~WallTimer() {
        if (seconds_)
            *seconds_ += std::chrono::duration<double>(Clock::now() - start_).count();
    }
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    WallTimer(const WallTimer&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    WallTimer& operator=(const WallTimer&) = delete;

private:
    double* seconds_;
    Clock::time_point start_;
};
}
#endif
