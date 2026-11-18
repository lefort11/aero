#ifndef AERO_WALL_TIMER_H
#define AERO_WALL_TIMER_H
#include <chrono>

namespace aero {
// Accumulate elapsed wall time for one scope. Disabled counters never read the clock.
class WallTimer {
public:
    using Clock=std::chrono::steady_clock;
    explicit WallTimer(double* seconds) : seconds_(seconds) {
        if (seconds_) start_=Clock::now();
    }
    ~WallTimer() {
        if (seconds_) *seconds_+=std::chrono::duration<double>(Clock::now()-start_).count();
    }
    WallTimer(const WallTimer&)=delete;
    WallTimer& operator=(const WallTimer&)=delete;
private:
    double* seconds_;
    Clock::time_point start_;
};
}
#endif
