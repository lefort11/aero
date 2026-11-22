#pragma once
#include "../parallel/wall_timer.h"
#include <cstdint>

namespace aero {
namespace implicit {
// Счётчики принадлежат отчёту текущего расчёта. Пустое представление выполняет
// только саму операцию: обычному запуску не нужны ни часы, ни ожидание GPU.
struct PhaseTiming {
    enum Phase { Matrix, ViscousMatrix, PreconditionerSetup, Solve, Prediction, Count };
    double* seconds;
    std::uint64_t* calls;
    void (*wait)(); // Завершить работу очереди Kokkos; для замера CPU всегда nullptr.

    template<class Work> void measure(Phase phase, const Work& work) const {
        if (!seconds) { work(); return; }
        // Начальный барьер исключает незавершённую работу предыдущей фазы.
        // Конечный включает вычисления на карте, а не только отправку команд.
        if (wait) wait();
        ++calls[phase];
        WallTimer timer(seconds+phase);
        work();
        if (wait) wait();
    }
};
}
}
