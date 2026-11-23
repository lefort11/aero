#pragma once
#include "../mesh/mesh_index.h"
#include "../parallel/kernel_math.h"

namespace aero {
// Наблюдаем только принятые слои. Три последовательных роста |dp/dt|
// должны сопровождаться заметным изменением самого давления.
struct PressureTrend {
    double pressure, rate, travel; // Последнее p, правая часть dp/dt и сумма |delta p| за серию.
    unsigned rising, calm;
    bool holding;
    double previous_pressure, previous_rate; // Предыдущий принятый слой для сообщения об ограничении.

    // Возвращаем 2 при новом уменьшении, 1 при удержании, 0 при разрешении роста.
    AERO_HOST_DEVICE double observe(double p, double rhs, double dynamic_pressure) {
        const double now = ::fabs(rhs), old = ::fabs(rate);
        const bool growing = now > old * 1.01;
        if (growing) {
            ++rising;
            travel += ::fabs(p-pressure);
            calm = 0;
        } else {
            rising = 0;
            travel = 0.;
            if (now <= old && calm < 3) ++calm;
            if (now > old) calm = 0;
            if (calm == 3) holding = false;
        }
        previous_pressure = pressure;
        previous_rate = rate;
        pressure = p;
        rate = rhs;
        if (rising >= 3 && travel >= .02 * dynamic_pressure) {
            rising = 0;
            travel = 0.;
            holding = true;
            return 2.;
        }
        return holding ? 1. : 0.;
    }
};

// Ноль означает отсутствие отдельного потолка. Удерживаем абсолютный dt
// при росте общего CFL. При восстановлении возвращаем до 25% за принятый шаг.
AERO_HOST_DEVICE inline double pressure_step_cap(double cap, double dt, double own,
                                                 double neighbour, double cfl_step) {
    if (own == 2. || neighbour == 2.) {
        const double reduced = dt * (own == 2. ? .5 : .75);
        return cap > 0. ? ::fmin(cap,reduced) : reduced;
    }
    if (!cap || own || neighbour) return cap;
    const double relaxed = 1.25 * cap;
    return relaxed >= cfl_step ? 0. : relaxed;
}

// В каждом диапазоне из 32 узлов храним только номера пристенной полосы.
// Дальнее поле требует лишь двух чтений каталога и не имеет истории давления.
struct PressureCaps {
    const MeshIndex *blocks, *nodes;
    const double* caps;
    AERO_HOST_DEVICE double apply(MeshIndex node, double dt) const {
        if (!blocks) return dt;
        const auto group = node / 32;
        for (auto i=blocks[group]; i<blocks[group+1]; ++i)
            if (nodes[i]==node && caps[i]>0.) return ::fmin(dt,caps[i]);
        return dt;
    }
};
}
