#pragma once

namespace aero {
// Баланс за один шаг: leading*(U - U_n) - history*(U_n - U_{n-1}).
// Газ и рабочая вязкость SA должны использовать одни и те же слои по времени.
struct BdfWeights {
    double leading, history;

    static BdfWeights for_step(bool second_order, double dt, double previous_dt) {
        // Первый шаг BDF2, в том числе после старого recovery без истории, выполняется по BDF1.
        const double ratio = second_order && previous_dt > 0. ? dt / previous_dt : 0.;
        const double fraction = ratio / (1. + ratio);
        return {1. + fraction, ratio * fraction};
    }
};
}
