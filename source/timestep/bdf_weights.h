#pragma once

namespace aero {
// Баланс за один шаг: leading*(U - U_n) - history*(U_n - U_{n-1}).
// Газ и рабочая вязкость SA должны использовать одни и те же слои по времени.
struct BdfWeights {
    // Коэффициенты текущего слоя и истории BDF.
    double leading, history;

    // Вычисляет коэффициенты BDF для текущего шага и доступной истории.
    static BdfWeights for_step(bool second_order, double dt, double previous_dt) {
        // Первый шаг BDF2, в том числе после старого recovery без истории, выполняется по BDF1.
        const double ratio = second_order && previous_dt > 0. ? dt / previous_dt : 0.;
        // При одинаковых шагах получаются коэффициенты BDF2 3/2 и 1/2; при отсутствии истории — BDF1.
        const double fraction = ratio / (1. + ratio);
        return {1. + fraction, ratio * fraction};
    }
};
}
