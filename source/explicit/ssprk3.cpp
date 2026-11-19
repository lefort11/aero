#include "explicit_stage.h"

namespace aero {
namespace explicit_detail {
Stages ssprk3_stages() {
    // В каждой строке указаны время невязки в долях dt и два веса. E — шаг Эйлера из состояния этой строки.
    static const Stage stages[] = {
        first_euler_stage(),     // R(t_n, U_n)         → Y1 = E(U_n)
        {1.0, 3.0 / 4, 1.0 / 4}, // R(t_n + dt, Y1)     → Y2 = 3/4 U_n + 1/4 E(Y1)
        {0.5, 1.0 / 3, 2.0 / 3}  // R(t_n + dt/2, Y2)   → U_next = 1/3 U_n + 2/3 E(Y2)
    };
    return {stages, 3};
}
}
}
