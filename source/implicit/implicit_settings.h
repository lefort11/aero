#ifndef AERO_IMPLICIT_SETTINGS_H
#define AERO_IMPLICIT_SETTINGS_H
namespace aero {
namespace implicit {
// Три независимых настройки линейной системы. Их передают вместе, чтобы
// порядок логических аргументов не мог случайно переключить численный метод.
struct MatrixSettings {
    bool roe = false;                 // Поток Роу вместо LF для матрицы поправки.
    bool simplified_jacobian = false; // Зафиксировать матрицу диссипации при дифференцировании.
    bool csgs = false;                // Симметричные проходы по соседям вместо независимых узлов BJ.
};
struct NewtonSettings : MatrixSettings {
    // Ноль — обновлять матрицу на каждой поправке Ньютона. N>0 — хранить
    // пространственные производные в течение N принятых шагов по времени.
    unsigned jacobian_update_every = 0;
    bool second_order = false;      // BDF2 использует два последних принятых состояния.
    double linear_relative = 1.e-8; // Норма линейной невязки относительно нормы текущей правой части.
    double newton_relative = 1.e-6; // Норма невязки BDF относительно начального значения на этом шаге.
    unsigned max_iterations = 20;   // Предельное число поправок к состоянию за один шаг по времени.
};
}
}
#endif
