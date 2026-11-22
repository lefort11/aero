#ifndef AERO_IMPLICIT_SETTINGS_H
#define AERO_IMPLICIT_SETTINGS_H
namespace aero {
namespace implicit {
constexpr unsigned default_solver_max_iter = 200;

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
    // При упрощённом Ньютоне все поправки используют матрицу первой линейной
    // системы шага. Положительный интервал также разрешает хранить её между шагами.
    bool simplified_newton() const { return jacobian_update_every != 0; }
    void set_simplified_newton(bool enabled) {
        if (!enabled) jacobian_update_every = 0;
        else if (!jacobian_update_every) jacobian_update_every = 1;
    }
    bool second_order = false;      // BDF2 использует два последних принятых состояния.
    unsigned linear_max_iterations = default_solver_max_iter; // Предел итераций BiCGSTAB для одной правой части.
    double linear_relative = 1.e-8; // Норма линейной невязки относительно нормы текущей правой части.
    double newton_relative = 1.e-6; // Норма невязки BDF относительно начального значения на этом шаге.
    unsigned max_iterations = 20;   // Предельное число поправок к состоянию за один шаг по времени.
};
}
}
#endif
