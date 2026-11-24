#ifndef AERO_EULER_CHARACTERISTIC_H
#define AERO_EULER_CHARACTERISTIC_H
#include "../parallel/kernel_math.h"
#include "euler.h"
namespace aero {
namespace euler_detail {
// Первая и последняя волны — звук, вторая переносит энтропию. Поперечные волны нужны реконструкции.
// Возвращаем также приращение импульса относительно средней скорости для их последующей проекции.
KOKKOS_FORCEINLINE_FUNCTION void project_normal_waves(unsigned d, const double* velocity, const double* n,
                                                  double gamma, double kinetic, double sound_speed,
                                                  double inverse_sound, const double* u,
                                                  double* momentum, double* result) {
    for (unsigned k = 0; k < d; ++k)
        momentum[k] = u[k + 1] - velocity[k] * u[0];
    // Линеаризуем давление в базисе Роу. Это приращение может быть отрицательным.
    const double pressure =
        (gamma - 1) * (u[d + 1] - dot(d, velocity, u + 1) + kinetic * u[0]);
    // Все наклоны ребра имеют общую скорость звука. При повторных проекциях вызывающая
    // функция подаёт готовое 1/c. Для разового вызова выполняем обычное деление.
    const double normal_momentum = euler_detail::dot(d, momentum, n);
    const double sum = inverse_sound ? (pressure * inverse_sound) * inverse_sound
                                     : (pressure / sound_speed) / sound_speed;
    const double difference = inverse_sound ? normal_momentum * inverse_sound : normal_momentum / sound_speed;
    // Нормальная составляющая даёт пару акустических волн. Оставшаяся часть плотности переходит в энтропийную волну.
    result[0] = 0.5 * (sum - difference);
    result[1] = u[0] - sum;
    result[d + 1] = 0.5 * (sum + difference);
}

// Проецирует вектор на волны, используя заранее проверенные базис Роу и единичную нормаль.
template <unsigned fixed_dimension = 0, bool full_basis = true, class Basis>
KOKKOS_FORCEINLINE_FUNCTION EulerStatus project_characteristic(const Basis& basis, const double* n,
                                                           double gamma, double kinetic, const double* u,
                                                           double* w, double inverse_sound = 0) {
    const unsigned d = fixed_dimension ? fixed_dimension : basis.dimension;
    double momentum[3] = {}, result[5] = {};
    project_normal_waves(d, basis.velocity, n, gamma, kinetic, basis.sound_speed, inverse_sound, u, momentum, result);
    for (unsigned t = 0; full_basis && t + 1 < d; ++t)
        for (unsigned k = 0; k < d; ++k)
            result[2 + t] += basis.coefficient(k + 1, 2 + t) * momentum[k];
    for (unsigned c = 0; c < d + 2; ++c)
        if (!euler_detail::finite(result[c]))
            return EulerStatus::NonFinite;
    for (unsigned c = 0; c < d + 2; ++c)
        w[c] = result[c];
    return EulerStatus::Success;
}

// Восстанавливает U = R·W. В расчёте рёбер build_roe_basis уже проверил матрицу.
// Если вызывающий код передал неконечный элемент, он попадёт в результат и будет обнаружен до записи.
template <unsigned fixed_width = 0, class Basis>
KOKKOS_FORCEINLINE_FUNCTION EulerStatus expand_characteristic(unsigned width, const Basis& basis,
                                                           const double* waves, double* state) {
    const unsigned count = fixed_width ? fixed_width : width;
    double result[5] = {};
    for (unsigned r = 0; r < count; ++r) {
        // Элементы одной строки R лежат подряд. Порядок сложения волн тот же для CPU и GPU.
        for (unsigned c = 0; c < count; ++c)
            result[r] += basis.coefficient(r, c) * waves[c];
        if (!finite(result[r]))
            return EulerStatus::NonFinite;
    }
    // Отложенная запись допускает использование одного массива для входа и выхода.
    for (unsigned r = 0; r < count; ++r)
        state[r] = result[r];
    return EulerStatus::Success;
}
} // namespace euler_detail
// Применяет левые собственные векторы фиксированного базиса к состоянию или его приращению.
// Направление и γ должны совпадать с переданными при построении базиса.
AERO_HOST_DEVICE inline EulerStatus conservative_to_characteristic(const EulerRoeBasis& basis,
                                                                   const double* direction, double gamma,
                                                                   const double* u, double* w) {
    const unsigned d = basis.dimension;
    if (d < 1 || d > 3)
        return EulerStatus::InvalidDimension;
    if (!u || !w)
        return EulerStatus::NullPointer;
    if (!(gamma > 1) || !euler_detail::finite(gamma))
        return EulerStatus::InvalidGamma;
    if (!(basis.sound_speed > 0) || !euler_detail::finite(basis.sound_speed))
        return EulerStatus::InvalidRoeState;
    double n[3];
    const EulerStatus status = euler_unit_direction(d, direction, n);
    if (status != EulerStatus::Success)
        return status;
    const double kinetic = 0.5 * euler_detail::dot(d, basis.velocity, basis.velocity);
    return euler_detail::project_characteristic(basis, n, gamma, kinetic, u, w);
}
// Восстанавливает консервативный вектор из амплитуд волн по формуле U = R·W.
// Базис строится отдельно. Преобразование линейное и применимо также к приращениям.
// Эту же реализацию используют CPU и CUDA.
AERO_HOST_DEVICE inline EulerStatus characteristic_to_conservative(const EulerRoeBasis& basis,
                                                                   const double* characteristic,
                                                                   double* conservative) {
    if (basis.dimension < 1 || basis.dimension > 3)
        return EulerStatus::InvalidDimension;
    if (!characteristic || !conservative)
        return EulerStatus::NullPointer;
    return euler_detail::expand_characteristic(basis.dimension + 2, basis, characteristic, conservative);
}
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY)
// Преобразует характеристические приращения обратно в консервативные компоненты.
std::vector<double> characteristic_to_conservative(const EulerRoeBasis& basis,
                                                   const std::vector<double>& characteristic);
#endif
} // namespace aero
#endif
