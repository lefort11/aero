#ifndef AERO_EULER_CHARACTERISTIC_H
#define AERO_EULER_CHARACTERISTIC_H
#include "../parallel/kernel_math.h"
#include "euler.h"
namespace aero {
namespace euler_detail {
// Проецирует вектор на волны, используя заранее проверенные базис Роу и единичную нормаль.
AERO_HOST_DEVICE inline EulerStatus project_characteristic(const EulerRoeBasis& basis, const double* n,
                                                           double gamma, double kinetic, const double* u,
                                                           double* w) {
    const unsigned d = basis.dimension;
    double momentum[3] = {}, result[5] = {};
    for (unsigned k = 0; k < d; ++k)
        momentum[k] = u[k + 1] - basis.velocity[k] * u[0];
    // Линеаризуем давление в базисе Роу. Это приращение может быть отрицательным.
    const double pressure =
        (gamma - 1) * (u[d + 1] - euler_detail::dot(d, basis.velocity, u + 1) + kinetic * u[0]);
    const double sum = (pressure / basis.sound_speed) / basis.sound_speed;
    const double difference = euler_detail::dot(d, momentum, n) / basis.sound_speed;
    // Нормальная составляющая даёт пару акустических волн. Оставшаяся часть плотности переходит в энтропийную волну.
    result[0] = 0.5 * (sum - difference);
    result[1] = u[0] - sum;
    result[d + 1] = 0.5 * (sum + difference);
    for (unsigned t = 0; t + 1 < d; ++t)
        for (unsigned k = 0; k < d; ++k)
            result[2 + t] += basis.right[5 * (k + 1) + 2 + t] * momentum[k];
    for (unsigned c = 0; c < d + 2; ++c)
        if (!euler_detail::finite(result[c]))
            return EulerStatus::NonFinite;
    for (unsigned c = 0; c < d + 2; ++c)
        w[c] = result[c];
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
    const unsigned count = basis.dimension + 2;
    double result[5] = {0, 0, 0, 0, 0};
    for (unsigned c = 0; c < count; ++c) {
        if (!euler_detail::finite(characteristic[c]))
            return EulerStatus::NonFinite;
        for (unsigned r = 0; r < count; ++r) {
            if (!euler_detail::finite(basis.right[5 * r + c]))
                return EulerStatus::NonFinite;
            // Суммируем вклады волн по столбцам матрицы R, чтобы вернуться к консервативным компонентам.
            result[r] += basis.right[5 * r + c] * characteristic[c];
        }
    }
    for (unsigned r = 0; r < count; ++r)
        if (!euler_detail::finite(result[r]))
            return EulerStatus::NonFinite;
    for (unsigned r = 0; r < count; ++r)
        conservative[r] = result[r];
    return EulerStatus::Success;
}
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
std::vector<double> characteristic_to_conservative(const EulerRoeBasis& basis,
                                                   const std::vector<double>& characteristic);
#endif
} // namespace aero
#endif
