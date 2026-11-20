#ifndef AERO_IMPLICIT_LF_EDGE_JACOBIAN_H
#define AERO_IMPLICIT_LF_EDGE_JACOBIAN_H

#include "edge_jacobian.h"

namespace aero {
namespace implicit {
namespace lf_detail {
// Один вызов строит один блок. Формула одинакова для обеих сторон, кроме
// знака производной скачка (U_right - U_left) и доли производной максимума.
AERO_HOST_DEVICE inline bool differentiate_side(const euler_detail::FaceStates& face, const EulerState& gas,
                                                const double* normal, double un, double alpha,
                                                double speed_weight, bool left_side, double area,
                                                double* block) {
    const unsigned energy = face.dimension + 1, width = energy + 1;
    const double kinetic = .5 * euler_detail::dot(face.dimension, gas.velocity, gas.velocity);
    for (unsigned col = 0; col < width; ++col) {
        const edge_detail::PhysicalColumn column(col, face.dimension, face.gamma, gas, normal, un, kinetic);
        double dalpha = 0.;
        if (speed_weight != 0. && face.upw_coeff != 0.) {
            // Дифференцируем |u·n|+c только на стороне, задающей максимум.
            // При u·n=0 берём симметричное значение производной модуля.
            const int sign = (un > 0.) - (un < 0.);
            const double dc =
                .5 * gas.sound_speed * (column.dp / gas.pressure - (col == 0 ? 1. / gas.density : 0.));
            dalpha = speed_weight * (sign * (column.rho_dun / gas.density) + dc);
        }
        for (unsigned row = 0; row < width; ++row) {
            double derivative = .5 * column.flux(row, gas, normal, un);
            if (face.upw_coeff != 0.) {
                // У произведения α*(U_right−U_left) два слагаемых: изменение
                // скорости умножает весь скачок, изменение скачка даёт диагональ.
                const double speed_term = dalpha == 0. ? 0. : dalpha * (face.right[row] - face.left[row]);
                const double jump_term = row == col ? (left_side ? -alpha : alpha) : 0.;
                derivative -= (.5 * face.upw_coeff) * (speed_term + jump_term);
            }
            const double value = area * derivative;
            if (!euler_detail::finite(value)) return false;
            block[row * width + col] = value;
        }
    }
    return true;
}
} // namespace lf_detail

// Строит якобиан первого порядка по двум узловым состояниям. Направление
// и модуль площади берутся из геометрии ребра, как при явной сборке потока.
// При ошибке возвращает false и сохраняет предыдущие значения результата.
AERO_HOST_DEVICE inline bool lf_edge_jacobian(const euler_detail::FaceStates& face, double area,
                                              EdgeFluxJacobian& result, bool simplified = false) {
    euler_detail::InterfaceGas gas;
    if (!edge_detail::read_states(face, area, gas)) return false;
    const double ul = euler_detail::dot(face.dimension, gas.left.velocity, gas.normal);
    const double ur = euler_detail::dot(face.dimension, gas.right.velocity, gas.normal);
    const double sl = euler_detail::signal_speed(ul, gas.left);
    const double sr = euler_detail::signal_speed(ur, gas.right);
    const double alpha = sl > sr ? sl : sr;
    if (!euler_detail::finite(alpha)) return false;
    // При равных скоростях у max нет единственной производной. Делим вклад
    // поровну, чтобы перестановка узлов и разворот нормали давали тот же якобиан.
    const double left_weight = sl == sr ? .5 : (sl > sr ? 1. : 0.);
    EdgeFluxJacobian blocks = {};
    blocks.width = face.dimension + 2;
    if (!lf_detail::differentiate_side(face, gas.left, gas.normal, ul, alpha, simplified ? 0. : left_weight,
                                       true, area, blocks.left) ||
        !lf_detail::differentiate_side(face, gas.right, gas.normal, ur, alpha,
                                       simplified ? 0. : 1. - left_weight, false, area, blocks.right))
        return false;
    result = blocks;
    return true;
}

} // namespace implicit
} // namespace aero
#endif
