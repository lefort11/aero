#ifndef AERO_IMPLICIT_ROE_EDGE_JACOBIAN_H
#define AERO_IMPLICIT_ROE_EDGE_JACOBIAN_H
#include "edge_jacobian.h"
#include "../eulersolver/characteristic.h"

namespace aero {
namespace implicit {
namespace roe_jacobian_detail {
// Производная модуля в точке излома выбирается симметрично.
AERO_HOST_DEVICE inline int sign(double x) { return (x > 0.) - (x < 0.); }

// Дифференцируем |A_Роу|*(U_R-U_L) по одной стороне. Шаг конечной разности
// здесь не нужен. При замороженной матрице остаётся только |A_Роу|*d(скачка).
AERO_HOST_DEVICE inline bool differentiate_side(const euler_detail::FaceStates& face, const EulerState& side,
                                                const euler_detail::RoeAverage& mean, const double* n,
                                                const double* jump, const double* wave, double weight,
                                                double jump_sign, double area, bool frozen, double* block) {
    const unsigned d = face.dimension, energy = d + 1, width = d + 2;
    const double c = mean.sound_speed, un = mean.normal_speed, ic = 1. / c;
    const double side_un = euler_detail::dot(d, side.velocity, n);
    const double transport = ::fabs(un);
    const double acoustic[2] = {::fabs(un - c) - transport, ::fabs(un + c) - transport};
    const double kinetic = .5 * euler_detail::dot(d, side.velocity, side.velocity);
    for (unsigned col = 0; col < width; ++col) {
        const edge_detail::PhysicalColumn column(col, d, face.gamma, side, n, side_un, kinetic);
        if (face.upw_coeff == 0.) {
            // Центральный поток состоит только из полусуммы физических потоков.
            for (unsigned row = 0; row < width; ++row) {
                block[row * width + col] = .5 * area * column.flux(row, side, n, side_un);
                if (!euler_detail::finite(block[row * width + col])) return false;
            }
            continue;
        }
        // Единичное возмущение одной консервативной компоненты меняет скачок со знаком стороны.
        double unit[5] = {}, momentum[3], strength[5];
        unit[col] = jump_sign;
        euler_detail::project_normal_waves(d, mean.velocity, n, face.gamma, mean.kinetic, c, ic, unit, momentum,
                                           strength);
        double dv[3] = {}, dh = 0., dc = 0., dun = 0., dt = 0.;
        double dminus = 0., dplus = 0.;
        if (!frozen) {
            // Изменение веса sqrt(rho) даёт добавку, пропорциональную разности
            // между величиной на этой стороне и её средним значением Роу.
            const double dr = col == 0 ? 1. : 0.;
            const double wr = weight / side.density;
            for (unsigned k = 0; k < d; ++k)
                dv[k] = wr * ((col == k + 1 ? 1. : 0.) - .5 * dr * (side.velocity[k] + mean.velocity[k]));
            const double dp = column.dp;
            dh = wr * ((col == energy ? 1. : 0.) + dp - .5 * dr * (side.enthalpy + mean.enthalpy));
            const double dk = euler_detail::dot(d, mean.velocity, dv);
            dc = .5 * (face.gamma - 1.) * (dh - dk) * ic;
            dun = euler_detail::dot(d, dv, n);
            dt = sign(un) * dun;
            // Изменение проекции исходного скачка на акустические характеристики:
            // сумма амплитуд равна dp/c², разность — (dm_n-u_n*dρ)/c.
            const double sum =
                (face.gamma - 1.) * (dk * jump[0] - euler_detail::dot(d, dv, jump + 1)) * ic * ic -
                2. * dc * ic * (wave[0] + wave[energy]);
            const double diff = -ic * (dun * jump[0] + dc * (wave[energy] - wave[0]));
            strength[0] += .5 * (sum - diff);
            strength[energy] += .5 * (sum + diff);
            dminus = sign(un - c) * (dun - dc) - dt;
            dplus = sign(un + c) * (dun + dc) - dt;
        }
        for (unsigned row = 0; row < width; ++row) {
            double minus, plus;
            mean.acoustic_row(row, d, n, minus, plus);
            const double center = row == 0 ? 0. : row == energy ? dh : dv[row - 1];
            const double shift = row == 0 ? 0. : row == energy ? dc * un + c * dun : dc * n[row - 1];
            const double derivative = transport * unit[row] + dt * jump[row] +
                                      minus * (acoustic[0] * strength[0] + dminus * wave[0]) +
                                      plus * (acoustic[1] * strength[energy] + dplus * wave[energy]) +
                                      (center - shift) * acoustic[0] * wave[0] +
                                      (center + shift) * acoustic[1] * wave[energy];
            double& entry = block[row * width + col];
            entry = .5 * area * (column.flux(row, side, n, side_un) - face.upw_coeff * derivative);
            if (!euler_detail::finite(entry)) return false;
        }
    }
    return true;
}
}

// Оба блока относятся к одному ориентированному потоку через ребро.
// Сборщик распределит их между двумя строками матрицы с противоположными знаками.
AERO_HOST_DEVICE inline bool roe_edge_jacobian(const euler_detail::FaceStates& face, double area,
                                               EdgeFluxJacobian& result, bool frozen = false) {
    euler_detail::InterfaceGas gas;
    if (!edge_detail::read_states(face, area, gas)) return false;
    euler_detail::RoeAverage mean;
    if (euler_detail::average_roe_state(face.dimension, gas.left, gas.right, gas.normal, face.gamma, mean) !=
        EulerStatus::Success)
        return false;
    EdgeFluxJacobian blocks = {};
    blocks.width = face.dimension + 2;
    double jump[5] = {}, momentum[3], wave[5];
    for (unsigned k = 0; k < blocks.width; ++k)
        jump[k] = face.right[k] - face.left[k];
    euler_detail::project_normal_waves(face.dimension, mean.velocity, gas.normal, face.gamma, mean.kinetic,
                                       mean.sound_speed, 1. / mean.sound_speed, jump, momentum, wave);
    for (unsigned k = 0; k < 2; ++k) {
        const auto& side = k ? gas.right : gas.left;
        double* block = k ? blocks.right : blocks.left;
        const double weight = side.density_root / (gas.left.density_root + gas.right.density_root);
        if (!roe_jacobian_detail::differentiate_side(face, side, mean, gas.normal, jump, wave, weight,
                                                     k ? 1. : -1., area, frozen, block))
            return false;
    }
    result = blocks;
    return true;
}
}
}
#endif
