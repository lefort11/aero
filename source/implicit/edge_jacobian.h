#ifndef AERO_IMPLICIT_EDGE_JACOBIAN_H
#define AERO_IMPLICIT_EDGE_JACOBIAN_H

#include "../eulersolver/euler.h"

namespace aero {
namespace implicit {

// Производные одного потока, направленного от левого узла к правому.
// Площадь уже учтена. Блоки записаны построчно с тем же шагом, что и у BlockJacobi.
struct EdgeFluxJacobian {
    unsigned width;   // Число уравнений Эйлера в узле: размерность плюс два.
    double left[25];  // Столбец j содержит производные по переменной U_left[j].
    double right[25]; // То же для переменных правого узла.

    // Прибавляет вклад ребра к четырём блокам матрицы. В именах блоков
    // первый индекс обозначает узел невязки, второй — узел меняемой переменной.
    // Вызывающая сторона обеспечивает исключительный доступ к этим блокам.
    AERO_HOST_DEVICE void add_to(double* ii, double* ij, double* ji, double* jj) const {
        for (unsigned k = 0; k < width * width; ++k) {
            ii[k] += left[k];
            ij[k] += right[k];
            ji[k] -= left[k];
            jj[k] -= right[k];
        }
    }
};

using LfEdgeJacobian = EdgeFluxJacobian;

namespace edge_detail {
// Производная p=(γ−1)(E−|m|²/(2ρ)) по одной консервативной переменной.
template<class Gas>
AERO_HOST_DEVICE inline double pressure_column(unsigned col, unsigned energy, double gamma, double kinetic,
                                               const Gas& gas) {
    return (gamma - 1.) * (col == 0 ? kinetic : col == energy ? 1. : -gas.velocity[col - 1]);
}
// Производная u_m = (rho*u_m)/rho по одной консервативной переменной.
// Эту же замену переменных используют открытая граница и вязкие члены.
template<class Gas>
AERO_HOST_DEVICE inline double velocity_column(unsigned col, unsigned axis, const Gas& gas) {
    const double momentum = col == axis+1 ? 1. : 0.;
    return (momentum - (col == 0 ? gas.velocity[axis] : 0.)) / gas.density;
}
// Общий столбец производных физического потока Эйлера для LF и Роу.
// Изменяем одну консервативную переменную. Производные давления и нормальной
// скорости нужны и центральному потоку, и диссипации, поэтому считаем их один раз.
struct PhysicalColumn {
    unsigned col, energy;
    double dp, dmn, rho_dun; // d(p), d(m·n), rho*d(u·n) для данного столбца.

    AERO_HOST_DEVICE PhysicalColumn(unsigned column, unsigned dimension, double gamma, const EulerState& state,
                                    const double* normal, double un, double kinetic)
        : col(column), energy(dimension + 1), dp(pressure_column(col, energy, gamma, kinetic, state)),
          dmn(col > 0 && col < energy ? normal[col - 1] : 0.), rho_dun(dmn - (col == 0 ? un : 0.)) {}

    AERO_HOST_DEVICE double flux(unsigned row, const EulerState& state, const double* normal, double un) const {
        if (row == 0) return dmn;
        if (row == energy) return state.enthalpy * rho_dun + un * (dp + (col == energy ? 1. : 0.));
        return state.velocity[row - 1] * rho_dun + normal[row - 1] * dp + (col == row ? un : 0.);
    }
};

AERO_HOST_DEVICE inline bool read_states(const euler_detail::FaceStates& face, double area,
                                         euler_detail::InterfaceGas& gas) {
    return area > 0. && euler_detail::finite(area) && face.upw_coeff >= 0. && face.upw_coeff <= 1. &&
           euler_detail::read_interface_states(face.dimension, face.left, face.right, face.direction,
                                               face.gamma, gas.left, gas.right,
                                               gas.normal) == EulerStatus::Success;
}
} // namespace edge_detail
}
}
#endif
