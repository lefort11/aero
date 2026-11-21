#pragma once
#include "../parallel/kernel_math.h"

namespace aero {
namespace turbulence {
// Данные газа и сетки, постоянные между скалярными итерациями SA при фиксированной скорости.
struct DdesPoint {
    double grid_length;   // C_DES * Δ, с учётом F_KH при выборе SLA.
    double gradient_norm; // Норма полного тензора производных скорости, включая чистое растяжение.

    // При малом аргументе tanh(x) вычисляем рациональной формулой до точности
    // double. В середине интервала используется библиотечная гиперболическая функция.
    AERO_HOST_DEVICE static double shielding_fraction(double x) {
        if (x>=20.) return 1.; // 1-tanh(20) меньше половины шага double у единицы.
        if (x>.375) return ::tanh(x);
        const double q=x*x;
        return x*(135135.+q*(17325.+q*(378.+q))) / (135135.+q*(62370.+q*(3150.+28.*q)));
    }
    // Формулы (6)–(8) Spalart et al., DOI 10.1007/s00162-006-0015-0.
    // inverse_wall равен нулю при отсутствии стенок и отрицателен на самой стенке.
    AERO_HOST_DEVICE double inverse_length(double inverse_wall, double total_nu, double kappa) const {
        if (inverse_wall<0.) return inverse_wall;
        if (inverse_wall==0.) return grid_length>0. ? 1./grid_length : -1.;
        const double wall_length=1./inverse_wall;
        if (grid_length>=wall_length || gradient_norm==0.) return inverse_wall;
        const double rd=(total_nu*inverse_wall)*(inverse_wall/gradient_norm)/(kappa*kappa);
        const double arg=8.*rd;
        // При arg >= 4 значение tanh(arg³) округляется до единицы в double.
        if (arg>=4.) return inverse_wall;
        const double protected_fraction=shielding_fraction(arg*arg*arg);
        // Эта форма сохраняет малый сеточный масштаб вдали от стенки.
        const double length=grid_length+protected_fraction*(wall_length-grid_length);
        return length>0. ? 1./length : -1.;
    }
};
}
}
