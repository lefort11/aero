#ifndef AERO_RESIDUAL_MP6_CARTESIAN_H
#define AERO_RESIDUAL_MP6_CARTESIAN_H
#include "../eulersolver/euler.h"
#include "../reconstruction/mp6_hybrid.h"

namespace aero {
namespace residual_detail {
// Оси для разложения вектора импульса или вектора потока импульса на грани.
// Первая ось направлена по единичной нормали, остальные лежат в плоскости грани.
struct FaceMomentumAxes {
    unsigned dimension;       // Число пространственных координат.
    const double* normal;     // Единичная нормаль, вычисленная решателем для этой грани.
    double tangents[2][3];    // В 2D нужна одна касательная, в 3D — две.

    AERO_HOST_DEVICE FaceMomentumAxes(unsigned dim, const double* area, const double* unit_normal)
        : dimension(dim), normal(unit_normal), tangents{} {
        euler_detail::face_tangents(dim, area, normal, tangents);
    }

    AERO_HOST_DEVICE const double* axis(unsigned k) const { return k == 0 ? normal : tangents[k-1]; }

    // Массу и энергию переносим как скаляры. Между ними записываем проекции
    // вектора импульса на нормаль и касательные в том же порядке, что у ограничителя.
    AERO_HOST_DEVICE void to_local(const double* cartesian, double* local) const {
        local[0] = cartesian[0];
        for (unsigned k = 0; k < dimension; ++k)
            local[1+k] = euler_detail::dot(dimension, axis(k), cartesian+1);
        local[dimension+1] = cartesian[dimension+1];
    }

    // Обратно поворачиваем только поправку. Исходные левые и правые значения
    // хранятся в декартовых осях, поэтому при нулевой поправке сохраняются в точности.
    AERO_HOST_DEVICE void add_local_correction(const double* local, double* cartesian) const {
        cartesian[0] += local[0];
        for (unsigned j = 0; j < dimension; ++j) {
            double change = 0.;
            for (unsigned k = 0; k < dimension; ++k)
                change += axis(k)[j] * local[1+k];
            cartesian[1+j] += change;
        }
        cartesian[dimension+1] += local[dimension+1];
    }
};

// left и right на входе содержат обычные декартовы приращения MP5 или MPWENO5.
// Рассчитываем добавку к ним в осях грани. Для состояния общий вес относится
// к плотности, нормальному импульсу и энергии. Для потока — к потокам этих величин.
template <bool weno>
AERO_HOST_DEVICE inline void correct_cartesian_pair(const FaceMomentumAxes& axes, const double slopes[5][5],
                                                   Mp5Settings settings, double h, double* left, double* right) {
    double local_slopes[5][5], local_left[5], local_right[5];
    for (unsigned row = 0; row < 5; ++row)
        axes.to_local(slopes[row], local_slopes[row]);
    axes.to_local(left, local_left);
    axes.to_local(right, local_right);
    // На входе у ограничителя базовые приращения в осях грани, на выходе — только добавка.
    mp6::reconstruct<weno, true>(local_slopes, axes.dimension+2, settings, h, local_left, local_right);
    axes.add_local_correction(local_left, left);
    axes.add_local_correction(local_right, right);
}
} // namespace residual_detail
} // namespace aero
#endif
