#ifndef AERO_COORDINATE_H
#define AERO_COORDINATE_H
#include "../parallel/kernel_math.h"

namespace aero {
// Координаты точки в порядке x, y, z. Три компоненты лежат подряд.
struct Coordinate {
    double values[3];
    AERO_HOST_DEVICE double& operator[](size_t axis) { return values[axis]; }
    AERO_HOST_DEVICE const double& operator[](size_t axis) const { return values[axis]; }
    AERO_HOST_DEVICE double* begin() { return values; }
    AERO_HOST_DEVICE double* end() { return values + 3; }
    AERO_HOST_DEVICE const double* begin() const { return values; }
    AERO_HOST_DEVICE const double* end() const { return values + 3; }
    AERO_HOST_DEVICE void fill(double value) {
        for (double& component : *this) component = value;
    }
};
// Сдвиг точки и разность координат используют один порядок осей на CPU и GPU.
AERO_HOST_DEVICE inline Coordinate operator+(Coordinate a, const Coordinate& b) {
    for (unsigned axis = 0; axis < 3; ++axis) a[axis] += b[axis];
    return a;
}
AERO_HOST_DEVICE inline Coordinate operator-(Coordinate a, const Coordinate& b) {
    for (unsigned axis = 0; axis < 3; ++axis) a[axis] -= b[axis];
    return a;
}
AERO_HOST_DEVICE inline double coordinate_length(const Coordinate& a) {
    return ::hypot(::hypot(a[0], a[1]), a[2]);
}
static_assert(sizeof(Coordinate) == 3 * sizeof(double), "Expected three contiguous doubles");
}
#endif
