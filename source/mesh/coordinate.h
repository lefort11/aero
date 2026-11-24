#ifndef AERO_COORDINATE_H
#define AERO_COORDINATE_H
#include "../parallel/kernel_math.h"

namespace aero {
// Координаты точки в порядке x, y, z. Три компоненты лежат подряд.
struct Coordinate {
    double values[3];
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE double& operator[](size_t axis) { return values[axis]; }
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE const double& operator[](size_t axis) const { return values[axis]; }
    // Возвращает начало диапазона для последовательного обхода.
    AERO_HOST_DEVICE double* begin() { return values; }
    // Возвращает границу диапазона после последнего элемента.
    AERO_HOST_DEVICE double* end() { return values + 3; }
    // Возвращает начало диапазона для последовательного обхода.
    AERO_HOST_DEVICE const double* begin() const { return values; }
    // Возвращает границу диапазона после последнего элемента.
    AERO_HOST_DEVICE const double* end() const { return values + 3; }
    // Заполняет все координатные компоненты указанным значением.
    AERO_HOST_DEVICE void fill(double value) {
        for (double& component : *this) component = value;
    }
};
// Сдвиг точки и разность координат используют один порядок осей на CPU и GPU.
AERO_HOST_DEVICE inline Coordinate operator+(Coordinate a, const Coordinate& b) {
    for (unsigned axis = 0; axis < 3; ++axis) a[axis] += b[axis];
    return a;
}
// Вычисляет покомпонентную разность координат.
AERO_HOST_DEVICE inline Coordinate operator-(Coordinate a, const Coordinate& b) {
    for (unsigned axis = 0; axis < 3; ++axis) a[axis] -= b[axis];
    return a;
}
// Умножаем все три координаты на один коэффициент, сохраняя порядок осей.
AERO_HOST_DEVICE inline Coordinate operator*(Coordinate a, double factor) {
    for (double& value : a) value *= factor;
    return a;
}
// Скалярное и векторное произведения нужны геометрии AES и направленным фильтрам LES/DDES.
AERO_HOST_DEVICE inline double dot3(const Coordinate& a, const Coordinate& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
// Вычисляет векторное произведение трёхмерных координат.
AERO_HOST_DEVICE inline Coordinate cross3(const Coordinate& a, const Coordinate& b) {
    return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};
}
// Вычисляет евклидову длину вектора через устойчивые вызовы hypot.
AERO_HOST_DEVICE inline double coordinate_length(const Coordinate& a) {
    return ::hypot(::hypot(a[0], a[1]), a[2]);
}
static_assert(sizeof(Coordinate) == 3 * sizeof(double), "Expected three contiguous doubles");
}
#endif
