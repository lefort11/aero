#ifndef AERO_KERNEL_MATH_H
#define AERO_KERNEL_MATH_H

// Общую функцию можно вызвать и в обычном цикле CPU, и внутри ядра GPU.
#include <Kokkos_Macros.hpp>
#define AERO_HOST_DEVICE KOKKOS_FUNCTION
#include <math.h>
#include <stddef.h>
#include <float.h>
#include <string.h>

namespace aero {
// Битовое представление binary64 нужно компактным формулам MP. memcpy
// сохраняет все биты, включая знак нуля, и допускает встраивание компилятором.
KOKKOS_INLINE_FUNCTION unsigned long long double_bits(double x) {
    unsigned long long bits;
    static_assert(sizeof(bits) == sizeof(x), "Expected binary64");
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}
KOKKOS_INLINE_FUNCTION double from_double_bits(unsigned long long bits) {
    double x;
    memcpy(&x, &bits, sizeof(x));
    return x;
}
KOKKOS_INLINE_FUNCTION unsigned high_word(double x) {
    return static_cast<unsigned>(double_bits(x) >> 32);
}
KOKKOS_INLINE_FUNCTION double inverse_value(double x) { return 1. / x; }
constexpr double max_finite_double = DBL_MAX;
}
#endif
