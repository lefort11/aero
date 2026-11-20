#ifndef AERO_KERNEL_MATH_H
#define AERO_KERNEL_MATH_H

// Общую функцию можно вызвать и в обычном цикле CPU, и внутри ядра GPU.
#ifdef __CUDACC__
#define AERO_HOST_DEVICE __host__ __device__
#define AERO_KERNEL_INLINE __forceinline__
#else
#define AERO_HOST_DEVICE
#define AERO_KERNEL_INLINE inline
#endif

// NVRTC предоставляет size_t и математические функции устройства без стандартной библиотеки CPU.
#ifndef __CUDACC_RTC__
#include <math.h>
#include <stddef.h>
#include <float.h>
#endif
// Общие математические константы ядра доступны также при компиляции CUDA без заголовков стандартной библиотеки.
namespace aero {
// Двойная точность на обеих платформах. В обычном диапазоне NVIDIA сначала даёт
// аппаратную оценку 1/x. Два уточнения Ньютона с FMA возвращают точность binary64.
// Для нуля и крайних порядков используем обычное деление. Если из границ формулы
// уже известно, что и x, и 1/x нормальные, повторно проверять порядок не требуется.
AERO_HOST_DEVICE inline double inverse_value(double x, bool known_normal = false) {
#ifdef __CUDA_ARCH__
    const unsigned exponent = (static_cast<unsigned>(__double2hiint(x)) >> 20) & 2047u;
    if (known_normal || (exponent > 500 && exponent < 1500)) {
        double inverse;
        asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(inverse) : "d"(x));
        inverse = ::fma(inverse, ::fma(-x, inverse, 1.0), inverse);
        return ::fma(inverse, ::fma(-x, inverse, 1.0), inverse);
    }
#endif
    (void)known_normal; // На CPU обычное деление обрабатывает весь диапазон.
    return 1 / x;
}
#ifdef __CUDACC_RTC__
constexpr double max_finite_double = 1.7976931348623157e308; // IEEE 754 binary64
#else
constexpr double max_finite_double = DBL_MAX;
#endif
}
#endif
