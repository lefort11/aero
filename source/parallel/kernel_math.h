#ifndef AERO_KERNEL_MATH_H
#define AERO_KERNEL_MATH_H

// Общую функцию можно вызвать и в обычном цикле CPU, и внутри ядра GPU.
#ifdef __CUDACC__
#define AERO_HOST_DEVICE __host__ __device__
#else
#define AERO_HOST_DEVICE
#endif

// NVRTC предоставляет size_t и математические функции устройства без стандартной библиотеки CPU.
#ifndef __CUDACC_RTC__
#include <math.h>
#include <stddef.h>
#include <float.h>
#endif
// Общие математические константы ядра доступны также при компиляции CUDA без заголовков стандартной библиотеки.
namespace aero {
#ifdef __CUDACC_RTC__
constexpr double max_finite_double = 1.7976931348623157e308; // IEEE 754 binary64
#else
constexpr double max_finite_double = DBL_MAX;
#endif
}
#endif
