#ifndef AERO_KERNEL_MATH_H
#define AERO_KERNEL_MATH_H

// NVRTC provides size_t and device math intrinsics without a host C library.
#ifndef __CUDACC_RTC__
#include <math.h>
#include <stddef.h>
#include <float.h>
#endif
namespace aero {
#ifdef __CUDACC_RTC__
constexpr double max_finite_double=1.7976931348623157e308; // IEEE 754 binary64
#else
constexpr double max_finite_double=DBL_MAX;
#endif
}
#endif
