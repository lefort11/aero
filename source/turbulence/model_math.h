#pragma once
#include "../parallel/kernel_math.h"

namespace aero {
namespace turbulence {
// Обратное значение для частых нормировок модели. На GPU одинарная точность
// даёт только начальное приближение. Две поправки Ньютона с FMA восстанавливают
// все значащие разряды double. За диапазоном начальной оценки действует обычное деление.
AERO_HOST_DEVICE inline double reciprocal(double x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if (::fabs(x)>1.e-30 && ::fabs(x)<1.e30) {
        double y=1.f/static_cast<float>(x);
        y=::fma(y,::fma(-x,y,1.),y);
        return ::fma(y,::fma(-x,y,1.),y);
    }
#endif
    return 1./x;
}
}
}
