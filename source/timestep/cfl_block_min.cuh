#ifndef AERO_CFL_BLOCK_MIN_CUH
#define AERO_CFL_BLOCK_MIN_CUH
#include "cfl_detail.h"
#include "../cuda/runtime.h"

namespace aero {
namespace cfl_detail {
constexpr unsigned block_threads = 128;
static_assert((block_threads & (block_threads - 1)) == 0, "CFL reduction requires a power-of-two block");
#ifndef AERO_MINGW_CUDA_HOST
__device__ inline unsigned long long minimum(unsigned long long a, unsigned long long b) {
    return b < a ? b : a;
}
// Сводит кандидатов одного блока CUDA к одному минимуму через буфер в общей памяти блока.
template <class T> __device__ inline T block_min(T value, T (&shared)[block_threads]) {
    const unsigned i = threadIdx.x;
    shared[i] = value;
    __syncthreads();
    // На каждом уровне вдвое уменьшаем число пар. Между уровнями все нити должны закончить предыдущую запись.
    for (unsigned n = block_threads / 2; n; n /= 2) {
        if (i < n)
            shared[i] = minimum(shared[i], shared[i + n]);
        __syncthreads();
    }
    return shared[0];
}
#endif
}
}
#endif
