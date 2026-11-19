#ifndef AERO_EXPLICIT_CUDA_CUH
#define AERO_EXPLICIT_CUDA_CUH
#include "explicit_stage.h"
#include "../cuda/runtime.h"

namespace aero {
namespace explicit_cuda_kernels {
using namespace explicit_detail;
#ifndef AERO_MINGW_CUDA_HOST
// Каждый поток CUDA обновляет свои компоненты по той же формуле, что и CPU.
AERO_GPU_KERNEL void stage_kernel(FieldStageUpdate update, size_t entries, int* invalid) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < entries; k += stride) {
        // Единственный общий флаг запрещает принять стадию, если хотя бы одна компонента недопустима.
        if (!update.apply(k))
            atomicExch(invalid, 1);
    }
}
#endif
}
}
#endif
