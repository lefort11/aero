#ifndef AERO_RECONSTRUCTION_CUDA_TESTS_CUH
#define AERO_RECONSTRUCTION_CUDA_TESTS_CUH
#include "reconstruction.h"
#include "../cuda/runtime.h"

namespace aero {
namespace reconstruction_cuda_tests_kernels {
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void reconstruct_kernel(aero::ReconstructionStencilView s, const double* values,
                                        ReconstructionIndex components, double* left, double* right) {
    const ReconstructionIndex rows = 2 * s.edge_count;
    const ReconstructionIndex step = static_cast<ReconstructionIndex>(blockDim.x) * gridDim.x;
    const aero::ReconstructionFieldView input = {values, 1, s.node_count};
    // Тестовый драйвер обходит сетку. Библиотека восстанавливает одну сторону ребра за вызов.
    // Выбор формулы WENO выполняется в reconstruction.h.
    for (ReconstructionIndex row = static_cast<ReconstructionIndex>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows; row += step) {
        const ReconstructionIndex edge = row / 2, owner = s.row_owner[row];
        aero::reconstruct_increment2(
            s, edge, row % 2 ? aero::ReconstructionSide::Right : aero::ReconstructionSide::Left,
            values + owner, components, input, (row % 2 ? right : left) + edge, s.node_count, s.edge_count);
    }
}
AERO_GPU_KERNEL void one_side_kernel(aero::ReconstructionStencilView s, ReconstructionIndex edge,
                                     unsigned side, const double* own, ReconstructionIndex components,
                                     const double* values, double* output) {
    const aero::ReconstructionFieldView input = {values, 1, s.node_count};
    // Ядро CUDA получает ровно одну сторону и один вектор состояния на конце ребра.
    aero::reconstruct_increment2(s, edge, static_cast<aero::ReconstructionSide>(side), own, components, input,
                                 output);
}
#endif
}
}
#endif
