#ifndef AERO_RESIDUAL_CUDA_CUH
#define AERO_RESIDUAL_CUDA_CUH
#include "flux_cfl.h"
#include "../timestep/cfl_block_min.cuh"

namespace aero {
namespace residual_cuda_kernels {
using namespace residual_detail;
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void flux_kernel(FaceBatch batch, Index* bad) {
    for (Index k = blockIdx.x * blockDim.x + threadIdx.x; k < batch.count; k += gridDim.x * blockDim.x) {
        const Index f = batch.face(k);
        if (!batch.compute(f))
            atomicMin(bad, f);
    }
}
AERO_GPU_KERNEL void flux_cfl_kernel(FluxCfl grid, double courant, DeviceCflMinimum* result) {
    __shared__ Index minima[cfl_detail::block_threads];
    Index best = ~Index(0);
    for (Index k = blockIdx.x * blockDim.x + threadIdx.x; k < grid.count; k += gridDim.x * blockDim.x) {
        const auto limit = grid.local(k, courant);
        if (limit.dt < 0)
            atomicMin(&result->bad, Index(limit.node));
        else if (limit.node != cfl_detail::unconstrained().node) {
            // Для положительных IEEE double сравнение беззнаковых битовых представлений сохраняет порядок чисел.
            const Index bits = static_cast<Index>(__double_as_longlong(limit.dt));
            if (bits < best)
                best = bits;
        }
    }
    best = cfl_detail::block_min(best, minima);
    if (!threadIdx.x)
        atomicMin(&result->bits, best);
}
AERO_GPU_KERNEL void gather_kernel(Geometry g, const EulerFluxResult* flux, double* r, Index* bad) {
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < g.nodes; i += gridDim.x * blockDim.x)
        // Потоку CUDA принадлежит строка одного узла. Сумма и порядок совпадают с CPU.
        if (!assemble_node_residual(g, i, flux, g.node_row(r, i)))
            atomicMin(bad, g.faces + i);
}
#endif
}
}
#endif
