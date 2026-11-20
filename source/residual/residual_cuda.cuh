#ifndef AERO_RESIDUAL_CUDA_CUH
#define AERO_RESIDUAL_CUDA_CUH
#include "flux_cfl.h"
#include "packed_weno.h"
#include "../timestep/cfl_block_min.cuh"

namespace aero {
namespace residual_cuda_kernels {
using namespace residual_detail;
#ifndef AERO_MINGW_CUDA_HOST
// Нормируем в той же арифметике, что и поток на GPU. Сетка во время расчёта неподвижна.
AERO_GPU_KERNEL void face_normals(Geometry g) {
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < g.edges; i += gridDim.x * blockDim.x)
        euler_unit_direction(g.dim, g.surfaces[i].area, g.packed.normals[i].axis);
}

// Каждое состояние читается многими смежными рёбрами. Давление, скорость и энтальпию
// вычисляем один раз в начале стадии, после обновления поля и обмена с соседними процессами.
AERO_GPU_KERNEL void nodal_gas(unsigned dimension, const double* field, double gamma, Index count, EulerState* gas) {
    for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        decode_nodal_gas(dimension, field, gamma, i, gas);
    }
}

template <unsigned scheme_order, bool use_mp5 = false, bool use_mpweno = false> __device__ __forceinline__ void packed_weno_flux(FaceBatch batch, Index* bad) {
    for (Index k = blockIdx.x * blockDim.x + threadIdx.x; k < batch.count; k += gridDim.x * blockDim.x) {
        const Index f = batch.face(k);
        const auto* tile = complete_tile<scheme_order>(batch.geometry, f);
        if (tile && !compute_packed_face<use_mp5, use_mpweno>(batch, f, *tile))
            atomicMin(bad, f);
    }
}

AERO_GPU_KERNEL void mpweno5_flux(FaceBatch batch, Index* bad) { packed_weno_flux<5, false, true>(batch, bad); }
AERO_GPU_KERNEL void mp5_flux(FaceBatch batch, Index* bad) { packed_weno_flux<5, true>(batch, bad); }
AERO_GPU_KERNEL void weno3_flux(FaceBatch batch, Index* bad) { packed_weno_flux<3>(batch, bad); }
AERO_GPU_KERNEL void weno5_flux(FaceBatch batch, Index* bad) { packed_weno_flux<5>(batch, bad); }

template <bool nodal> __device__ __forceinline__ void compute_fluxes(const FaceBatch& batch, Index* bad) {
    for (Index k = blockIdx.x * blockDim.x + threadIdx.x; k < batch.count; k += gridDim.x * blockDim.x) {
        const Index f = batch.face(k);
        if (!(nodal ? batch.compute_nodal_roe_2d(f) : batch.compute(f)))
            atomicMin(bad, f);
    }
}
AERO_GPU_KERNEL void flux_kernel(FaceBatch batch, Index* bad) { compute_fluxes<false>(batch, bad); }
AERO_GPU_KERNEL void nodal_roe_2d_kernel(FaceBatch batch, Index* bad) { compute_fluxes<true>(batch, bad); }
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
