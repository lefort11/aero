#ifndef AERO_TIMESTEP_CUDA_CUH
#define AERO_TIMESTEP_CUDA_CUH
#include "cfl_block_min.cuh"

namespace aero { namespace timestep_cuda_kernels {
using namespace cfl_detail;
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void nodal_min(Grid grid,WaveSpeeds speeds,double cfl,CflStep* partial) {
    __shared__ CflStep buffer[block_threads];
    CflStep best=unconstrained();
    for (size_t i=blockIdx.x*block_threads+threadIdx.x;i<grid.nodes;i+=gridDim.x*block_threads)
        best=minimum(best,local(grid,speeds,cfl,i));
    best=block_min(best,buffer);
    if (!threadIdx.x) partial[blockIdx.x]=best;
}
AERO_GPU_KERNEL void global_min(const CflStep* partial,unsigned blocks,CflStep* answer) {
    __shared__ CflStep buffer[block_threads];
    CflStep best=unconstrained();
    for (unsigned i=threadIdx.x;i<blocks;i+=block_threads) best=minimum(best,partial[i]);
    best=block_min(best,buffer);
    if (!threadIdx.x) *answer=best;
}
#endif
}}
#endif
