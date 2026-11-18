#ifndef AERO_TIMESTEP_CUDA_CUH
#define AERO_TIMESTEP_CUDA_CUH
#include "cfl_detail.h"
#include "../cuda/runtime.h"

namespace aero { namespace timestep_cuda_kernels {
using namespace cfl_detail;
constexpr unsigned threads=128;
#ifndef AERO_MINGW_CUDA_HOST
__device__ void block_min(CflStep value,CflStep* shared,CflStep* out) {
    const unsigned i=threadIdx.x;
    shared[i]=value;
    __syncthreads();
    for (unsigned n=threads/2;n;n/=2) {
        if (i<n) shared[i]=minimum(shared[i],shared[i+n]);
        __syncthreads();
    }
    if (i==0) *out=shared[0];
}
AERO_GPU_KERNEL void nodal_min(Grid grid,WaveSpeeds speeds,double cfl,CflStep* partial) {
    __shared__ CflStep buffer[threads];
    CflStep best=unconstrained();
    for (size_t i=blockIdx.x*threads+threadIdx.x;i<grid.nodes;i+=gridDim.x*threads)
        best=minimum(best,local(grid,speeds,cfl,i));
    block_min(best,buffer,partial+blockIdx.x);
}
AERO_GPU_KERNEL void global_min(const CflStep* partial,unsigned blocks,CflStep* answer) {
    __shared__ CflStep buffer[threads];
    CflStep best=unconstrained();
    for (unsigned i=threadIdx.x;i<blocks;i+=threads) best=minimum(best,partial[i]);
    block_min(best,buffer,answer);
}
#endif
}}
#endif
