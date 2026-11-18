#ifndef AERO_EXPLICIT_CUDA_CUH
#define AERO_EXPLICIT_CUDA_CUH
#include "explicit_stage.h"
#include "../cuda/runtime.h"

namespace aero { namespace explicit_cuda_kernels {
using namespace explicit_detail;
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void stage_kernel(Stage s,size_t size,size_t components,const double* volumes,
    const double* original,const double* current,const double* residual,double dt,double* next,int* invalid) {
    const size_t stride=static_cast<size_t>(gridDim.x)*blockDim.x;
    for (size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;k<size;k+=stride) {
        next[k]=update(s,original[k],current[k],dt,volumes[k/components],residual[k]);
        if (!finite_value(next[k])) atomicExch(invalid,1);
    }
}
#endif
}}
#endif
