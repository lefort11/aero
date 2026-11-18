#ifndef AERO_RESIDUAL_CUDA_CUH
#define AERO_RESIDUAL_CUDA_CUH
#include "residual_detail.h"
#include "../cuda/runtime.h"

namespace aero { namespace residual_cuda_kernels {
using namespace residual_detail;
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void flux_kernel(Geometry g,const double* u,const double* u0,EulerFluxResult* flux,
                            double* edge,double* boundary,Index* bad) {
    for (Index f=blockIdx.x*blockDim.x+threadIdx.x; f<g.faces; f+=gridDim.x*blockDim.x) {
        EulerFluxResult result{};
        if (!surface_flux(g,f,u,u0,result)) atomicMin(bad,f);
        flux[f]=result;
        if (f<g.edges) edge[f]=result.max_wave_speed;
        else boundary[f-g.edges]=result.max_wave_speed;
    }
}
AERO_GPU_KERNEL void gather_kernel(Geometry g,const EulerFluxResult* flux,double* r,Index* bad) {
    for (Index i=blockIdx.x*blockDim.x+threadIdx.x;i<g.nodes;i+=gridDim.x*blockDim.x)
        if (!gather(g,i,flux,r)) atomicMin(bad,g.faces+i);
}
#endif
}}
#endif
