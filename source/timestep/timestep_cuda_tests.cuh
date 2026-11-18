#ifndef AERO_TIMESTEP_CUDA_TESTS_CUH
#define AERO_TIMESTEP_CUDA_TESTS_CUH
#include "timestep.h"
#include "../explicit/explicit.h"
#include "../eulersolver/lax_friedrichs.h"
#include "../cuda/runtime.h"

namespace aero { namespace timestep_cuda_tests_kernels {
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void speed_kernel(const double* u,double* speed) {
    const double n=1;
    EulerFluxResult f={};
    const auto status=lax_friedrichs_conservative(1,u,u,&n,1.4,&f);
    *speed=status==EulerStatus::Success?f.max_wave_speed:-1;
}
AERO_GPU_KERNEL void residual_kernel(double* r) { r[0]=0; r[1]=0; r[2]=-5; }
#endif
}}
#endif
