#include "explicit_stage.h"
#include "explicit_cuda.cuh"

namespace aero { namespace explicit_detail {
namespace {
using namespace aero::explicit_cuda_kernels;
}
void launch_update(Stage s,std::size_t size,std::size_t components,const double* volumes,
    const double* original,const double* current,const double* residual,double dt,double* next,int* invalid,void* stream) {
    const std::size_t blocks=(size+255)/256;
    AERO_LAUNCH(stage_kernel, static_cast<unsigned>(blocks>65535?65535:blocks), 256, static_cast<cudaStream_t>(stream), s,size,components,volumes,original,current,residual,dt,next,invalid);
}
}}
