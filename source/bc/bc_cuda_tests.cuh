#ifndef AERO_BC_CUDA_TESTS_CUH
#define AERO_BC_CUDA_TESTS_CUH
#include "bc_test_case.h"
#include "periodic.h"
#include "../cuda/runtime.h"

namespace aero { namespace bc_cuda_tests_kernels {
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void periodic_rows(ReconstructionStencilView stencil,const PeriodicEdge* edges,
                              const double* field,double* output) {
    const unsigned row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=2*stencil.edge_count) return;
    const PeriodicEdge& edge=edges[row/2];
    const bool right=(row%2)!=0;
    ReconstructionFieldView q={field,1,1};
    auto status=reconstruct_increment2(stencil,row/2,right?ReconstructionSide::Right:ReconstructionSide::Left,
        field+(right?edge.end:edge.begin),1,q,output+row);
    if (status!=ReconstructionStatus::Success) output[row]=1.e100;
}

AERO_GPU_KERNEL void test_flux(const BcTestCase* cases,size_t count,int* failed) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const BcTestCase& t=cases[i];
    EulerFluxResult result={};
    result.flux[0]=123;
    BoundaryStatus status=boundary_flux(t.dimension,t.config,t.current,t.initial_available?t.initial:nullptr,
                                        t.area,t.gamma,&result);
    bool ok=status==t.expected_status;
    if (status==BoundaryStatus::Success) {
        ok=ok && result.components==t.dimension+2 &&
            fabs(result.max_wave_speed-t.expected.max_wave_speed)<1e-11;
        for (unsigned k=0;k<t.dimension+2;++k)
            ok=ok && fabs(result.flux[k]-t.expected.flux[k])<1e-11;
    } else ok=ok && result.flux[0]==123;
    if (!ok) atomicMin(failed,static_cast<int>(i));
}
#endif
}}
#endif
