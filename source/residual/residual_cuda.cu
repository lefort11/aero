#define AERO_RESIDUAL_KERNEL_ONLY
#define AERO_BC_KERNEL_ONLY
#define AERO_EULERSOLVER_KERNEL_ONLY
#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "residual_detail.h"
#include "residual_cuda.cuh"
#include <stdexcept>
namespace aero {
using namespace residual_detail;
namespace {
using namespace aero::residual_cuda_kernels;
unsigned blocks(Index n) {
    const Index count=(n+127)/128;
    return static_cast<unsigned>(count>65535 ? 65535 : count);
}
void launched() {
    const auto status=cudaGetLastError();
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
void launch_residual(Geometry g,const double* u,const double* u0,EulerFluxResult* flux,
    double* r,double* edge,double* boundary,Index* bad,cudaStream_t stream) {
    if (g.faces) {
        AERO_LAUNCH(flux_kernel, blocks(g.faces), 128, stream, g,u,u0,flux,edge,boundary,bad);
        launched();
    }
    AERO_LAUNCH(gather_kernel, blocks(g.nodes), 128, stream, g,flux,r,bad);
    launched();
}
}
