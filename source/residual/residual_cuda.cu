#define AERO_RESIDUAL_KERNEL_ONLY
#define AERO_BC_KERNEL_ONLY
#define AERO_EULERSOLVER_KERNEL_ONLY
#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "residual_detail.h"
#include "residual_cuda.cuh"
#include <stdexcept>
#include <string>
#include <cstring>
#include <limits>
namespace aero {
using namespace residual_detail;
namespace {
using namespace aero::residual_cuda_kernels;
unsigned blocks(Index n,unsigned threads=128) {
    const Index count=(n+threads-1)/threads;
    return static_cast<unsigned>(count>65535 ? 65535 : count);
}
void checked(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
void residual_detail::gpu_face_fluxes(const FaceBatch& batch,Index* bad,void* stream) {
    if (!batch.count) return;
    AERO_LAUNCH(flux_kernel,blocks(batch.count),128,static_cast<cudaStream_t>(stream),batch,bad);
    checked(cudaGetLastError());
}
void launch_residual(Geometry g,const double* u,const double* u0,EulerFluxResult* flux,
    double* r,double* edge,double* boundary,Index* bad,cudaStream_t stream) {
    gpu_face_fluxes({g,nullptr,g.faces,u,u0,flux,edge,boundary},bad,stream);
    AERO_LAUNCH(gather_kernel, blocks(g.nodes), 128, stream, g,flux,r,bad);
    checked(cudaGetLastError());
}
double residual_detail::gpu_flux_cfl(const FluxCfl& grid,double courant,DeviceCflMinimum* device_minimum,void* stream_handle) {
    if (!grid.count) return std::numeric_limits<double>::infinity();
    const auto stream=static_cast<cudaStream_t>(stream_handle);
    checked(cudaMemsetAsync(device_minimum,0xff,sizeof(DeviceCflMinimum),stream));
    const unsigned threads=cfl_detail::block_threads;
    AERO_LAUNCH(flux_cfl_kernel, blocks(grid.count,threads),threads,stream,grid,courant,device_minimum);
    checked(cudaGetLastError());
    DeviceCflMinimum result;
    checked(cudaMemcpyAsync(&result,device_minimum,sizeof(result),cudaMemcpyDeviceToHost,stream));
    checked(cudaStreamSynchronize(stream));
    if (result.bad!=~Index(0)) throw std::domain_error("Invalid CFL at local node "+std::to_string(result.bad));
    if (result.bits==~Index(0)) return std::numeric_limits<double>::infinity();
    double dt;
    static_assert(sizeof(result.bits)==sizeof(dt),"Unexpected CFL result size");
    std::memcpy(&dt,&result.bits,sizeof(dt));
    return dt;
}
}
