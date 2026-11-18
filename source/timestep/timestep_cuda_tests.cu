#include "timestep.h"
#include "../explicit/explicit.h"
#define AERO_EULERSOLVER_KERNEL_ONLY
#include "../eulersolver/lax_friedrichs.h"
#include "timestep_cuda_tests.cuh"
#include <stdexcept>

using namespace aero;
namespace {
void checked(cudaError_t code) {
    if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Values {
    double* p;
    explicit Values(std::size_t n):p(nullptr) { if (n) checked(cudaMalloc(&p,n*sizeof(double))); }
    ~Values() { if (p) cudaFree(p); }
    void copy(const double* src,std::size_t count) {
        if (count) checked(cudaMemcpyAsync(p,src,count*sizeof(double),cudaMemcpyHostToDevice,0));
    }
};
using namespace aero::timestep_cuda_tests_kernels;
}
CflStep gpu_inputs(int device,const CflTimestep& plan,double cfl,WaveSpeeds host) {
    checked(cudaSetDevice(device));
    Values edges(host.edge_count),boundary(host.boundary_count);
    edges.copy(host.edge,host.edge_count);
    boundary.copy(host.boundary,host.boundary_count);
    return plan.compute(cfl,{edges.p,host.edge_count,boundary.p,host.boundary_count});
}
CflStep gpu_accepted(const CflTimestep& plan,const ExplicitField& field) {
    Values speed(1);
    AERO_LAUNCH(speed_kernel, 1, 1, static_cast<cudaStream_t>(field.stream), field.values,speed.p);
    checked(cudaGetLastError());
    return plan.compute(.5,{speed.p,1,nullptr,0},field.stream);
}
void gpu_residual(const ExplicitField& f,double* r) {
    AERO_LAUNCH(residual_kernel, 1, 1, static_cast<cudaStream_t>(f.stream), r);
    checked(cudaGetLastError());
}
