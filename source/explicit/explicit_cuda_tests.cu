#include "explicit_test_problem.h"
#include "explicit_cuda_tests.cuh"
#include <stdexcept>
namespace {
using namespace aero::explicit_cuda_tests_kernels;
}
void device_test_residual(TestProblem problem,double time,aero::ExplicitField field,double* residual) {
    const unsigned blocks=static_cast<unsigned>((field.nodes*field.components+127)/128);
    AERO_LAUNCH(evaluate, blocks, 128, static_cast<cudaStream_t>(field.stream), problem,time,field,residual);
    const cudaError_t code=cudaGetLastError();
    if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
