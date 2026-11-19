#include "../../qa/cuda_test_buffer.h"
#include "timestep.h"
#include "../explicit/explicit.h"
#define AERO_EULERSOLVER_KERNEL_ONLY
#include "../eulersolver/lax_friedrichs.h"
#include "timestep_cuda_tests.cuh"
#include <stdexcept>

using namespace aero;
namespace {
using aero::cuda_detail::checked;
using namespace aero::timestep_cuda_tests_kernels;
}
CflStep gpu_inputs(int device, const CflTimestep& plan, double cfl, WaveSpeeds host) {
    checked(cudaSetDevice(device));
    aero_test::DeviceBuffer<double> edges(host.edge_count), boundary(host.boundary_count);
    edges.upload_async(host.edge, host.edge_count, 0);
    boundary.upload_async(host.boundary, host.boundary_count, 0);
    return plan.compute(cfl, {edges.data, host.edge_count, boundary.data, host.boundary_count});
}
// Проверяет, что вычисление скорости и выбор шага видят данные одного потока CUDA.
CflStep gpu_accepted(const CflTimestep& plan, const ExplicitField& field) {
    aero_test::DeviceBuffer<double> speed(1);
    AERO_LAUNCH(speed_kernel, 1, 1, static_cast<cudaStream_t>(field.stream), field.values, speed.data);
    checked(cudaGetLastError());
    return plan.compute(.5, {speed.data, 1, nullptr, 0}, field.stream);
}
void gpu_residual(const ExplicitField& f, double* r) {
    AERO_LAUNCH(residual_kernel, 1, 1, static_cast<cudaStream_t>(f.stream), r);
    checked(cudaGetLastError());
}
