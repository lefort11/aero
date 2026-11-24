#include "../../tests/device_test_buffer.h"
#include "timestep.h"
#include "../explicit/explicit.h"
#define AERO_EULERSOLVER_KERNEL_ONLY
#include "../eulersolver/lax_friedrichs.h"
#include "../../tests/device_cases.h"
#include <stdexcept>

using namespace aero;
namespace {
using aero::gpu::checked;
using namespace aero::device_tests;
}
// Проверяет обработку входных данных оценки CFL на GPU.
CflStep gpu_inputs(int device, const CflTimestep& plan, double cfl, WaveSpeeds host) {
    checked(aero::gpu::select(device));
    aero_test::DeviceBuffer<double> edges(host.edge_count), boundary(host.boundary_count);
    edges.upload_async(host.edge, host.edge_count, 0);
    boundary.upload_async(host.boundary, host.boundary_count, 0);
    return plan.compute(cfl, {edges.data, host.edge_count, boundary.data, host.boundary_count});
}
// Проверяет, что вычисление скорости и выбор шага видят данные одного потока CUDA.
CflStep gpu_accepted(const CflTimestep& plan, const ExplicitField& field) {
    aero_test::DeviceBuffer<double> speed(1);
    aero::device_work::check(speed_kernel(field.stream, field.values, speed.data));

    return plan.compute(.5, {speed.data, 1, nullptr, 0}, field.stream);
}
// Сравнивает ограничения шага, полученные при сборке невязки на GPU.
void gpu_residual(const ExplicitField& f, double* r) {
    aero::device_work::check(residual_kernel(f.stream, r));

}
