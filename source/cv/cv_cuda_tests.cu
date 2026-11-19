#include "../../qa/cuda_test_buffer.h"
#include "cv.h"
#include "cv_cuda_tests.cuh"
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace {
using aero::cuda_detail::checked;

// Проверяем на GPU все три массива: интегрируем x·n_x по внешней поверхности единичного куба (узлы 4–11).
using namespace aero::cv_cuda_tests_kernels;
}
int check_cv_devices(aero::CvView host, bool required) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && !count)) {
        std::puts("No CUDA device available.");
        return required ? 1 : 0;
    }
    checked(status);
    double reference = 0;
    for (size_t i = 0; i < host.edge_count; ++i)
        reference += std::fabs(host.edge_areas[i].x);
    for (int dev = 0; dev < count; ++dev) {
        checked(cudaSetDevice(dev));
        aero_test::DeviceBuffer<double> volumes(host.node_count), output(3);
        aero_test::DeviceBuffer<aero::CvVector> areas(host.edge_count);
        aero_test::DeviceBuffer<aero::CvBoundary> boundary(host.boundary_count);
        volumes.upload(host.volumes, host.node_count);
        areas.upload(host.edge_areas, host.edge_count);
        boundary.upload(host.boundary, host.boundary_count);
        aero::CvView view = host;
        view.volumes = volumes.data;
        view.edge_areas = areas.data;
        view.boundary = boundary.data;
        AERO_LAUNCH(inspect, 1, 1, nullptr, view, output.data);
        checked(cudaGetLastError());
        double result[3];
        checked(cudaMemcpy(result, output.data, sizeof(result), cudaMemcpyDeviceToHost));
        if (!(std::fabs(result[0] - 1) < 1e-12 && std::fabs(result[1] - 1) < 1e-12 &&
              std::fabs(result[2] - reference) < 1e-12))
            throw std::runtime_error("CUDA median dual mismatch");
        std::printf("CUDA median dual test passed on device %d.\n", dev);
    }
    return 0;
}
