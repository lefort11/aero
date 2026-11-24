#include "../../tests/device_test_buffer.h"
#include "cv.h"
#include "../../tests/device_cases.h"
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace {
using aero::gpu::checked;

// Проверяем на GPU все три массива: интегрируем x·n_x по внешней поверхности единичного куба (узлы 4–11).
using namespace aero::device_tests;
}
// Сравнивает геометрию контрольных объёмов на CPU и доступных GPU.
int check_cv_devices(aero::CvView host, bool required) {
    int count = 0;
    aero::gpu::Status status = aero::gpu::count(&count);
    if (status == aero::gpu::Status::NoDevice || status == aero::gpu::Status::Unavailable ||
        (status == aero::gpu::Status::Success && !count)) {
        std::puts("No CUDA device available.");
        return required ? 1 : 0;
    }
    checked(status);
    double reference = 0;
    for (size_t i = 0; i < host.edge_count; ++i)
        reference += std::fabs(host.edge_areas[i].x);
    for (int dev = 0; dev < count; ++dev) {
        checked(aero::gpu::select(dev));
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
        aero::device_work::check(inspect(nullptr, view, output.data));

        double result[3];
        checked(aero::gpu::copy(result, output.data, sizeof(result), aero::gpu::Copy::ToHost));
        if (!(std::fabs(result[0] - 1) < 1e-12 && std::fabs(result[1] - 1) < 1e-12 &&
              std::fabs(result[2] - reference) < 1e-12))
            throw std::runtime_error("CUDA median dual mismatch");
        std::printf("CUDA median dual test passed on device %d.\n", dev);
    }
    return 0;
}
