#include "../../qa/device_test_buffer.h"
#include "bc_test_case.h"
#include "periodic.h"
#include "../../qa/device_cases.h"
#include <cstdio>
#include <stdexcept>
namespace aero {
namespace {
using aero::gpu::checked;

using namespace aero::device_tests;
}
int check_bc_devices(const BcTestCase* cases, size_t count, const FreeBoundaryRules* rules, size_t rule_count,
                     bool required) {
    int devices = 0;
    aero::gpu::Status code = aero::gpu::count(&devices);
    if (code == aero::gpu::Status::NoDevice || code == aero::gpu::Status::Unavailable ||
        (code == aero::gpu::Status::Success && !devices)) {
        std::puts("No CUDA device available.");
        return required ? 1 : 0;
    }
    checked(code);
    // Запускаем один набор эталонных случаев на каждой обнаруженной карте.
    for (int dev = 0; dev < devices; ++dev) {
        checked(aero::gpu::select(dev));
        aero_test::DeviceBuffer<BcTestCase> input(count);
        aero_test::DeviceBuffer<FreeBoundaryRules> rule_table(rule_count);
        checked(aero::gpu::copy(rule_table.data, rules, rule_count * sizeof(*rules), aero::gpu::Copy::ToDevice));
        aero_test::DeviceBuffer<int> failure(1);
        int first = static_cast<int>(count);
        checked(aero::gpu::copy(input.data, cases, count * sizeof(*cases), aero::gpu::Copy::ToDevice));
        checked(aero::gpu::copy(failure.data, &first, sizeof(first), aero::gpu::Copy::ToDevice));
        aero::device_work::check(test_flux(nullptr, input.data, count, rule_table.data, failure.data));

        checked(aero::gpu::copy(&first, failure.data, sizeof(first), aero::gpu::Copy::ToHost));
        if (first != static_cast<int>(count)) {
            std::fprintf(stderr, "CUDA boundary case %d failed on GPU %d\n", first, dev);
            return 1;
        }
        std::printf("%u boundary flux cases passed on GPU %d.\n", static_cast<unsigned>(count), dev);
    }
    return 0;
}
}
void periodic_gpu(int device, aero::PeriodicView view, aero::ReconstructionStencilView stencil,
                  const double* field, const double* expected) {
    using namespace aero;
    checked(aero::gpu::select(device));
    aero_test::DeviceBuffer<PeriodicEdge> edges(view.edge_count);
    aero_test::DeviceBuffer<double> q(view.node_count), out(2 * view.edge_count);
    checked(aero::gpu::copy(q.data, field, view.node_count * sizeof(double), aero::gpu::Copy::ToDevice));
    checked(
        aero::gpu::copy(edges.data, view.edges, view.edge_count * sizeof(PeriodicEdge), aero::gpu::Copy::ToDevice));
    aero::device_work::check(periodic_rows(nullptr, stencil, edges.data, q.data, out.data));

    std::vector<double> result(2 * view.edge_count);
    checked(aero::gpu::copy(result.data(), out.data, result.size() * sizeof(double), aero::gpu::Copy::ToHost));
    for (size_t i = 0; i < result.size(); ++i)
        if (!(std::fabs(result[i] - expected[i]) < 1e-11))
            throw std::runtime_error("periodic GPU row mismatch");
}
