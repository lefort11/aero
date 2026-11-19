#include "../../qa/cuda_test_buffer.h"
#include "bc_test_case.h"
#include "periodic.h"
#include "bc_cuda_tests.cuh"
#include <cstdio>
#include <stdexcept>
namespace aero {
namespace {
using aero::cuda_detail::checked;

using namespace aero::bc_cuda_tests_kernels;
}
int check_bc_devices(const BcTestCase* cases, size_t count, const FreeBoundaryRules* rules, size_t rule_count,
                     bool required) {
    int devices = 0;
    cudaError_t code = cudaGetDeviceCount(&devices);
    if (code == cudaErrorNoDevice || code == cudaErrorInsufficientDriver ||
        (code == cudaSuccess && !devices)) {
        std::puts("No CUDA device available.");
        return required ? 1 : 0;
    }
    checked(code);
    // Запускаем один набор эталонных случаев на каждой обнаруженной карте.
    for (int dev = 0; dev < devices; ++dev) {
        checked(cudaSetDevice(dev));
        aero_test::DeviceBuffer<BcTestCase> input(count);
        aero_test::DeviceBuffer<FreeBoundaryRules> rule_table(rule_count);
        checked(cudaMemcpy(rule_table.data, rules, rule_count * sizeof(*rules), cudaMemcpyHostToDevice));
        aero_test::DeviceBuffer<int> failure(1);
        int first = static_cast<int>(count);
        checked(cudaMemcpy(input.data, cases, count * sizeof(*cases), cudaMemcpyHostToDevice));
        checked(cudaMemcpy(failure.data, &first, sizeof(first), cudaMemcpyHostToDevice));
        AERO_LAUNCH(test_flux, static_cast<unsigned>((count + 63) / 64), 64, nullptr, input.data, count,
                    rule_table.data, failure.data);
        checked(cudaGetLastError());
        checked(cudaMemcpy(&first, failure.data, sizeof(first), cudaMemcpyDeviceToHost));
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
    checked(cudaSetDevice(device));
    aero_test::DeviceBuffer<PeriodicEdge> edges(view.edge_count);
    aero_test::DeviceBuffer<double> q(view.node_count), out(2 * view.edge_count);
    checked(cudaMemcpy(q.data, field, view.node_count * sizeof(double), cudaMemcpyHostToDevice));
    checked(
        cudaMemcpy(edges.data, view.edges, view.edge_count * sizeof(PeriodicEdge), cudaMemcpyHostToDevice));
    AERO_LAUNCH(periodic_rows, static_cast<unsigned>((2 * view.edge_count + 63) / 64), 64, nullptr, stencil,
                edges.data, q.data, out.data);
    checked(cudaGetLastError());
    std::vector<double> result(2 * view.edge_count);
    checked(cudaMemcpy(result.data(), out.data, result.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < result.size(); ++i)
        if (!(std::fabs(result[i] - expected[i]) < 1e-11))
            throw std::runtime_error("periodic GPU row mismatch");
}
