#define AERO_RESIDUAL_KERNEL_ONLY
#define AERO_BC_KERNEL_ONLY
#define AERO_EULERSOLVER_KERNEL_ONLY
#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "../cuda/check.h"
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
unsigned blocks(Index n, unsigned threads = 128) {
    const Index count = (n + threads - 1) / threads;
    return static_cast<unsigned>(count > 65535 ? 65535 : count);
}
using aero::cuda_detail::checked;
}
void residual_detail::gpu_prepare_face_normals(const Geometry& g) {
    if (g.edges) {
        AERO_LAUNCH(face_normals, blocks(g.edges), 128, nullptr, g);
        checked(cudaGetLastError());
    }
}

// Считает один поток на площадку. Пакет может содержать как внутренние рёбра, так и граничные площадки.
void residual_detail::gpu_face_fluxes(const FaceBatch& batch, Index* bad, void* stream) {
    if (!batch.count)
        return;
    FaceBatch remaining = batch;
    const auto& packed = batch.geometry.packed;
    if (packed.gas) {
        AERO_LAUNCH(nodal_gas, blocks(packed.state_nodes), 128, static_cast<cudaStream_t>(stream),
                    batch.geometry.dim, batch.state, batch.geometry.options.gamma, packed.state_nodes, packed.gas);
        checked(cudaGetLastError());
    }
    if (packed.tiles) {
        if (packed.order == 5) {
            AERO_LAUNCH(weno5_flux, blocks(batch.count), 128, static_cast<cudaStream_t>(stream), batch, bad);
        } else {
            AERO_LAUNCH(weno3_flux, blocks(batch.count), 128, static_cast<cudaStream_t>(stream), batch, bad);
        }
        checked(cudaGetLastError());
        remaining.indices = batch.geometry.packed.remaining;
        remaining.count = batch.geometry.packed.count;
        if (!remaining.count)
            return;
    }
    if (remaining.nodal_roe_2d()) {
        AERO_LAUNCH(nodal_roe_2d_kernel, blocks(remaining.count), 128, static_cast<cudaStream_t>(stream), remaining, bad);
    } else {
        AERO_LAUNCH(flux_kernel, blocks(remaining.count), 128, static_cast<cudaStream_t>(stream), remaining, bad);
    }
    checked(cudaGetLastError());
}
void launch_residual(Geometry g, const double* u, const double* u0, EulerFluxResult* flux, double* r,
                     double* edge, double* boundary, Index* bad, cudaStream_t stream) {
    gpu_face_fluxes({g, nullptr, g.faces, u, u0, flux, edge, boundary}, bad, stream);
    // Оба ядра стоят в одном потоке CUDA. К началу суммирования все потоки площадок уже готовы.
    AERO_LAUNCH(gather_kernel, blocks(g.nodes), 128, stream, g, flux, r, bad);
    checked(cudaGetLastError());
}
double residual_detail::gpu_flux_cfl(const FluxCfl& grid, double courant, DeviceCflMinimum* device_minimum,
                                     void* stream_handle) {
    if (!grid.count)
        return std::numeric_limits<double>::infinity();
    const auto stream = static_cast<cudaStream_t>(stream_handle);
    // Все единичные биты означают, что минимум пока пуст и ошибочной строки ещё нет.
    checked(cudaMemsetAsync(device_minimum, 0xff, sizeof(DeviceCflMinimum), stream));
    const unsigned threads = cfl_detail::block_threads;
    AERO_LAUNCH(flux_cfl_kernel, blocks(grid.count, threads), threads, stream, grid, courant, device_minimum);
    checked(cudaGetLastError());
    // На CPU возвращаем только итоговый минимум и индекс ошибки, массивы узлов остаются на GPU.
    DeviceCflMinimum result;
    checked(cudaMemcpyAsync(&result, device_minimum, sizeof(result), cudaMemcpyDeviceToHost, stream));
    checked(cudaStreamSynchronize(stream));
    if (result.bad != ~Index(0))
        throw std::domain_error("Invalid CFL at local node " + std::to_string(result.bad));
    if (result.bits == ~Index(0))
        return std::numeric_limits<double>::infinity();
    double dt;
    static_assert(sizeof(result.bits) == sizeof(dt), "Unexpected CFL result size");
    std::memcpy(&dt, &result.bits, sizeof(dt));
    return dt;
}
}
