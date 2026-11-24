#define AERO_RESIDUAL_KERNEL_ONLY
#define AERO_BC_KERNEL_ONLY
#define AERO_EULERSOLVER_KERNEL_ONLY
#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "../parallel/device_check.h"
#include "residual_detail.h"
#include "../parallel/device_work.h"
#include <stdexcept>
#include <string>
#include <cstring>
#include <limits>
namespace aero {
using namespace residual_detail;
namespace {
using aero::gpu::checked;
}
// Готовит единичные нормали площадок на GPU.
void residual_detail::gpu_prepare_face_normals(const Geometry& g) {
    if (g.edges) {
        device_work::check(device_work::aero_device_normals(g));
    }
}

// Считает один поток на площадку. Пакет может содержать как внутренние рёбра, так и граничные площадки.
void residual_detail::gpu_face_fluxes(const FaceBatch& batch, Index* bad, void* stream, const FaceBatch* device_faces) {
    if (!batch.count)
        return;
    device_work::check(device_work::aero_device_faces(batch, bad, stream, device_faces));
}
// Запускает вычисление потоков и их сборку в узловую невязку.
void launch_residual(Geometry g, const double* u, const double* u0, EulerFluxResult* flux, double* r,
                     double* edge, double* boundary, Index* bad, aero::gpu::Stream stream) {
    gpu_face_fluxes({g, nullptr, g.faces, u, u0, flux, edge, boundary, false}, bad, stream);
    // Оба ядра стоят в одном потоке CUDA. К началу суммирования все потоки площадок уже готовы.
    device_work::check(device_work::aero_device_gather(g, flux, r, bad, stream));
}
// Находит минимальный шаг по скоростям потоков на GPU.
double residual_detail::gpu_flux_cfl(const FluxCfl& grid, double courant, void* stream_handle) {
    if (!grid.count)
        return std::numeric_limits<double>::infinity();
    CflStep result;
    device_work::check(device_work::aero_device_flux_step(grid, courant, &result, stream_handle));
    if (result.dt < 0)
        throw std::domain_error("Invalid CFL at local node " + std::to_string(result.node));
    return result.node == cfl_detail::unconstrained().node ? std::numeric_limits<double>::infinity() : result.dt;
}
}
