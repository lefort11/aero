#ifndef AERO_FLUX_CFL_H
#define AERO_FLUX_CFL_H
#include "../parallel/kernel_math.h"
#include "residual_detail.h"
#include "../timestep/cfl_detail.h"
#include "../mesh/mesh_index.h"

namespace aero {
namespace residual_detail {
struct FaceSpeeds {
    const OrientedFace* incidence; // Из записи списка узла получаем номер площадки.
    const Surface* surfaces;       // Модуль площади умножает скорость в сумме CFL.
    const EulerFluxResult* flux;   // Скорости, уже вычисленные при сборке потока.
    AERO_HOST_DEVICE cfl_detail::FaceSpeed operator()(Index j) const {
        const auto face = incidence[j].face();
        return {flux[face].max_wave_speed, surfaces[face].measure};
    }
};
// Без списка узлов обходим всю сетку. При разбиении процесс передаёт список собственных узлов.
struct FluxCfl {
    const Index* offsets; // Диапазоны смежных площадок для каждого узла.
    FaceSpeeds speeds;    // Передаём в редукцию Kokkos только нужные ей массивы.
#ifdef AERO_HAS_VISCOSITY
    const viscosity::Row* diffusion;
#endif
    const double* volumes;
    const MeshIndex* owned;
    Index count;
    FluxCfl(const Geometry& grid, const EulerFluxResult* flux, const double* node_volumes,
            const MeshIndex* owned_nodes, Index rows)
        : offsets(grid.offsets), speeds{grid.incidence, grid.surfaces, flux},
#ifdef AERO_HAS_VISCOSITY
          diffusion(grid.options.diffusion),
#endif
          volumes(node_volumes),
          owned(owned_nodes), count(rows) {}
    AERO_HOST_DEVICE CflStep local(Index k, double courant) const {
        const Index i = owned ? owned[k] : k;
        const double volume = volumes[i];
        if (!owned &&
            (!(volume > 0 && volume <= max_finite_double) || offsets[i] == offsets[i + 1]))
            return {-1, i};
        double viscous_rate = 0.;
#ifdef AERO_HAS_VISCOSITY
        if (diffusion) viscous_rate = diffusion[i].cfl_rate;
#endif
        return cfl_detail::local(offsets, volume, courant, i, speeds, viscous_rate);
    }
};
double cpu_flux_cfl(const FluxCfl&, double courant, const MeshIndex* global_ids = nullptr);
double gpu_flux_cfl(const FluxCfl&, double courant, void* stream);
}
}
#endif
