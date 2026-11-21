#ifndef AERO_FLUX_CFL_H
#define AERO_FLUX_CFL_H
#include "../parallel/kernel_math.h"
#include "residual_detail.h"
#include "../timestep/cfl_detail.h"
#include "../timestep/adaptive_step.h"
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
    const double* sa_rate; // Диагональ переноса SA, в единицах обратного времени.
    FluxCfl(const Geometry& grid, const EulerFluxResult* flux, const double* node_volumes,
            const MeshIndex* owned_nodes, Index rows, const double* transport_rate = nullptr)
        : offsets(grid.offsets), speeds{grid.incidence, grid.surfaces, flux},
#ifdef AERO_HAS_VISCOSITY
          diffusion(grid.options.diffusion),
#endif
          volumes(node_volumes),
          owned(owned_nodes), count(rows), sa_rate(transport_rate) {}
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
        const auto gas_step = cfl_detail::local(offsets, volume, courant, i, speeds, viscous_rate);
        if (!sa_rate) return gas_step;
        // Ограничения газа и SA действуют на один шаг. Сравниваем их до
        // редукции, чтобы одного обхода узлов и одного минимума MPI хватило для обоих.
        const double rate = sa_rate[i];
        if (!(rate >= 0. && rate <= max_finite_double)) return {-1., i};
        if (rate == 0.) return gas_step;
        const double transport_dt = courant / rate;
        return cfl_detail::minimum(gas_step, {transport_dt, i});
    }
};
// Один обход сохраняет шаг каждого владельца и собирает границы диапазона для отчёта.
// Пустая подобласть участвует в MPI-редукции с нейтральными границами.
struct StepRange { double minimum, maximum; };
struct LocalSteadySteps {
    using value_type = StepRange;
    FluxCfl gas;
    double courant;
    AdaptiveStepChange change; // Пустое поле означает обычный шаг CFL.
    double* steps;         // Общий шаг газа и SA в нумерации узлов подобласти.

    AERO_HOST_DEVICE void init(StepRange& range) const { range = {max_finite_double, 0.}; }
    AERO_HOST_DEVICE void join(volatile StepRange& range, const volatile StepRange& next) const {
        if (next.minimum < range.minimum) range.minimum = next.minimum;
        if (next.maximum > range.maximum) range.maximum = next.maximum;
    }
    AERO_HOST_DEVICE void operator()(Index row, StepRange& range) const {
        const Index node = gas.owned ? gas.owned[row] : row;
        double dt = change.field ? change.next(row,node,steps[node]) : gas.local(row,courant).dt;
        if (!(dt > 0. && dt < max_finite_double)) dt = -1.;
        steps[node] = dt;
        if (dt < range.minimum) range.minimum = dt;
        if (dt > range.maximum) range.maximum = dt;
    }
};
// После MPI-редукции всем процессам известен один предел. Меняем только свои узлы.
struct LimitLocalSteps {
    const MeshIndex* owned;
    double* steps;
    double ceiling;
    AERO_HOST_DEVICE void operator()(Index row) const {
        const Index node = owned ? owned[row] : row;
        if (steps[node] > ceiling) steps[node] = ceiling;
    }
};
double cpu_flux_cfl(const FluxCfl&, double courant, const MeshIndex* global_ids = nullptr);
double gpu_flux_cfl(const FluxCfl&, double courant, void* stream);
}
}
#endif
