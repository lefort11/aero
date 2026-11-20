#ifndef AERO_RESIDUAL_H
#define AERO_RESIDUAL_H

#include "../eulersolver/linear_euler.h"
#include "../parallel/modules.h"
#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/transport.h"
#endif

namespace aero {
namespace implicit { class SpatialJacobian; }
namespace residual_detail { struct Geometry; }
enum class ResidualVariables { Conservative, Characteristic, Flux };
enum class ResidualSolver { Roe, LocalLF };
struct ResidualOptions {
#ifdef AERO_HAS_VISCOSITY
    const viscosity::Row* diffusion = nullptr; // Интегральный вклад AES в невязку каждого узла.
    const unsigned char* wall = nullptr; // У неподвижной стенки компоненты импульса закреплены.
#endif
    LinearBackground background; // Параметры линейных уравнений, общие для всех граней.
    ResidualVariables variables;
    ResidualSolver solver;
    double upw_coeff = 1; // Вес EBR в реконструкции и множитель диссипации LF/Roe. Оценка CFL полная.
    double gamma;
    bool retry_nonphysical = false; // Активный откат сам собирает номера всех отказавших узлов.
    double mp6_hybrid = 0; // Доля ограниченной центральной поправки на полном MP-шаблоне.
    double trace_ratio_min, trace_ratio_max;
    ResidualOptions(ResidualVariables v = ResidualVariables::Conservative,
                    ResidualSolver s = ResidualSolver::Roe, double g = 1.4, double lo = 0, double hi = 1e16)
        : variables(v), solver(s), gamma(g), trace_ratio_min(lo), trace_ratio_max(hi) {}
};
}

#if !defined(AERO_RESIDUAL_KERNEL_ONLY)
#include "../bc/periodic.h"
#include <memory>

namespace aero {
ResidualOptions residual_options(const std::string& variables, const std::string& solver, double gamma,
                                 double trace_ratio_min = 0, double trace_ratio_max = 1e16);

// Импульс и нормаль выражены в одном ортонормированном базисе. Для вложенной 1D/2D-сетки
// касательные оси задаются в мировых координатах.
struct ResidualFrame {
    CvVector axis[3];
    ResidualFrame() : axis{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} {}
};

class SpatialResidual {
public:
    SpatialResidual(MeshView mesh, CvView cv, ReconstructionStencil stencil, const BoundaryPatches& patches,
                    ResidualOptions options, int device = -1, const ResidualFrame& frame = ResidualFrame());
    SpatialResidual(unsigned dimension, PeriodicView grid, ReconstructionStencil stencil,
                    const BoundaryPatches& patches, ResidualOptions options, int device = -1,
                    const ResidualFrame& frame = ResidualFrame());
    ~SpatialResidual();
    SpatialResidual(const SpatialResidual&) = delete;
    SpatialResidual& operator=(const SpatialResidual&) = delete;

    // Все указатели относятся к выбранному исполнителю. Размещение U, U0 и R — [узел][d+2].
    // U0 может отсутствовать, если граничным условиям не нужны начальные значения. Выходные массивы должны
    // занимать отдельную память. При ошибке их значения отбрасывают. Скорости следуют порядку рёбер и граничных площадок.
    void assemble(const double* u, const double* u0, double* residual, double* edge_speeds,
                  double* boundary_speeds, void* stream = nullptr);
    // На CPU можно накапливать время цикла по граням в секундах.
    void assemble(const double* u, const double* u0, double* residual, double* edge_speeds,
                  double* boundary_speeds, void* stream, double* flux_seconds);
    // На принятом состоянии сохранённые потоки и связи с узлами используются также для CFL.
    // Объёмы и u находятся на одном исполнителе. На следующих стадиях нужна только правая часть.
    double assemble_cfl(const double* u, const double* u0, double* residual, const double* volumes,
                        double courant, void* stream = nullptr, double* flux_seconds = nullptr,
                        double* cfl_seconds = nullptr);
    void assemble_rhs(const double* u, const double* u0, double* residual, void* stream = nullptr,
                      double* flux_seconds = nullptr);

private:
    friend class implicit::SpatialJacobian;
    // Для якобиана используем уже спроецированные площади и ту же таблицу границ.
    int jacobian_geometry(residual_detail::Geometry& host, residual_detail::Geometry& active) const;
    struct Impl;
    std::shared_ptr<Impl> p_; // Геометрия остаётся живой, пока ей пользуется матрица.
};
}
#endif
#endif
