#ifndef AERO_RESIDUAL_H
#define AERO_RESIDUAL_H

namespace aero {
enum class ResidualVariables { Conservative, Characteristic, Flux };
enum class ResidualSolver { Roe, LocalLF };
struct ResidualOptions {
    ResidualVariables variables;
    ResidualSolver solver;
    double gamma;
    double trace_ratio_min, trace_ratio_max;
    ResidualOptions(ResidualVariables v=ResidualVariables::Conservative, ResidualSolver s=ResidualSolver::Roe,
                    double g=1.4, double lo=0, double hi=1e16)
        : variables(v),solver(s),gamma(g),trace_ratio_min(lo),trace_ratio_max(hi) {}
};
}

#if !defined(AERO_RESIDUAL_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
#include "../bc/periodic.h"
#include <memory>

namespace aero {
ResidualOptions residual_options(const std::string& variables, const std::string& solver, double gamma,
                                 double trace_ratio_min=0, double trace_ratio_max=1e16);

// Momentum components and normals share this orthonormal frame. For an
// embedded 1D/2D mesh supply the tangent axis/axes in world coordinates.
struct ResidualFrame {
    CvVector axis[3];
    ResidualFrame() : axis{{1,0,0},{0,1,0},{0,0,1}} {}
};

class SpatialResidual {
public:
    SpatialResidual(MeshView mesh, CvView cv, ReconstructionStencil stencil,
        const std::vector<BoundaryConfig>& patches, ResidualOptions options,
        int device=-1, const ResidualFrame& frame=ResidualFrame());
    SpatialResidual(unsigned dimension, PeriodicView grid, ReconstructionStencil stencil,
        const std::vector<BoundaryConfig>& patches, ResidualOptions options,
        int device=-1, const ResidualFrame& frame=ResidualFrame());
    ~SpatialResidual();
    SpatialResidual(const SpatialResidual&)=delete;
    SpatialResidual& operator=(const SpatialResidual&)=delete;

    // All pointers are on the selected backend. U, U0 and R: [node][d+2].
    // U0 may be null when the active BC uses no initial value. Output arrays
    // must be disjoint from inputs and one another. On error discard outputs.
    // Edge and boundary speeds use the order of the supplied geometry view.
    void assemble(const double* u, const double* u0, double* residual,
        double* edge_speeds, double* boundary_speeds, void* stream=nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
#endif
#endif
