#ifndef AERO_RECONSTRUCTION_H
#define AERO_RECONSTRUCTION_H
#include "ebr.h"
#include "weno.h"
#if defined(__CUDACC__)
#define AERO_RECONSTRUCTION_HD __host__ __device__
#else
#define AERO_RECONSTRUCTION_HD
#endif

namespace aero {
typedef unsigned long long ReconstructionIndex;

struct ReconstructionStencilView {
    ReconstructionIndex node_count, edge_count;
    const ReconstructionIndex* offsets; // 2*edge_count+1, rows 2*edge=L, 2*edge+1=R
    const ReconstructionIndex* nodes;
    const ReconstructionIndex* row_owner; // endpoint node for each side row
    const double* weights; // zero-sum rows for doubled increments; empty = zero
    // Optional nonlinear data. Four derivative rows per reconstruction side:
    // central, near on this side, near on opposite side, far on this side.
    const ReconstructionIndex* slope_offsets;
    const ReconstructionIndex* slope_nodes;
    const double* slope_weights;
    const unsigned char* weno_order; // 0 = linear, 3 or 5 = WENO-JS
    const double* weno_epsilon_scale; // h*sqrt(epsilon), in field units
};

struct ReconstructionFieldView {
    const double* values;
    ReconstructionIndex node_stride, component_stride;
    AERO_RECONSTRUCTION_HD double operator()(ReconstructionIndex /*row*/, ReconstructionIndex node,
                                  ReconstructionIndex component) const {
        return values[node*node_stride + component*component_stride];
    }
};

// An accessor may depend on row (edge and side), for a future externally
// supplied characteristic basis. This module does not compute that basis.
// Returns doubled increments: left=2*(Q_L-Q_i), right=2*(Q_j-Q_R).
// Unchecked low-level operation; arrays must be accessible to the executor.
template<class Reader>
AERO_RECONSTRUCTION_HD inline double evaluate_reconstruction_sparse(const ReconstructionIndex* offsets,const ReconstructionIndex* nodes,
                                             const double* weights,ReconstructionIndex sparse_row,ReconstructionIndex read_row,
                                             ReconstructionIndex component,const Reader& read) {
    const ReconstructionIndex begin=offsets[sparse_row],end=offsets[sparse_row+1];
    if (begin==end) return 0.0; // FirstOrder: no field reads needed.
    const double reference=read(read_row,nodes[begin],component);
    double result = 0.0;
    // Difference form annihilates constants exactly and avoids cancellation
    // between large reconstructed states. The first coefficient is implicit
    // through the zero row sum, and its difference is identically zero.
    for (ReconstructionIndex k=begin+1; k<end; ++k)
        result += weights[k]*(read(read_row,nodes[k],component)-reference);
    return result;
}


enum class ReconstructionSide { Left, Right };
enum class ReconstructionStatus { Success, InvalidEdge, InvalidSide, InvalidLayout, NullPointer };

// A side borrows one endpoint vector. The accessor supplies ONLY the other
// nodes: read(row,node,component), where row=2*edge+side. No global field is
// required; the accessor can refer to a compact local stencil or external data.
template<class Reader>
struct ReconstructionSideInput {
    ReconstructionStencilView stencil;
    ReconstructionIndex row;
    const double* own;
    ReconstructionIndex own_stride;
    const Reader& neighbours;
    AERO_RECONSTRUCTION_HD double operator()(ReconstructionIndex r,ReconstructionIndex node,ReconstructionIndex component) const {
        return node==stencil.row_owner[row]?own[component*own_stride]:neighbours(r,node,component);
    }
    AERO_RECONSTRUCTION_HD double linear(ReconstructionIndex component) const {
        return evaluate_reconstruction_sparse(stencil.offsets,stencil.nodes,stencil.weights,row,row,component,*this);
    }
    AERO_RECONSTRUCTION_HD unsigned order() const { return stencil.weno_order?stencil.weno_order[row]:0; }
    AERO_RECONSTRUCTION_HD double slope(unsigned k,ReconstructionIndex component) const {
        return evaluate_reconstruction_sparse(stencil.slope_offsets,stencil.slope_nodes,stencil.slope_weights,4*row+k,row,component,*this);
    }
    AERO_RECONSTRUCTION_HD double epsilon_scale() const { return stencil.weno_epsilon_scale[row]; }
};

// One side of ONE edge per call. Same function on CPU and inside a CUDA kernel.
// Buffers and accessor data must be accessible in the execution memory space.
// No allocation, no grid traversal, no kernel launch. Output must not alias input.
// Left = 2*(Q_L-Q_i); Right = 2*(Q_j-Q_R), both in the i -> j sign convention.
template<class Reader>
AERO_RECONSTRUCTION_HD inline ReconstructionStatus reconstruct_increment2(
    ReconstructionStencilView stencil,ReconstructionIndex edge,ReconstructionSide side,
    const double* own,ReconstructionIndex components,const Reader& neighbours,double* output,
    ReconstructionIndex own_stride=1,ReconstructionIndex output_stride=1) {
    if (edge>=stencil.edge_count) return ReconstructionStatus::InvalidEdge;
    if (side!=ReconstructionSide::Left && side!=ReconstructionSide::Right) return ReconstructionStatus::InvalidSide;
    const ReconstructionIndex limit=~static_cast<ReconstructionIndex>(0);
    if (!components || !own_stride || !output_stride ||
        (components-1)>limit/own_stride || (components-1)>limit/output_stride)
        return ReconstructionStatus::InvalidLayout;
    if (!own || !output || !stencil.row_owner) return ReconstructionStatus::NullPointer;
    const ReconstructionIndex row=2*edge+(side==ReconstructionSide::Right?1:0);
    const ReconstructionSideInput<Reader> input={stencil,row,own,own_stride,neighbours};
    if (stencil.weno_order) reconstruct_weno_increment2(input,components,output,output_stride);
    else reconstruct_ebr_increment2(input,components,output,output_stride);
    return ReconstructionStatus::Success;
}
} // namespace aero

// CUDA geometry ownership only. Kernels and grid traversal belong to the caller.
extern "C" {
int aero_reconstruction_cuda_count(int* count,char* error,unsigned capacity);
int aero_reconstruction_cuda_create(aero::ReconstructionStencilView view,int device,void** state,char* error,unsigned capacity);
void aero_reconstruction_cuda_destroy(void* state);
void aero_reconstruction_cuda_view(void* state,aero::ReconstructionStencilView* view);
}

#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
#include "../mesh/mesh.h"
#include <memory>
#include <map>

namespace aero {
class ReconstructionError : public std::runtime_error {
public:
    explicit ReconstructionError(const std::string& message) : std::runtime_error(message) {}
};

enum class ReconstructionScheme { FirstOrder, Ebr3, Ebr5, Weno3, Weno5 };
enum class ReconstructionSideScheme { FirstOrder, EdgeAverage, Ebr3, Ebr5, Weno3, Weno5 };

struct ReconstructionOptions {
    ReconstructionScheme scheme;
    // Geometry options apply to EBR/WENO. FirstOrder needs no outer points.
    double relative_tolerance;
    // WENO only: epsilon for squared divided differences, in mesh units.
    double weno_epsilon;
    // Strict mode rejects any downgrade from the requested scheme.
    bool require_complete_stencil;
    ReconstructionOptions() : scheme(ReconstructionScheme::Ebr3), relative_tolerance(1.e-10), weno_epsilon(1.e-40), require_complete_stencil(false) {}
};

struct ReconstructionStencilData {
    ReconstructionScheme scheme;
    ReconstructionIndex node_count, edge_count;
    std::vector<ReconstructionIndex> offsets, nodes, row_owner;
    std::vector<double> weights;
    // For WENO, weights above describe the OPTIMAL LINEAR operator, not its
    // nonlinear Jacobian. These additional CSR rows store only geometry.
    std::vector<ReconstructionIndex> slope_offsets, slope_nodes;
    std::vector<double> slope_weights, weno_epsilon_scale;
    std::vector<unsigned char> weno_order;
    // Per row: 1 = complete requested scheme, 0 = downgraded side.
    // FirstOrder is complete on every side, including boundary nodes.
    std::vector<unsigned char> complete;
    std::vector<ReconstructionSideScheme> side_scheme;
    // Distance to the outer point in mesh coordinate units; zero if absent.
    std::vector<double> outer_distance;
    // EBR5: distance from endpoint to the interpolated second-layer position.
    // Diagnostic only: the far derivative is an interpolation of derivatives,
    // NOT a divided difference over this averaged distance. Zero if not built.
    // WENO5 instead stores the actual second ray intersection; its far slope
    // uses (far_distance - outer_distance) as denominator.
    std::vector<double> far_distance;
    ReconstructionStencilData() : scheme(ReconstructionScheme::Ebr3), node_count(0), edge_count(0), offsets(1,0), slope_offsets(1,0) {}
};

class ReconstructionStencil {
public:
    const ReconstructionStencilData& data() const { return data_; }
    ReconstructionStencilView view() const;
    ReconstructionIndex incomplete_sides() const;
private:
    ReconstructionStencilData data_;
    friend class PeriodicDomain;
    friend ReconstructionStencil prepare_reconstruction(const Mesh&, const ReconstructionOptions&);
    friend ReconstructionStencil prepare_ebr(const Mesh&, const ReconstructionOptions&);
    friend ReconstructionStencil prepare_weno(const Mesh&, const ReconstructionOptions&);
};

// Internal shared geometry services for the EBR and WENO preparation units.
// Coordinates/topology are borrowed only for the lifetime of this helper.
namespace detail {
typedef std::map<ReconstructionIndex,double> SparseRow;
struct RayPoint {
    double distance;
    bool outward;
    SparseRow weights;
    RayPoint();
};
void add_scaled_row(SparseRow& destination,const SparseRow& source,double factor);
SparseRow ray_derivative(MeshIndex node,const RayPoint& hit,double h,unsigned side);
class ReconstructionGeometry {
public:
    ReconstructionGeometry(const Mesh& mesh,double tolerance,bool build_stars);
    ~ReconstructionGeometry();
    ReconstructionGeometry(const ReconstructionGeometry&) = delete;
    ReconstructionGeometry& operator=(const ReconstructionGeometry&) = delete;
    double edge_length(MeshIndex edge) const;
    RayPoint star_hit(MeshIndex edge,unsigned side) const;
    RayPoint layer_hit(MeshIndex edge,unsigned side,const std::vector<MeshIndex>& faces) const;
    RayPoint parallel_star_hit(MeshIndex node,MeshIndex edge,unsigned side) const;
    void append_row(ReconstructionStencilData& data,const SparseRow& row,MeshIndex edge,unsigned side) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace detail


ReconstructionStencil prepare_reconstruction(const Mesh& mesh,
    const ReconstructionOptions& options=ReconstructionOptions());

// Checked CPU convenience overload for one endpoint vector.
template<class Reader>
std::vector<double> reconstruct_increment2(const ReconstructionStencil& stencil,
    ReconstructionIndex edge,ReconstructionSide side,const std::vector<double>& own,const Reader& neighbours) {
    std::vector<double> result(own.size());
    const ReconstructionStatus status=reconstruct_increment2(stencil.view(),edge,side,own.data(),
        static_cast<ReconstructionIndex>(own.size()),neighbours,result.data());
    if (status!=ReconstructionStatus::Success) throw ReconstructionError("Invalid single-side reconstruction arguments");
    return result;
}

enum class ReconstructionBackend { Auto, Cpu, Cuda };

// Owns prepared CPU geometry and optionally its device copy. It never owns
// solution fields or launches a reconstruction over the grid. Auto falls back
// to CPU when CUDA is not built or no device exists.
class ReconstructionContext {
public:
    explicit ReconstructionContext(ReconstructionStencil stencil,
        ReconstructionBackend backend=ReconstructionBackend::Auto,int device=0);
    ~ReconstructionContext();
    ReconstructionContext(const ReconstructionContext&) = delete;
    ReconstructionContext& operator=(const ReconstructionContext&) = delete;
    ReconstructionBackend backend() const;
    int device() const;
    ReconstructionStencilView host_view() const;
    ReconstructionStencilView device_view() const;
    static bool cuda_compiled();
    static int cuda_device_count();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace aero
#endif
#undef AERO_RECONSTRUCTION_HD
#endif
