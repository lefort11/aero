#include "reconstruction.h"
#include "stencil_builder.h"
#include <cmath>
#include <limits>

namespace aero {
namespace {
std::size_t checked_size(ReconstructionIndex count, ReconstructionIndex components) {
    const ReconstructionIndex limit =
        static_cast<ReconstructionIndex>((std::numeric_limits<std::size_t>::max)() / sizeof(double));
    if (!components || (count && components > limit / count)) {
        throw ReconstructionError("Invalid component count or array size overflow");
    }
    return static_cast<std::size_t>(count * components);
}
// У первого порядка приращение равно нулю: каждой стороне достаточно пустой строки и номера её узла.
void build_first_order_rows(MeshView mesh, ReconstructionStencilData& data) {
    const auto rows = checked_size(2 * data.edge_count, 1);
    data.complete.assign(rows, 1);
    data.outer_distance.assign(rows, 0.0);
    data.side_scheme.assign(rows, ReconstructionSideScheme::FirstOrder);
    data.far_distance.assign(rows, 0.0);
    data.offsets.assign(rows + 1, 0);
    data.row_owner.reserve(rows);
    for (MeshIndex edge = 0; edge < mesh.edge_count; ++edge) {
        data.row_owner.push_back(mesh.edge_begin[edge]);
        data.row_owner.push_back(mesh.edge_end[edge]);
    }
}
} // namespace

ReconstructionStencilView ReconstructionStencil::view() const {
    const ReconstructionStencilView result = {data_.node_count,
                                              data_.edge_count,
                                              data_.offsets.data(),
                                              data_.nodes.data(),
                                              data_.row_owner.data(),
                                              data_.weights.data(),
                                              data_.slope_offsets.data(),
                                              data_.slope_nodes.data(),
                                              data_.slope_weights.data(),
                                              data_.weno_order.empty() ? nullptr : data_.weno_order.data(),
                                              data_.weno_epsilon_scale.data(),
                                              {data_.scheme == ReconstructionScheme::Mp5, data_.scheme == ReconstructionScheme::MPWENO5,
                                               data_.mpweno_ad_corr, data_.mp5_alpha}};
    return result;
}
ReconstructionIndex ReconstructionStencil::incomplete_sides() const {
    return static_cast<ReconstructionIndex>(std::count(data_.complete.begin(), data_.complete.end(), 0));
}

// Одна точка входа для обычной сетки, периодического покрытия и локальной сетки MPI.
ReconstructionStencil prepare_reconstruction(const Mesh& mesh, const ReconstructionOptions& options) {
    const auto scheme = reconstruction_scheme(options.scheme);
    // Mesh проверяет топологию при создании и предоставляет связность только для чтения.
    if (uses_slope_rows(scheme.family) &&
        (!(options.weno_epsilon > 0) || !std::isfinite(options.weno_epsilon))) {
        throw ReconstructionError("WENO epsilon must be positive and finite");
    }
    ReconstructionStencil result;
    auto& data = result.data_;
    const auto view = mesh.view();
    data.scheme = options.scheme;
    data.mp5_alpha = options.mp5_alpha;
    data.mpweno_ad_corr = options.mpweno_ad_corr;
    if (scheme.family == ReconstructionFamily::Mp)
        validate_mp5_alpha(options.mp5_alpha);
    data.node_count = view.node_count;
    data.edge_count = view.edge_count;
    // Две стороны ребра. Для WENO у каждой стороны ещё четыре строки производных.
    const unsigned rows_per_edge = uses_slope_rows(scheme.family) ? 8 : 2;
    if (data.edge_count > ((std::numeric_limits<ReconstructionIndex>::max)() - 1) / rows_per_edge) {
        throw ReconstructionError("Too many mesh edges");
    }
    switch (scheme.family) {
    case ReconstructionFamily::Constant:
        build_first_order_rows(view, data);
        break;
    case ReconstructionFamily::Ebr:
    case ReconstructionFamily::Mp:
    case ReconstructionFamily::Weno:
        detail::build_layer_stencil(mesh, options, scheme, data);
        break;
    }
    return result;
}

ReconstructionStencil prepare_ebr(const Mesh& mesh, const ReconstructionOptions& options) {
    if (reconstruction_scheme(options.scheme).family != ReconstructionFamily::Ebr) {
        throw ReconstructionError("Invalid reconstruction scheme");
    }
    return prepare_reconstruction(mesh, options);
}

ReconstructionStencil prepare_weno(const Mesh& mesh, const ReconstructionOptions& options) {
    if (reconstruction_scheme(options.scheme).family != ReconstructionFamily::Weno) {
        throw ReconstructionError("prepare_weno requires Weno3 or Weno5");
    }
    return prepare_reconstruction(mesh, options);
}

struct ReconstructionContext::Impl {
    ReconstructionStencil stencil;
    ReconstructionBackend backend;
    void* gpu;
    int device;
    explicit Impl(ReconstructionStencil s)
        : stencil(std::move(s)), backend(ReconstructionBackend::Cpu), gpu(nullptr), device(-1) {}
    ~Impl() {
#ifdef AERO_WITH_KOKKOS_GPU
        aero_reconstruction_device_destroy(gpu);
#endif
    }
};
bool ReconstructionContext::cuda_compiled() {
#ifdef AERO_WITH_KOKKOS_GPU
    return true;
#else
    return false;
#endif
}
int ReconstructionContext::cuda_device_count() {
#ifdef AERO_WITH_KOKKOS_GPU
    int count = 0;
    char error[512] = {};
    if (aero_reconstruction_device_count(&count, error, sizeof(error))) {
        throw ReconstructionError(error);
    }
    return count;
#else
    return 0;
#endif
}
ReconstructionContext::ReconstructionContext(ReconstructionStencil stencil, ReconstructionBackend backend,
                                             int device)
    : impl_(new Impl(std::move(stencil))) {
    if (device < 0) {
        throw ReconstructionError("Negative CUDA device index");
    }
    if (backend != ReconstructionBackend::Cpu && backend != ReconstructionBackend::Cuda &&
        backend != ReconstructionBackend::Auto) {
        throw ReconstructionError("Invalid EBR backend");
    }
    if (backend == ReconstructionBackend::Cpu) {
        return;
    }
    const int count = cuda_device_count();
    if (!count && backend == ReconstructionBackend::Auto) {
        return;
    }
    if (device >= count) {
        throw ReconstructionError("Requested CUDA device is unavailable (or CUDA module was not built)");
    }
#ifdef AERO_WITH_KOKKOS_GPU
    char error[512] = {};
    if (aero_reconstruction_device_create(impl_->stencil.view(), device, &impl_->gpu, error, sizeof(error))) {
        throw ReconstructionError(error);
    }
    impl_->backend = ReconstructionBackend::Cuda;
    impl_->device = device;
#endif
}
ReconstructionContext::~ReconstructionContext() {}
ReconstructionBackend ReconstructionContext::backend() const {
    return impl_->backend;
}
int ReconstructionContext::device() const {
    return impl_->device;
}
ReconstructionStencilView ReconstructionContext::host_view() const {
    return impl_->stencil.view();
}
ReconstructionStencilView ReconstructionContext::device_view() const {
    if (impl_->backend != ReconstructionBackend::Cuda) {
        throw ReconstructionError("Device view requires CUDA backend");
    }
    ReconstructionStencilView view = {};
#ifdef AERO_WITH_KOKKOS_GPU
    aero_reconstruction_device_view(impl_->gpu, &view);
#endif
    return view;
}
} // namespace aero
