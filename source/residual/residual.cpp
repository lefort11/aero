#include "flux_cfl.h"
#include "boundary_flux_failure.h"
#include "../bc/profile.h"
#include "packed_weno.h"
#include "flux_dispatch.h"
#include "../parallel/parallel_for.h"
#include "../parallel/upload_compact.h"
#include "../parallel/wall_timer.h"
#include <cmath>
#include <limits>
#include <numeric>
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#endif

namespace aero {
using namespace residual_detail;
// Находит минимальный шаг по скоростям потоков на CPU.
double residual_detail::cpu_flux_cfl(const FluxCfl& grid, double courant, const MeshIndex* global_ids) {
    const auto limit = cfl_detail::host_min(grid.count, [&](size_t k) { return grid.local(k, courant); });
    if (limit.dt < 0)
        throw std::domain_error("Invalid CFL at node " +
                                std::to_string(global_ids ? global_ids[limit.node] : limit.node));
    return limit.node == cfl_detail::unconstrained().node ? std::numeric_limits<double>::infinity()
                                                          : limit.dt;
}
namespace {
// Проверяет размеры, параметры газа и допустимость конфигурации пространственного оператора.
void validate(ResidualOptions o) {
    if (!(o.gamma > 1 && std::isfinite(o.gamma)))
        throw std::invalid_argument("residual: invalid gamma");
    if (!(o.trace_ratio_min >= 0 && o.trace_ratio_min < 1 && o.trace_ratio_max > 1 &&
          std::isfinite(o.trace_ratio_max)))
        throw std::invalid_argument("residual: invalid trace ratio bounds");
    if (!(o.mp6_hybrid >= 0 && o.mp6_hybrid <= 1))
        throw std::invalid_argument("residual: mp6_hybrid must be in [0, 1]");
    if (o.variables != ResidualVariables::Conservative && o.variables != ResidualVariables::Characteristic &&
        o.variables != ResidualVariables::Flux && o.variables != ResidualVariables::CharacteristicFlux)
        throw std::invalid_argument("residual: invalid reconstruction variables");
    if (o.solver != ResidualSolver::Roe && o.solver != ResidualSolver::LocalLF)
        throw std::invalid_argument("residual: invalid solver");
}
}
// Проверяет касательность фоновой скорости на стенках линейной задачи.
void residual_detail::validate_linear_walls(const Geometry& g) {
    const auto& background = g.options.background;
    if (!background.active()) return;
    double speed_scale = background.sound;
    for (unsigned j = 0; j < g.dim; ++j) speed_scale += std::fabs(background.velocity[j]);
    const double tolerance = 64 * std::numeric_limits<double>::epsilon() * speed_scale;
    for (Index f = g.edges; f < g.faces; ++f) {
        const auto& wall = g.surfaces[f];
        if (!solid_wall(g.patches.by_patch[wall.patch].type())) continue;
        const double un = euler_detail::dot(g.dim, background.velocity, wall.area) / wall.measure;
        if (std::fabs(un) > tolerance)
            throw std::invalid_argument("linear.txt: background velocity must be tangent to a stationary wall");
    }
}
// Переводит вектор площади в базис компонент импульса решателя.
void residual_detail::project_surface(Surface& s, CvVector world, unsigned d, const ResidualFrame& frame) {
    s.measure = world.length();
    if (!(s.measure > 0 && std::isfinite(s.measure)))
        throw std::invalid_argument("residual: invalid area");
    CvVector remainder = world;
    for (unsigned k = 0; k < d; ++k) {
        const auto axis = frame.axis[k];
        const double a = world.dot(axis);
        s.area[k] = a;
        remainder.x -= a * axis.x;
        remainder.y -= a * axis.y;
        remainder.z -= a * axis.z;
    }
    if (!(remainder.length() <= 1.e-10 * s.measure))
        throw std::invalid_argument("residual: area outside the supplied momentum frame");
}
namespace {
// Перечисляет вклады поверхности в узлы в порядке сетки. Одна и та же схема нужна для подсчёта и заполнения.
template <class Visit>
void for_each_node_contribution(Index edges, const std::vector<Surface>& surfaces, const Visit& visit) {
    for (Index face = 0; face < surfaces.size(); ++face) {
        const Surface& surface = surfaces[face];
        // В начале ребра нормаль направлена из объёма узла. У физической площадки она тоже наружная.
        visit(surface.begin, OrientedFace::outward(face));
        // На втором конце та же нормаль направлена внутрь. У периодической петли оба вклада идут одному узлу.
        if (face < edges)
            visit(surface.end, OrientedFace::inward(face));
    }
}
}

// Один раз при подготовке сетки строит непрерывный список вкладов для каждого узла.
void residual_detail::build_incidence(Index nodes, Index edges, const std::vector<Surface>& surfaces,
                                      std::vector<Index>& offsets, std::vector<OrientedFace>& incidence) {
    offsets.assign(nodes + 1, 0);
    for_each_node_contribution(edges, surfaces, [&](Index node, OrientedFace) { ++offsets[node + 1]; });
    // После накопления узлу i принадлежит ровно диапазон [offsets[i], offsets[i+1]).
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    incidence.resize(offsets.back());
    auto next = offsets;
    // next отмечает свободное место в строке узла. Само начало строки сохраняем для всех стадий расчёта.
    for_each_node_contribution(edges, surfaces,
                               [&](Index node, OrientedFace entry) { incidence[next[node]++] = entry; });
}
namespace {
// Формирует ошибку вычисления потока с номером проблемной площадки.
void failure(Index bad, const Geometry& g) {
    if (bad < g.edges)
        throw std::domain_error("residual: invalid flux at edge " + std::to_string(bad));
    if (bad < g.faces)
        throw std::domain_error("residual: invalid boundary flux at piece " + std::to_string(bad - g.edges));
    throw std::domain_error("residual: nonfinite sum at node " + std::to_string(bad - g.faces));
}
}
#ifdef AERO_WITH_KOKKOS_GPU
// Запускает вычисление потоков и их сборку в узловую невязку.
void launch_residual(Geometry, const double*, const double*, EulerFluxResult*, double*, double*, double*,
                     Index*, aero::gpu::Stream);
#endif

namespace {
// Полная плитка хранит геометрию сразу обеих сторон. Выбираем только формулу
// реконструкции, а построение базиса и запись потока общие с ядрами GPU.
template <unsigned order>
struct CpuPackedFlux {
    const FaceBatch& batch;
    Index face;
    const WenoEdgeTile<order>& tile;
    template<bool weno> bool apply_mp() const {
        return compute_packed_face<!weno,weno>(batch,face,tile);
    }
    bool apply_weno() const { return compute_packed_face<false>(batch,face,tile); }
};

// Вычисляет поток площадки с выбранным типом реконструкции на CPU.
template <unsigned order> bool cpu_face_with_stencil(const FaceBatch& batch, Index f) {
    // У линейных схем EBR все четыре компоненты в 2D используют одну строку весов.
    if (batch.linear_roe_2d())
        return batch.compute_face<true,false,4>(f);
    const auto* tile = complete_tile<order>(batch.geometry, f);
    if (tile)
        return dispatch_packed_reconstruction<order>(batch.geometry,CpuPackedFlux<order>{batch,f,*tile});
    // Для границы либо неполной стороны порядок и способ чтения определяются на месте.
    return batch.roe_2d() ? batch.compute_face<true>(f) : batch.compute_face(f);
}

// Номер face уже переведён из локального списка MPI в индекс площадки подобласти.
bool cpu_flux_for_face(const FaceBatch& batch, Index face) {
    if (batch.nodal_roe_2d())
        return batch.compute_face<true,true>(face);
    if (batch.geometry.packed.order == 5)
        return cpu_face_with_stencil<5>(batch, face);
    return cpu_face_with_stencil<3>(batch, face);
}
// В первом порядке оба состояния уже известны в узлах. Граничный поток остаётся общим
// для всех схем, а на внутреннем ребре LF сразу обрабатывает ровно dim+2 компоненты.
template<unsigned dim> bool cpu_nodal_lf(const FaceBatch& batch, Index face) {
    return batch.compute_face<false,true>(face,dim);
}

// Вычисляет потоки линеаризованных уравнений Эйлера на CPU.
bool cpu_linear_flux(const FaceBatch& batch, Index face) { return batch.compute_linear(face); }

// Способ вычисления потока выбран перед обходом. Нумерация MPI и обработка ошибок общие.
template<bool (*compute)(const FaceBatch&, Index)>
void compute_cpu_faces(const FaceBatch& batch, const MeshIndex* global_nodes) {
    parallel_for(
        batch.count,
        [&](Index k) {
            const Index f = batch.face(k);
            if (compute(batch, f))
                return;
            if (batch.is_boundary(f)) throw BoundaryFluxFailure(f);
            if (global_nodes)
                throw std::domain_error("Invalid flux at global node " +
                                        std::to_string(global_nodes[batch.geometry.surfaces[f].begin]));
            failure(f, batch.geometry);
        },
        256);
}
}

// Вычисляет потоки набора внутренних и граничных площадок на CPU.
void residual_detail::cpu_face_fluxes(const FaceBatch& batch, const MeshIndex* global_nodes) {
    const auto& packed = batch.geometry.packed;
    dispatch_face_fluxes(batch, [&] {
        parallel_for(packed.state_nodes, [&](Index i) {
            decode_nodal_gas(batch.geometry.dim, batch.state, batch.geometry.options.gamma, i, packed.gas);
        });
    }, [&] {
        compute_cpu_faces<cpu_linear_flux>(batch, global_nodes);
    }, [&] {
        compute_cpu_faces<split_face>(batch,global_nodes);
    }, [&] {
        if (batch.nodal_lf()) {
            const auto launch = for_dimension(batch.geometry.dim, compute_cpu_faces<cpu_nodal_lf<1>>,
                                                           compute_cpu_faces<cpu_nodal_lf<2>>,
                                                           compute_cpu_faces<cpu_nodal_lf<3>>);
            launch(batch,global_nodes);
        } else {
            compute_cpu_faces<cpu_flux_for_face>(batch, global_nodes);
        }
    });
}

// Переносит настройки постановки в параметры пространственного оператора.
ResidualOptions residual_options(const std::string& variables, const std::string& solver, double gamma,
                                 double trace_ratio_min, double trace_ratio_max) {
    ResidualOptions o{};
    if (variables == "conservative")
        o.variables = ResidualVariables::Conservative;
    else if (variables == "characteristic")
        o.variables = ResidualVariables::Characteristic;
    else if (variables == "characteristic_flux")
        o.variables = ResidualVariables::CharacteristicFlux;
    else if (variables == "flux")
        o.variables = ResidualVariables::Flux;
    else
        throw std::invalid_argument("residual: unknown reconstruction variables: " + variables);
    if (solver == "roe")
        o.solver = ResidualSolver::Roe;
    else if (solver == "lf")
        o.solver = ResidualSolver::LocalLF;
    else
        throw std::invalid_argument("residual: unknown solver: " + solver);
    o.gamma = gamma;
    o.trace_ratio_min = trace_ratio_min;
    o.trace_ratio_max = trace_ratio_max;
    validate(o);
    return o;
}

// Геометрия, исполнитель и рабочие буферы сборки пространственной невязки.
struct SpatialResidual::Impl {
    int device;                          // Отрицательное значение выбирает CPU.
    ReconstructionScheme scheme;         // Якобиан первого порядка требует той же схемы у невязки.
    ReconstructionContext context;       // Владеет шаблоном и его копией на устройстве.
    std::vector<Surface> surfaces;       // Сначала внутренние рёбра, затем физические площадки.
    // Узловые данные профилей физических границ.
    std::vector<ProfileState> profiles;
    BoundaryPatches patches;             // Метки и правила физических границ.
    std::vector<Index> offsets;          // Начало и конец списка для каждого узла.
    std::vector<OrientedFace> incidence; // Номера поверхностей с ориентацией относительно узла.
    std::vector<EulerFluxResult> buffer; // На CPU хранит вычисленные потоки до сборки по узлам.
    Geometry g{};                        // Представление массивов на выбранном устройстве.
    // Собирает пространственную невязку из потоков площадок и дополнительных вкладов.
    void assemble(const double* u, const double* u0, double* r, double* edge, double* boundary, void* stream,
                  double* flux_seconds);
#ifdef AERO_WITH_KOKKOS_GPU
    // Владелец выделенных на GPU массивов невязки.
    gpu::DeviceArrays storage;
    // Потоки площадок на активном GPU.
    EulerFluxResult* device_flux = nullptr;
    Index* device_error = nullptr; // Первый узел или поверхность с ошибкой расчёта.
#endif
    // Сохраняет выбранную схему и создаёт контекст реконструкции нужного исполнителя.
    Impl(ReconstructionStencil stencil, int id)
        : device(id), scheme(stencil.data().scheme),
          context(std::move(stencil), id < 0 ? ReconstructionBackend::Cpu : ReconstructionBackend::Cuda,
                  id < 0 ? 0 : id) {
        if (id < -1)
            throw std::invalid_argument("residual: invalid device");
#ifndef AERO_WITH_KOKKOS_GPU
        if (id >= 0)
            throw std::invalid_argument("residual: CUDA is not enabled in this build");
#endif
    }
    // Строит площадки, узловую связность и граничные данные пространственного оператора.
    template <class EdgeAt>
    void prepare(unsigned d, Index nodes, Index edges, Index pieces, const CvBoundary* boundary,
                 EdgeAt edge_at, const BoundaryPatches& bc, ResidualOptions options,
                 const ResidualFrame& frame) {
        validate(options);
        const auto stencil = context.host_view();
        if (d < 1 || d > 3 || !nodes || nodes != stencil.node_count || edges != stencil.edge_count ||
            (pieces && !boundary) || edges > surfaces.max_size() || pieces > surfaces.max_size() - edges ||
            nodes >= offsets.max_size() || nodes > std::vector<double>().max_size() / (d + 2))
            throw std::invalid_argument("residual: incompatible grid or stencil");
        for (unsigned i = 0; i < d; ++i)
            for (unsigned j = 0; j <= i; ++j)
                if (!(std::abs(frame.axis[i].dot(frame.axis[j]) - (i == j ? 1 : 0)) < 1.e-12))
                    throw std::invalid_argument("residual: invalid momentum frame");
        surfaces.resize(edges + pieces);
        std::vector<unsigned char> inlet_outlet_node(nodes, 0);
        for (Index f = 0; f < surfaces.size(); ++f) {
            Surface& s = surfaces[f];
            if (f < edges) {
                const auto e = edge_at(f);
                s.begin = e.begin;
                s.end = e.end;
                if (s.begin >= nodes || s.end >= nodes || stencil.row_owner[2 * f] != s.begin ||
                    stencil.row_owner[2 * f + 1] != s.end)
                    throw std::invalid_argument("residual: edge/stencil endpoint mismatch");
                project_surface(s, e.area, d, frame);
            } else {
                const auto& b = boundary[f - edges];
                s.begin = b.node;
                s.patch = b.patch;
                if (b.node >= nodes || b.patch >= bc.by_patch.size() ||
                    bc.by_patch[b.patch].type() == BoundaryType::Periodic)
                    throw std::invalid_argument("residual: invalid physical boundary");
                if (needs_nodal_traces(bc.by_patch[b.patch].type()))
                    inlet_outlet_node[b.node] = 1;
                append_boundary_profile(profiles,bc,b,d,frame.axis,options.gamma);
                project_surface(s, b.area, d, frame);
            }
        }
        // При пустых линейных строках приращения с обеих сторон равны нулю.
        // Узловые следы сразу дают тот же поток, включая шаблон первого порядка из публичного API.
        const bool all_nodal = !stencil.weno_order && stencil.offsets && stencil.offsets[2 * edges] == 0;
        for (Index f = 0; f < edges; ++f) {
            auto& s = surfaces[f];
            s.use_nodal_traces = all_nodal || inlet_outlet_node[s.begin] || inlet_outlet_node[s.end];
        }
        build_incidence(nodes, edges, surfaces, offsets, incidence);
        patches = bc;
        g = {d,
             nodes,
             edges,
             surfaces.size(),
             surfaces.data(),
             offsets.data(),
             incidence.data(),
             patches.view(profiles),
             stencil,
             options, {}, false};
        validate_linear_walls(g);
        if (device < 0) {
            buffer.resize(surfaces.size());
            return;
        }
#ifdef AERO_WITH_KOKKOS_GPU
        gpu::DeviceScope scope(device);
        gpu::checked(scope.status);
        storage.device = device;
        g.surfaces = upload_records<Surface32>(storage,surfaces.size(),surfaces.data());
        g.offsets = upload_indices(storage,offsets.size(),offsets.data());
        g.incidence = upload_records<OrientedFace32>(storage,incidence.size(),incidence.data());
        g.patches = upload_boundary_patches(storage, g.patches);
        g.stencil = context.device_view();
        device_flux = storage.allocate<EulerFluxResult>(surfaces.size());
        device_error = storage.allocate<Index>(1);
        // Сразу после возврата сборка может читать геометрию в неблокирующем потоке CUDA.
        gpu::checked(aero::gpu::wait(nullptr));
#endif
    }
};

// Подготавливает геометрию, реконструкцию и границы пространственного оператора.
SpatialResidual::SpatialResidual(MeshView mesh, CvView cv, ReconstructionStencil stencil,
                                 const BoundaryPatches& patches, ResidualOptions options, int device,
                                 const ResidualFrame& frame)
    : p_(new Impl(std::move(stencil), device)) {
    if (cv.dimension != mesh.dimension || cv.edge_count != mesh.edge_count ||
        cv.node_count != mesh.node_count ||
        (cv.edge_count && (!mesh.edge_begin || !mesh.edge_end || !cv.edge_areas)))
        throw std::invalid_argument("residual: incompatible mesh and control volumes");
    p_->prepare(
        cv.dimension, cv.node_count, cv.edge_count, cv.boundary_count, cv.boundary,
        [&](Index e) -> PeriodicEdge { return {mesh.edge_begin[e], mesh.edge_end[e], e, cv.edge_areas[e]}; },
        patches, options, frame);
}
// Подготавливает геометрию, реконструкцию и границы пространственного оператора.
SpatialResidual::SpatialResidual(unsigned d, PeriodicView grid, ReconstructionStencil stencil,
                                 const BoundaryPatches& patches, ResidualOptions options, int device,
                                 const ResidualFrame& frame)
    : p_(new Impl(std::move(stencil), device)) {
    if (grid.edge_count && !grid.edges)
        throw std::invalid_argument("residual: missing periodic edges");
    p_->prepare(
        d, grid.node_count, grid.edge_count, grid.boundary_count, grid.boundary,
        [&](Index e) { return grid.edges[e]; }, patches, options, frame);
}
// Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
SpatialResidual::~SpatialResidual() {}

// Для якобиана используем уже спроецированные площади и ту же таблицу границ.
int SpatialResidual::jacobian_geometry(Geometry& host, Geometry& active) const {
    if (p_->scheme != ReconstructionScheme::FirstOrder || p_->g.options.solver != ResidualSolver::LocalLF)
        throw std::invalid_argument("LF Jacobian requires first-order reconstruction and LF flux");
    active = p_->g;
    host = active;
    host.surfaces = p_->surfaces.data();
    host.offsets = p_->offsets.data();
    host.incidence = p_->incidence.data();
    host.patches = p_->patches.view(p_->profiles);
    host.stencil = p_->context.host_view();
    return p_->device;
}

// Собирает пространственную невязку из потоков площадок и дополнительных вкладов.
void SpatialResidual::assemble(const double* u, const double* u0, double* r, double* edge, double* boundary,
                               void* stream) {
    assemble(u, u0, r, edge, boundary, stream, nullptr);
}
// Собирает пространственную невязку из потоков площадок и дополнительных вкладов.
void SpatialResidual::assemble(const double* u, const double* u0, double* r, double* edge, double* boundary,
                               void* stream, double* flux_seconds) {
    const auto& g = p_->g;
    if ((g.edges && !edge) || (g.faces > g.edges && !boundary))
        throw std::invalid_argument("residual: missing input or output");
    p_->assemble(u, u0, r, edge, boundary, stream, flux_seconds);
}
// Переводит интегральную невязку в правую часть временного уравнения.
void SpatialResidual::assemble_rhs(const double* u, const double* u0, double* r, void* stream,
                                   double* flux_seconds) {
    p_->assemble(u, u0, r, nullptr, nullptr, stream, flux_seconds);
}
// На принятом состоянии сохранённые потоки и связи с узлами используются также для CFL.
// Объёмы и u находятся на одном исполнителе. На следующих стадиях нужна только правая часть.
double SpatialResidual::assemble_cfl(const double* u, const double* u0, double* r, const double* volumes,
                                     double courant, void* stream, double* flux_seconds,
                                     double* cfl_seconds) {
    if (!volumes || !(courant > 0 && std::isfinite(courant)))
        throw std::invalid_argument("invalid CFL or volumes");
    assemble_rhs(u, u0, r, stream, flux_seconds);
    WallTimer timer(cfl_seconds);
    const auto& g = p_->g;
    double dt;
#ifdef AERO_WITH_KOKKOS_GPU
    if (p_->device >= 0) {
        gpu::DeviceScope scope(p_->device);
        gpu::checked(scope.status);
        dt = gpu_flux_cfl({g, p_->device_flux, volumes, nullptr, g.nodes}, courant, stream);
    } else
#endif
        dt = cpu_flux_cfl({g, p_->buffer.data(), volumes, nullptr, g.nodes}, courant);
    if (!std::isfinite(dt))
        throw std::domain_error("no finite global CFL step");
    return dt;
}
// Собирает пространственную невязку из потоков площадок и дополнительных вкладов.
void SpatialResidual::Impl::assemble(const double* u, const double* u0, double* r, double* edge,
                                     double* boundary, void* stream, double* flux_seconds) {
    if (!u || !r)
        throw std::invalid_argument("residual: missing input or output");
#ifdef AERO_WITH_KOKKOS_GPU
    if (device >= 0) {
        gpu::DeviceScope scope(device);
        gpu::checked(scope.status);
        const auto queue = static_cast<aero::gpu::Stream>(stream);
        gpu::checked(aero::gpu::fill_async(device_error, 0xff, sizeof(Index), queue));
        launch_residual(g, u, u0, device_flux, r, edge, boundary, device_error, queue);
        Index bad;
        gpu::checked(aero::gpu::copy_async(&bad, device_error, sizeof(bad), aero::gpu::Copy::ToHost, queue));
        gpu::checked(aero::gpu::wait(queue));
        if (bad != ~Index(0))
            failure(bad, g);
        return;
    }
#endif
    if (stream)
        throw std::invalid_argument("residual: CUDA stream on CPU");
    // Первый проход по поверхностям: каждое ребро или площадка заполняет свою запись в buffer.
    {
        WallTimer timer(flux_seconds);
        cpu_face_fluxes({g, nullptr, g.faces, u, u0, buffer.data(), edge, boundary, false});
    }
    // Второй проход по узлам: каждый поток читает готовые вклады и пишет только строку своего узла.
    parallel_for(
        g.nodes,
        [&](Index i) {
            if (!assemble_node_residual(g, i, buffer.data(), g.node_row(r, i)))
                failure(g.faces + i, g);
        },
        256);
}
}
