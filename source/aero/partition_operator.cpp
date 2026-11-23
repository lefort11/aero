#include <iostream>
#include "partition_operator.h"
#include "../residual/boundary_flux_failure.h"
#include "../bc/profile.h"
#include "../io/settings.h"
#include "partition_kernels.h"
#include "../parallel/parallel_for.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../implicit/block_layout.h"
#include "../implicit/spatial_jacobian.h"
#include "../parallel/device_check.h"
#include "../parallel/device_work.h"
#include "../io/fields.h"
#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/jacobian.h"
#endif
#endif
#include "../residual/flux_cfl.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace aero {
namespace edge_tile = residual_detail::edge_tile;
using Id = MeshIndex;
using residual_detail::Surface;

PartitionOperator::PartitionOperator(const MeshPartition& part, CvView cv, const PeriodicPartition* periodic,
                                     const BoundaryPatches& patches, ResidualOptions options,
                                     const ResidualFrame& frame, const Processes& world,
                                     StencilRequests* requests, bool cuda, bool cell_height_cfl)
    : nodes_(world.size, cv.dimension + 2), edges_(world.size, cv.dimension + 3), width_(cv.dimension + 2),
      volumes_(periodic ? periodic->volumes.data() : cv.volumes),
      heights_(cell_height_cfl ? (periodic ? periodic->min_heights.data() : cv.min_heights) : nullptr),
      ids_(periodic ? periodic->ids : part.nodes),
      owners_(periodic ? periodic->owners : part.node_owners), world_(world) {
    const auto mesh = part.mesh.view();
    for (Id node = 0; node < ids_.size(); ++node) {
        const int owner = owners_[node];
        if (owner == world.rank) {
            owned.push_back(node);
        } else {
            nodes_.imports[owner].push_back(node);
        }
    }
    add_edge_surfaces(mesh, cv, periodic, frame, requests);
    const auto edge_count = surfaces_.size();
    add_boundary_surfaces(cv, periodic, part.physical_patches, patches, frame, options.gamma, requests);

    // Ядра ожидают сначала все рёбра в порядке сетки, затем части физических границ.
    residual_detail::build_incidence(ids_.size(), edge_count, surfaces_, offsets_, incidence_);
    grid_ = {
        mesh.dimension,    ids_.size(),    edge_count, surfaces_.size(), surfaces_.data(), offsets_.data(),
        incidence_.data(), patches.view(boundary_profiles_), {},         options, {}, false};
    if (!cuda) {
        flux_.resize(surfaces_.size());
        normals_.resize(edge_count);
        parallel_for(edge_count, [&](Id i) {
            if (euler_unit_direction(mesh.dimension, surfaces_[i].area, normals_[i].axis) != EulerStatus::Success)
                throw std::domain_error("Invalid edge normal");
        });
        grid_.packed.normals = normals_.data();
    }
    residual_detail::validate_linear_walls(grid_);
    prepare_nodal_gas();
}

void PartitionOperator::add_edge_surfaces(MeshView mesh, CvView cv, const PeriodicPartition* periodic,
                                          const ResidualFrame& frame, StencilRequests* requests) {
    const Id count = periodic ? periodic->edges.size() : mesh.edge_count;
    const bool local = world_.size == 1;
    if (local) {
        // На одном процессе номер ребра уже является номером поверхности.
        // Оставляем место и для границы, чтобы её добавление не копировало весь массив рёбер.
        const Id boundary = periodic ? periodic->boundary.size() : cv.boundary_count;
        surfaces_.reserve(count + boundary);
        compute_.reserve(count + boundary);
        surfaces_.resize(count);
        compute_.resize(count);
        if (requests) requests->edges.resize(count);
    }
    const auto prepare_edge = [&](Id edge) {
        const Id begin = periodic ? periodic->edges[edge].begin : mesh.edge_begin[edge];
        const Id end = periodic ? periodic->edges[edge].end : mesh.edge_end[edge];
        const int sender = owners_[begin];
        const int receiver = owners_[end];
        if (sender != world_.rank && receiver != world_.rank) {
            return;
        }

        const Id index = local ? edge : surfaces_.size();
        Surface surface{};
        surface.begin = begin;
        surface.end = end;
        surface.use_nodal_traces = true;
        const auto& area = periodic ? periodic->edges[edge].area : cv.edge_areas[edge];
        residual_detail::project_surface(surface, area, mesh.dimension, frame);
        if (local) surfaces_[index] = surface;
        else surfaces_.push_back(surface);

        // Поток вычисляется ровно один раз на владельце начала ребра.
        if (sender != world_.rank) {
            edges_.imports[sender].push_back(index);
            return;
        }
        if (local) compute_[index] = index;
        else compute_.push_back(index);
        if (requests) {
            const auto original =
                periodic ? periodic->edges[edge].original : std::make_pair(ids_[begin], ids_[end]);
            const StencilEdge request{index, begin, end, original};
            if (local) requests->edges[index] = request;
            else requests->edges.push_back(request);
        }
        if (receiver != sender) {
            edges_.exports[receiver].push_back(index);
        }
    };
    // На целой сетке каждый поток пишет в свою готовую позицию. При обмене
    // через MPI компактные списки соседей заполняются в прежнем порядке.
    if (local) parallel_for(count, prepare_edge);
    else for (Id edge = 0; edge < count; ++edge) prepare_edge(edge);
}

void PartitionOperator::add_boundary_surfaces(CvView cv, const PeriodicPartition* periodic,
                                              Id physical_patches, const BoundaryPatches& patches,
                                              const ResidualFrame& frame, double gamma, StencilRequests* requests) {
    const Id count = periodic ? periodic->boundary.size() : cv.boundary_count;
    for (Id index = 0; index < count; ++index) {
        const auto& piece = periodic ? periodic->boundary[index] : cv.boundary[index];
        if (piece.patch >= physical_patches || owners_[piece.node] != world_.rank) {
            continue;
        }
        if (requests && residual_detail::needs_nodal_traces(patches.by_patch[piece.patch].type())) {
            requests->inlet_nodes.insert(ids_[piece.node]);
        }
        Surface surface{};
        append_boundary_profile(boundary_profiles_,patches,piece,cv.dimension,frame.axis,gamma);
        surface.begin = piece.node;
        surface.patch = piece.patch;
        residual_detail::project_surface(surface, piece.area, cv.dimension, frame);
        compute_.push_back(surfaces_.size());
        surfaces_.push_back(surface);
    }
}

Id PartitionOperator::state_nodes() const {
    return ids_.size() + stencil_.extra_nodes.size();
}

PartitionWork PartitionOperator::work(Id leaves) const {
    // В compute_ граничные площадки идут сразу после рёбер, принадлежащих процессу.
    const Id edges = compute_.size() - (surfaces_.size() - grid_.edges);
    return {leaves, owned.size(), state_nodes() - owned.size(), edges, nodes_.bytes(), edges_.bytes()};
}

namespace {
using residual_detail::PackedEdgeOrder;
using residual_detail::WenoEdgeTile;

// У полной пары обе строки имеют запрошенный порядок. Только ядро чистой
// MPWENO5 умеет в одной плитке совместить полную сторону и укороченную WENO3.
PackedEdgeOrder packed_side_order(unsigned requested, unsigned left, unsigned right, bool allow_mixed) {
    if (left == requested && right == requested)
        return PackedEdgeOrder::from_sides(false, false);
    if (allow_mixed && (left == 3 || left == 5) && (right == 3 || right == 5))
        return PackedEdgeOrder::from_sides(left == 3, right == 3);
    return {}; // Эту пару обработает общий обход разреженных строк.
}

// Упаковываем общий центральный и два ближних наклона. Пятый порядок добавляет два дальних.
// Возвращаем порядки сторон только после успешной упаковки. Пустая запись
// оставит ребро в списке для прохода по исходным разреженным строкам.
template <unsigned scheme_order>
PackedEdgeOrder pack_weno_edge(const PartitionStencil& stencil, MeshIndex edge, WenoEdgeTile<scheme_order>& tile,
                               unsigned lane, bool allow_mixed) {
    const unsigned left = stencil.weno_order[2 * edge], right = stencil.weno_order[2 * edge + 1];
    const auto order = packed_side_order(scheme_order, left, right, allow_mixed);
    if (!order.ready())
        return {};
    const auto& rows = stencil.slopes;
    // В исходной записи по четыре наклона с каждой стороны. Ближний правый наклон при WENO5
    // доступен как противоположный у левой стороны (2), при WENO3 — как собственный у правой (4 + 1).
    unsigned term = 0, weight = 0;
    for (unsigned r = 0; r < tile.rows_per_edge; ++r) {
        const auto row = 8 * edge + residual_detail::edge_slope_offset(r, left);
        const auto first = rows.offsets[row], end = rows.offsets[row + 1];
        const unsigned capacity = residual_detail::slope_terms(r);
        const bool unused = (r == residual_detail::FarLeftSlope && order.shortened(0)) ||
                            (r == residual_detail::FarRightSlope && order.shortened(1));
        // Укороченной стороне дальний наклон не нужен. Для неё берём центральный
        // узел и нулевые веса. Такую же нулевую разность дописываем, если луч
        // попал точно в вершину и в строке оказалось меньше слагаемых.
        const auto terms = unused ? 0 : end - first;
        if (!unused && (terms == 0 || terms > capacity))
            return {};
        const auto anchor = unused ? rows.offsets[8 * edge] : first;
        for (unsigned j = 0; j < capacity; ++j, ++term) {
            const auto node = j < terms ? first + j : anchor;
            tile.nodes(term, lane) = static_cast<unsigned>(rows.nodes[node]);
            if (j)
                tile.weights(weight++, lane) = j < terms ? rows.weights[node] : 0.;
        }
    }
    return order;
}
}

void PartitionOperator::prepare_nodal_gas() {
    if (grid_.options.background.active()) return;
    grid_.packed.state_nodes = state_nodes();
    if (!flux_.empty()) {
        nodal_gas_.resize(state_nodes());
        grid_.packed.gas = nodal_gas_.data();
    }
}

template <unsigned scheme_order>
void PartitionOperator::prepare_weno_tiles(std::vector<WenoEdgeTile<scheme_order>>& tiles) {
    auto& g = grid_;
    // На CPU пакеты используются для характеристик. У GPU есть также ядро,
    // которое читает консервативные переменные или потоки из того же пакета.
    const bool on_gpu = flux_.empty();
    const bool characteristic = g.options.variables == ResidualVariables::Characteristic;
    if (g.options.background.active() || g.dim != 2 || (!on_gpu && !characteristic) ||
        g.options.solver != ResidualSolver::Roe || stencil_.weno_order.empty() ||
        state_nodes() > std::numeric_limits<unsigned>::max())
        return;
    // На CPU потоки площадок хранятся в flux_. На GPU для малой сетки один общий запуск
    // быстрее двух запусков для полных и остальных шаблонов. Веса реконструкции те же.
    const MeshIndex small_gpu_edge_limit = 8192;
    if (on_gpu && g.edges <= small_gpu_edge_limit)
        return;
    const auto count = edge_tile::count(g.edges);
    const bool allow_mixed = scheme_order == 5 && on_gpu && characteristic &&
                             g.stencil.mp5.weno && g.options.mp6_hybrid == 0;
    tiles.resize(count);
    parallel_for(count, [&](MeshIndex t) {
        auto& tile = tiles[t];
        for (unsigned lane = 0; lane < edge_tile::width; ++lane) {
            const auto edge = edge_tile::edge(t,lane);
            tile.order[lane] = edge < g.edges && !g.surfaces[edge].use_nodal_traces
                ? pack_weno_edge(stencil_, edge, tile, lane, allow_mixed) : PackedEdgeOrder{};
        }
    });
    bool have_packed_edges = false;
    for (const auto face : compute_) {
        if (face < g.edges) {
            const auto& tile = tiles[edge_tile::index(face)];
            const unsigned lane = edge_tile::lane(face);
            if (tile.order[lane].complete()) {
                have_packed_edges = true;
                continue;
            }
            // При разных порядках сторон GPU возьмёт ребро в общий пакет MPWENO5.
            // Сохраняем его место в списке остальных площадок для обычного пути.
            have_packed_edges |= tile.order[lane].ready();
        }
        other_faces_.push_back(face);
    }
    if (!have_packed_edges) {
        tiles.clear();
        return;
    }
    // Границы уже идут после внутренних рёбер в compute_. Разделяем оставшиеся рёбра
    // по способу расчёта следов один раз, до первой стадии. Это верно и для списка MPI.
    auto boundary = std::find_if(other_faces_.begin(), other_faces_.end(), [&](Id f) { return f >= g.edges; });
    auto nodal = std::stable_partition(other_faces_.begin(), boundary,
                                       [&](Id f) { return !g.surfaces[f].use_nodal_traces; });
    g.packed.mixed_edges = 0;
    if (allow_mixed) {
        // Уже включённые в пакет рёбра перемещаем в хвост разреженного списка.
        const auto mixed = std::stable_partition(other_faces_.begin(), nodal, [&](Id f) {
            return !tiles[edge_tile::index(f)].order[edge_tile::lane(f)].ready();
        });
        g.packed.mixed_edges = static_cast<Id>(nodal - mixed);
    }
    g.packed.sparse_edges = static_cast<Id>(nodal - other_faces_.begin());
    g.packed.nodal_edges = static_cast<Id>(boundary - nodal);
    g.packed.tiles = tiles.data();
    g.packed.order = scheme_order;
    g.packed.remaining = other_faces_.data();
    g.packed.count = other_faces_.size();
}

Id PartitionOperator::prepare_stencil(const MeshPartition& part, const PeriodicPartition* seam,
                                      const std::vector<PeriodicPair>& pairs, PartitionInput* input,
                                      const StencilRequests& requests, const ReconstructionOptions& options) {
    stencil_ = prepare_partition_stencil(part, seam, pairs, input, requests, grid_.edges, options, world_);
    world_.local([&] {
        parallel_for(requests.edges.size(), [&](Id i) {
            const auto& edge = requests.edges[i];
            surfaces_[edge.surface].use_nodal_traces = stencil_.nodal_traces[edge.surface] != 0;
        });
        for (Id j = 0; j < stencil_.extra_nodes.size(); ++j) {
            nodes_.imports[stencil_.extra_nodes[j].second].push_back(ids_.size() + j);
        }
        grid_.stencil = stencil_.view(state_nodes());
        prepare_nodal_gas();
        if (options.scheme == ReconstructionScheme::Weno5 || reconstruction_scheme(options.scheme).family == ReconstructionFamily::Mp)
            prepare_weno_tiles(weno5_tiles_);
        else if (options.scheme == ReconstructionScheme::Weno3)
            prepare_weno_tiles(weno3_tiles_);
    });
    return world_.sum(stencil_.incomplete);
}

#ifdef AERO_HAS_VISCOSITY
void PartitionOperator::prepare_viscosity(const MeshPartition& part, const PeriodicPartition* seam,
        const ResidualFrame& frame, const RunSettings& run) {
    const auto& settings=run.viscosity;
    if (settings.law == "off") return;
    compact_jacobian_ = run.implicit_settings.compact_jacobian;
    // Периодический сосед может находиться вне конвективного шаблона. Включаем его
    // в уже существующий обмен полями, один раз для всех элементов этого процесса.
    std::map<Id,Id> slots;
    // На целой сетке глобальный номер обычно совпадает с локальным.
    // Проверяем это один раз, до обхода всех вершин вязких элементов.
    bool direct = world_.size == 1 && !seam && stencil_.extra_nodes.empty();
    for (Id i = 0; direct && i < ids_.size(); ++i) direct = ids_[i] == i;
    world_.local([&] {
        aes_.reset(new viscosity::Aes);
        if (run.use_cuda) aes_->prepare_device_geometry();
#ifdef AERO_HAS_TURBULENCE
        aes_->configure_les(run.turbulence,settings.prandtl,run.use_cuda,
                            world_.size==1 && !run.uses_implicit_time());
#endif
        if (!direct) for (Id i = 0; i < ids_.size(); ++i) slots.emplace(ids_[i],i);
        for (Id k = 0; k < stencil_.extra_nodes.size(); ++k)
            slots.emplace(stencil_.extra_nodes[k].first,ids_.size()+k);
    });
    const auto locate = [&](Id global, int owner) {
        if (direct) return global;
        const auto inserted = slots.emplace(global,state_nodes());
        const Id local = inserted.first->second;
        if (inserted.second) {
            if (owner == world_.rank) throw MeshError("AES: missing owned node");
            stencil_.extra_nodes.emplace_back(global,owner);
            nodes_.imports[owner].push_back(local);
        }
        return local;
    };
    aes_->prepare(part,seam,frame,world_,settings,grid_.options.gamma,run.mach,grid_.nodes,locate);
    world_.local([&] {
        for (Id face = grid_.edges; face < grid_.faces; ++face) {
            const auto& s = surfaces_[face];
            if (grid_.patches.by_patch[s.patch].type() == BoundaryType::Wall)
                aes_->wall[s.begin] = 1;
        }
        if (!run.use_cuda) {
            aes_->allocate_host(state_nodes());
            grid_.options.diffusion = aes_->view().rows;
            grid_.options.wall = aes_->wall.data();
        }
        prepare_nodal_gas();
        if (!stencil_.row_owner.empty()) grid_.stencil = stencil_.view(state_nodes());
    });
}
// При задании u=0 убираем только кинетическую энергию: начальные p и T сохраняются.
void PartitionOperator::apply_wall(std::vector<double>& field) const {
    for (Id node : owned) {
        if (!aes_->wall[node]) continue;
        double* u = field.data()+node*width_;
        for (unsigned m = 1; m+1 < width_; ++m) {
            u[width_-1] -= .5 * (u[m]/u[0])*u[m];
            u[m] = 0.;
        }
    }
}
#endif

void PartitionOperator::connect(bool cuda, bool implicit_time) {
    Processes::Messages node_requests, edge_keys;
    world_.local([&] {
        node_requests.resize(world_.size);
        edge_keys.resize(world_.size);
        for (int p = 0; p < world_.size; ++p) {
            for (Id i : nodes_.imports[p]) {
                const Id global = i < ids_.size() ? ids_[i] : stencil_.extra_nodes[i - ids_.size()].first;
                mdec::append_word(node_requests[p], global);
            }
            for (Id f : edges_.exports[p]) {
                mdec::append_word(edge_keys[p], ids_[surfaces_[f].begin]);
                mdec::append_word(edge_keys[p], ids_[surfaces_[f].end]);
            }
        }
    });
    const auto requests = world_.exchange(node_requests);
    const auto keys = world_.exchange(edge_keys);
    // Один раз сверяем порядок по глобальным номерам. На шагах MPI уже передаёт только числа.
    world_.local([&] {
        for (int p = 0; p < world_.size; ++p) {
            for (std::size_t k = 0; k < requests[p].size(); k += 8) {
                const Id global = mdec::read_word(requests[p], k);
                const auto it = std::lower_bound(ids_.begin(), ids_.end(), global);
                if (it == ids_.end() || *it != global) {
                    throw MeshError("Missing requested state");
                }
                const auto local = static_cast<Id>(it - ids_.begin());
                if (owners_[local] != world_.rank) {
                    throw MeshError("Invalid state owner");
                }
                nodes_.exports[p].push_back(local);
            }
            if (keys[p].size() % 16 || keys[p].size() / 16 != edges_.imports[p].size()) {
                throw MeshError("Interface edge count mismatch");
            }
            std::size_t k = 0;
            for (Id f : edges_.imports[p]) {
                if (mdec::read_word(keys[p], k) != ids_[surfaces_[f].begin] ||
                    mdec::read_word(keys[p], k + 8) != ids_[surfaces_[f].end]) {
                    throw MeshError("Interface edge order mismatch");
                }
                k += 16;
            }
        }
        nodes_.allocate(cuda);
        edges_.allocate(cuda);
        if (need_pressure_rate_ && world_.size > 1) {
            pressure_exchange_.reset(new PartitionTransfers(nodes_, 1));
            pressure_exchange_->allocate(false); // Поля для файлов собираются в памяти CPU.
        }
#ifdef AERO_HAS_TURBULENCE
        if(sa_active() || (aes_ && aes_->view().computes_les())) {
            turbulence_exchange_.reset(new PartitionTransfers(nodes_,1));
            turbulence_exchange_->allocate(cuda);
        }
        if (sa_high_order_) {
            sa_edge_exchange_.reset(new PartitionTransfers(edges_,1));
            sa_edge_exchange_->allocate(cuda);
        }
#endif
        if (implicit_time) {
            derivative_exchange_.reset(new PartitionTransfers(edges_, 2 * width_ * width_));
            derivative_exchange_->allocate(cuda);
            if (!cuda) linear_input_.resize(state_nodes() * width_);
        }
    });
}

#ifdef AERO_WITH_KOKKOS_GPU
void PartitionOperator::layout_gpu(bool compact_buffer) {
#ifdef AERO_HAS_VISCOSITY
    gpu_->set_viscosity(aes_.get());
#endif
    gpu_->layout(grid_, compute_, owned, volumes_, state_nodes(), compact_buffer, world_.size == 1);
    if (heights_) device_heights_ = gpu_->layout_node_scalars(grid_.nodes, heights_);
    if (need_pressure_rate_) device_pressure_rate_ = gpu_->layout_node_scalars(owned.size());
    if (steady_steps_) {
        steady_steps_->values = gpu_->layout_node_scalars(grid_.nodes);
        if (rate_control_) rate_history_ = gpu_->layout_node_scalars(LocalRateLimit::Width*owned.size());
    }
    if (neighbour_limit_) {
        neighbour_limit_->scratch = gpu_->layout_node_scalars(grid_.nodes);
        if (neighbour_limit_->halo) neighbour_limit_->halo->prepare_gpu(*gpu_);
    }
    if (track_time_) device_moments_ = gpu_->layout_moments(grid_.nodes);
    nodes_.prepare_gpu(*gpu_);
    edges_.prepare_gpu(*gpu_);
#ifdef AERO_HAS_TURBULENCE
    if(turbulence_exchange_) turbulence_exchange_->prepare_gpu(*gpu_);
    if (sa_high_order_) sa_correction_ = gpu_->layout_node_scalars(grid_.edges);
    if (sa_edge_exchange_) sa_edge_exchange_->prepare_gpu(*gpu_);
#endif
    if (derivative_exchange_) {
        derivative_exchange_->prepare_gpu(*gpu_);
        gpu_->layout_linear_input(state_nodes() * width_);
    }
}

#ifdef AERO_HAS_VISCOSITY
// Для компактного неявного расчёта на одной карте все эти поля вычисляются
// заново после каждой поправки. Предиктор и сборка якобиана их не занимают:
// вектора начинают запись только при входе в сам BiCGSTAB.
static implicit::KrylovScratch krylov_workspace(const residual_detail::Geometry& g,
        bool contiguous, EulerFluxResult* flux=nullptr, viscosity::Gas* gas=nullptr, viscosity::Row* rows=nullptr) {
    implicit::KrylovScratch result;
    const auto unknowns=g.nodes*g.components();
    if (contiguous) {
        // Режим доступен для одного процесса: все локальные узлы являются собственными.
        result.borrow(flux,FluxWorkspace(g.faces,g.nodes,g.nodes).bytes,unknowns);
        return result;
    }
    result.borrow(flux,g.faces*sizeof(EulerFluxResult),unknowns);
    result.borrow(rows,g.nodes*sizeof(viscosity::Row),unknowns);
    result.borrow(gas,g.nodes*sizeof(viscosity::Gas),unknowns);
    return result;
}
#endif

// Верхняя граница дополнительной памяти BDF1. У кратных рёбер CSR может объединить
// несколько вхождений в один блок. Оценка относится только к собственным строкам.
static MeshIndex estimated_matrix_bytes(const residual_detail::Geometry& g, const std::vector<MeshIndex>& owned,
                                        const implicit::NewtonSettings& settings) {
    const MeshIndex width = g.components(), block = width * width * sizeof(double);
    MeshIndex columns = owned.size(); // Одна диагональ на узел, один внедиагональный блок на соседа.
    for (auto node : owned)
        for (auto k = g.offsets[node]; k < g.offsets[node + 1]; ++k)
            if (g.incidence[k].face() < g.edges) columns = partition_bytes(columns, 1, 1);
    const MeshIndex neighbours = columns - owned.size();
    const MeshIndex neighbour_block = width * width * sizeof(implicit::LinearCoefficient);
    MeshIndex bytes = partition_bytes(0, columns, sizeof(MeshIndex));
    bytes = partition_bytes(bytes, owned.size(), block);
    bytes = partition_bytes(bytes, neighbours, neighbour_block);
    if (implicit::mixed_linear || settings.csgs)
        bytes = partition_bytes(bytes, columns, sizeof(MeshIndex)); // Адреса в раздельной или переставленной матрице.
    bytes = partition_bytes(bytes, owned.size() + 1, sizeof(MeshIndex));       // Смещения строк.
    bytes = partition_bytes(bytes, g.offsets[g.nodes], sizeof(MeshIndex));    // Вхождение грани в строку.
    return bytes;
}

static MeshIndex implicit_storage_bytes(const residual_detail::Geometry& g, const std::vector<MeshIndex>& owned,
                                        const implicit::NewtonSettings& settings, MeshIndex gradient_bytes,
                                        const implicit::GpuMatrixPlan* plan=nullptr, unsigned shared_vectors=0) {
    const MeshIndex width=g.components(), block=width*width*sizeof(double);
    // Для компактного якобиана уже известны все столбцы и выравнивание цветных проходов.
    MeshIndex bytes=plan ? implicit::gpu_matrix_bytes(*plan) : estimated_matrix_bytes(g,owned,settings);
    // Производные по концам ребра занимают два блока, граничная площадка — один.
    // До них в той же рабочей области собираются вязкие градиенты.
    const auto face_bytes = partition_bytes(partition_bytes(0, g.faces, block), g.edges, block);
    bytes = partition_bytes(bytes, 1, plan ? gradient_bytes : std::max(face_bytes, gradient_bytes));
    // В компактной раскладке решатель раскладывает исходный диагональный блок
    // в регистрах потока. Постоянная копия LU нужна только обычной раскладке.
    if (!shared_vectors) bytes = partition_bytes(bytes, owned.size(), block + width * sizeof(unsigned char));
    // История BDF2 и промежуточное поле CSGS выделяются по запросу.
    const auto capacity = std::max<MeshIndex>(1,owned.size()*width);
    if (g.options.background.active() || settings.jacobian_update_every) bytes = partition_bytes(bytes, owned.size(), width*sizeof(double));
    if (settings.second_order) bytes=partition_bytes(bytes,capacity,sizeof(double));
    if (settings.csgs && !plan) {
        bytes=partition_bytes(bytes,capacity,sizeof(double));
        bytes=partition_bytes(bytes,owned.size(),2*sizeof(MeshIndex)); // Порядок и обратный порядок.
        bytes=partition_bytes(bytes,g.nodes,sizeof(MeshIndex));       // Полный узел -> собственная строка.
    }
    if (owned.empty()) return partition_bytes(bytes, 2, sizeof(double)); // Рабочие адреса двух пустых буферов.
    return partition_bytes(bytes, owned.size(), (implicit::NewtonFieldCount + implicit::KrylovFieldCount-shared_vectors) *
                                                width * sizeof(double));
}

void PartitionOperator::reserve_gpu(bool compact_buffer, GpuMemoryFit& fit, const implicit::NewtonSettings* settings) {
    shared_csgs_ = settings && settings->csgs;
    MeshIndex extra = 0, gradient_bytes = 0;
#ifdef AERO_HAS_VISCOSITY
    if (settings && aes_) {
        const auto v = aes_->view();
        // На смешанной сетке часть пар добавляет новые столбцы; считаем верхнюю границу до резервирования карты.
        if (!settings->compact_jacobian)
            extra = partition_bytes(extra,v.neighbour_count,
                sizeof(Id)*(1u+unsigned(implicit::mixed_linear || settings->csgs))
                +width_*width_*sizeof(implicit::LinearCoefficient));
        extra = partition_bytes(extra,state_nodes()+1,sizeof(Id));
        extra = partition_bytes(extra,v.vertex_count,sizeof(viscosity::ColumnElement));
        gradient_bytes = partition_bytes(0,v.vertex_count,sizeof(viscosity::Gradient));
        if (settings->csgs) extra = partition_bytes(extra,state_nodes()-grid_.nodes,sizeof(Id));
    }
#endif
    if (settings) extra = partition_bytes(extra, 1, implicit_storage_bytes(grid_, owned, *settings, gradient_bytes));
    const auto try_layout = [&] {
        gpu_.reset(new PartitionGpu);
        layout_gpu(compact_buffer);
        fit.failed = !gpu_->reserve(fit, extra);
    };
    try_layout();
    // На тесной карте расчёт может поместиться с исходными разреженными строками, без дополнительных буферов.
    if (fit.failed && grid_.packed.state_nodes) {
        grid_.packed = {};
        try_layout();
    }
#ifdef AERO_HAS_VISCOSITY
    // Обе компактные раскладки собирают блоки в готовом графе матрицы.
    // Выбираем их по объёму фактических таблиц после осмотра связности сетки.
    const auto try_compact_geometry = [&] {
        if (world_.size==1 && settings && settings->compact_jacobian) {
            // Однопроцессная сборка берёт адреса блоков из готового графа CSR и
            // вычисляет производную в строке. Для обработки по группам буфер
            // градиентов уже входит в память AES, второй раз его не учитываем.
            if (!matrix_plan_) {
                implicit::Partition topology;
                topology.compact_jacobian=true;
                topology.shared_csgs=shared_csgs_;
                topology.host_aes=aes_->view();
                topology.host_aes.node_count=state_nodes();
                matrix_plan_=implicit::gpu_matrix_plan(grid_,owned.size(),topology);
            }
            extra=implicit_storage_bytes(grid_,owned,*settings,aes_->has_gradient_tiles() ? 0 : gradient_bytes,
                                         matrix_plan_.get(),krylov_workspace(grid_,aes_->uses_implicit_coordinates()).count);
        }
        try_layout();
    };
    if (fit.failed && aes_ && aes_->use_affine_geometry()) {
        const auto previous=fit.layout_bytes;
        try_compact_geometry();
        std::cout << "GPU rank " << world_.rank << ": compact AES geometry, saved "
                  << previous-fit.layout_bytes << " bytes" << std::endl;
    }
    const bool coordinate_jacobian = world_.size==1 && settings && settings->compact_jacobian;
    if (fit.failed && (!settings || coordinate_jacobian) && aes_ && aes_->use_coordinate_geometry(settings != nullptr)) {
        try_compact_geometry();
        std::cout << "GPU rank " << world_.rank << ": AES using resident vertex coordinates" << std::endl;
    }
    if (aes_ && !fit.failed) aes_->finish_device_geometry();
#endif
}

void PartitionOperator::upload_state(const std::vector<double>& state, const std::vector<double>& initial) {
    gpu_->upload_state(state, initial);
}
#endif

void PartitionOperator::copy_backup(std::vector<double>& state, std::vector<double>& backup, bool restore) {
    if (restore) gas_current_ = false;
#ifdef AERO_HAS_TURBULENCE
    if(sa_active()) {
        if(restore) sa_dt_=0.;
        sa_apply(sa_step(),restore?turbulence::Operation::RestoreBackup:turbulence::Operation::SaveBackup);
    }
#endif
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) { gpu_->copy_backup(state.size(), restore); return; }
#endif
    if (restore) state = backup;
    else backup = state;
}

void PartitionOperator::failed_nodes(const std::vector<double>& state, std::vector<Id>& failed, bool implicit) {
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) { gpu_->failed_nodes(failed, implicit); return; }
#else
    (void)implicit;
#endif
    for (Id node : owned)
        if (!partition_detail::admissible_node(grid_, state.data(), node)) failed.push_back(node);
}

// Пересобираем матрицу только при ручном изменении потока. Для BDF2 переносим предыдущий слой.
void PartitionOperator::set_flux_weights(double upwind, double hybrid) {
    if (grid_.options.upw_coeff == upwind && grid_.options.mp6_hybrid == hybrid) return;
    grid_.options.upw_coeff = upwind;
    grid_.options.mp6_hybrid = hybrid;
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) gpu_->set_flux_options(grid_.options);
#endif
    if (bdf_) {
        implicit::BdfHistory history;
        // После пересборки исчезает сохранённое начало прошлого шага и его невязка.
        previous_cfl_ = 0.;
        reset_rate_history();
        bdf_->export_history(history);
        bdf_.reset();
        if (history.dt > 0.) implicit_step().import_history(history);
    }
}

void PartitionOperator::set_courant(double cfl, bool implicit_time) {
    grid_.stencil.mp5.alpha = mp5_alpha_for_time_scheme(cfl, implicit_time);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) gpu_->set_alpha(grid_.stencil.mp5.alpha);
#endif
}

void PartitionOperator::set_low_order(std::vector<double>& marks) {
    // Тот же обмен, что у состояния. Отметка хранится в первом компоненте строки.
#ifdef AERO_WITH_KOKKOS_GPU
    // Последняя распаковка состояния может ещё читать закреплённый буфер MPI.
    // Дожидаемся её до повторного использования этого буфера на CPU для меток.
    if (gpu_) world_.local([&] { gpu_->synchronize(); });
#endif
    halo_host(marks);
    for (Id e = 0; e < grid_.edges; ++e) {
        auto& face = surfaces_[e];
        if (face.use_nodal_traces == 1) continue; // Постоянное условие открытой границы.
        face.use_nodal_traces = (marks[width_ * face.begin] ? 2 : 0) |
                                (marks[width_ * face.end] ? 4 : 0);
    }
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) gpu_->update_surfaces(surfaces_);
#endif
}

void PartitionOperator::copy_owned(const std::vector<double>& state, std::vector<double>& compact) const {
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        gpu_->download_owned(compact);
        return;
    }
#endif
    parallel_for(owned.size(), [&](std::size_t k) {
        std::copy_n(state.data() + width_ * owned[k], width_, compact.data() + width_ * k);
    });
}

void PartitionOperator::download(std::vector<double>& state) const {
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        gpu_->download(state);
    }
#else
    (void)state;
#endif
}

void PartitionOperator::halo(std::vector<double>& u, double* seconds) {
    double* field = u.data();
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) field = gpu_->active_grid().state;
#endif
    nodes_.transfer(world_, {width_, field, nullptr}, device(), seconds);
#ifdef AERO_HAS_TURBULENCE
    if(sa_active()) sa_exchange();
#endif
}

// Накопление выполняется там же, где хранится принятое состояние. На GPU
// за один шаг читаем только узел и его историю, без копирования поля на CPU.
void PartitionOperator::sample_accepted(const std::vector<double>& state, double dt) {
    if (!track_time_) return;
    diagnostic::Sample sample{grid_.dim, grid_.options.gamma, dt, grid_.options.background,
                              moments_.data(), state.data()};
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        sample.values = device_moments_;
        sample.state = gpu_->active_grid().state;
        if (const auto error = device_work::aero_device_sample(sample, grid_.nodes)) throw FieldError(error);
        return;
    }
#endif
    parallel_for(grid_.nodes, [&](MeshIndex i) { sample(i); });
}

void PartitionOperator::copy_samples(std::vector<diagnostic::Moments>& target) const {
    if (!track_time_) { target.clear(); return; }
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        target.resize(grid_.nodes);
        if (!target.empty() && gpu::copy(target.data(),device_moments_,target.size()*sizeof(target[0]),
                                         gpu::Copy::ToHost)!=gpu::Status::Success)
            throw FieldError("Cannot download time averages");
        return;
    }
#endif
    target = moments_;
}

void PartitionOperator::start_samples(const std::vector<double>& state, std::vector<diagnostic::Moments>& saved, bool restored) {
    if (!track_time_) { saved.clear(); return; }
    if (!restored) {
        // На GPU историю уже хранит массив устройства. Память CPU нужна только при записи снимка.
#ifdef AERO_WITH_KOKKOS_GPU
        if (!gpu_)
#endif
            moments_.resize(grid_.nodes);
        sample_accepted(state,0.);
        return;
    }
    // При смене числа процессов в файле есть все собственные строки. Заполняем
    // соседние узлы тем же обменом, что используется для начального поля газа.
    if (world_.size>1) {
        std::vector<double> pack(state.size());
        for (unsigned first=0; first<diagnostic::Moments::words; first+=width_) {
            const unsigned count=std::min<unsigned>(width_,diagnostic::Moments::words-first);
            for (auto i:owned) for (unsigned k=0;k<count;++k) pack[i*width_+k]=saved[i].get(first+k);
            halo_host(pack);
            for (Id i=0;i<grid_.nodes;++i) for (unsigned k=0;k<count;++k) saved[i].set(first+k,pack[i*width_+k]);
        }
    }
    saved.resize(grid_.nodes);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        if (!saved.empty() && gpu::copy(device_moments_,saved.data(),saved.size()*sizeof(saved[0]),
                                        gpu::Copy::ToDevice)!=gpu::Status::Success)
            throw FieldError("Cannot restore time averages on GPU");
        saved.clear();
        return;
    }
#endif
    moments_.swap(saved);
}

std::vector<double> PartitionOperator::output_pressure_rate(const std::vector<double>& state) {
    std::vector<double> compact, result;
    world_.local([&] {
        compact.resize(owned.size());
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            const auto& g = gpu_->active_grid();
            const partition_detail::PressureRate rate{
                g.geometry, g.flux, g.volumes, g.state, g.owned, device_pressure_rate_};
            device_work::check(device_work::aero_device_pressure_rate(rate, owned.size()));
            // Потоки и вязкие члены остаются на карте. Для вывода нужен лишь один скаляр на узел.
            if (!compact.empty())
                gpu::checked(gpu::copy(compact.data(), device_pressure_rate_, compact.size()*sizeof(double),
                                       gpu::Copy::ToHost), "Cannot download dp_dt: ");
        } else
#endif
        {
            const partition_detail::PressureRate rate{
                grid_, flux_.data(), volumes_, state.data(), owned.data(), compact.data()};
            parallel_for(owned.size(), rate);
        }
        if (pressure_exchange_) result.resize(state_nodes());
        for (Id row = 0; row < compact.size(); ++row) {
            if (!std::isfinite(compact[row]))
                throw std::domain_error("Nonfinite dp_dt at global node " + std::to_string(ids_[owned[row]]));
            if (pressure_exchange_) result[owned[row]] = compact[row];
        }
    });
    if (!pressure_exchange_) return compact;
    // Соседу принадлежит полная строка невязки его узла, включая грани за пределами нашей подобласти.
    pressure_exchange_->transfer(world_, {1, result.data(), nullptr});
    return result;
}

// Снимок уже находится на CPU. На физической границе берём значение самого узла,
// поэтому разность между значениями на граничной площадке и в узле равна нулю.
void PartitionOperator::output_gradients(const std::vector<double>& state, std::vector<double>* rotation,
                                         std::vector<double>* schlieren) {
    std::vector<double> buffer(state.size(),0.);
    const auto velocity=[&](Id i,unsigned axis) {
        const double* u=state.data()+i*width_;
        return grid_.options.background.active()?u[axis+1]:u[axis+1]/u[0];
    };
    parallel_for(owned.size(),[&](Id row){
        const auto i=owned[row];
        diagnostic::Curl curl{};
        double gradient[3]{};
        for (auto k=offsets_[i];k<offsets_[i+1];++k) {
            const auto entry=incidence_[k];
            if (entry.face()>=grid_.edges) continue;
            const auto& face=surfaces_[entry.face()];
            const Id j=i==face.begin?face.end:face.begin;
            // Значение в середине ребра отличается от узлового на половину скачка.
            const double factor=entry.outward_flux(.5);
            if (schlieren) {
                // Первая компонента состояния — плотность Euler или пульсация плотности Linear.
                const double density_jump=state[j*width_]-state[i*width_];
                for (unsigned a=0;a<grid_.dim;++a) gradient[a]+=factor*density_jump*face.area[a];
            }
            if (rotation) {
                double jump[3]{};
                for (unsigned a=0;a<grid_.dim;++a) jump[a]=velocity(j,a)-velocity(i,a);
                curl.add(face.area,jump,factor);
            }
        }
        if (rotation) buffer[i*width_]=curl.magnitude(1./volumes_[i]);
        if (schlieren) buffer[i*width_+1]=std::sqrt(euler_detail::dot(grid_.dim,gradient,gradient))/volumes_[i];
    });
    if (world_.size>1) halo_host(buffer);
    if (rotation) rotation->resize(grid_.nodes);
    if (schlieren) schlieren->resize(grid_.nodes);
    for (Id i=0;i<grid_.nodes;++i) {
        if (rotation) (*rotation)[i]=buffer[i*width_];
        if (schlieren) (*schlieren)[i]=buffer[i*width_+1];
    }
}

void PartitionOperator::halo_host(std::vector<double>& u) {
    nodes_.transfer(world_, {width_, u.data(), nullptr});
}

bool PartitionOperator::assemble_fluxes(const std::vector<double>& u, const std::vector<double>& initial,
                                        SolverTimings& timing, bool trial, bool retry_boundary) {
#ifdef AERO_HAS_TURBULENCE
    if(sa_active()) sa_exchange();
    if (aes_ && aes_->view().computes_les()) {
        // У пробной поправки Ньютона рабочая вязкость SA фиксирована. Газ нужен
        // сразу, а длина DDES понадобится только при подтверждении этой поправки.
        const bool frozen_ddes=trial && sa_dt_>0. && aes_->view().c_des>0.;
        update_les(u,timing,frozen_ddes ? viscosity::Evaluation::DdesGasOnly : viscosity::Evaluation::LesClosure);
    }
#endif
    bool defer_viscous_rows=false;
#ifdef AERO_HAS_TURBULENCE
    const bool frozen_sa=sa_dt_>0. && trial;
    defer_viscous_rows=sa_dt_>0. && !frozen_sa;
#endif
    (void)defer_viscous_rows;
    bool physical = true;
    std::vector<Id> failed;
    const auto compute = [&] {
        WallTimer timer(timing.counter(SolverTimings::Flux));
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            gpu_->fluxes(first_order_start,defer_viscous_rows);
            return;
        }
#endif
        // Каждый процесс вычисляет назначенные ему поверхности, включая физические площадки своих узлов.
        const Id* faces = world_.size == 1 ? nullptr : compute_.data();
        residual_detail::FaceBatch batch = {
            grid_, faces, compute_.size(), u.data(), initial.data(), flux_.data(), nullptr, nullptr, gas_current_};
        if (first_order_start) batch.use_first_order();
        residual_detail::cpu_face_fluxes(batch, ids_.data());
#ifdef AERO_HAS_VISCOSITY
        if (aes_) aes_->evaluate(u.data(), grid_.packed.gas, gas_current_ && grid_.packed.gas,
            defer_viscous_rows?viscosity::Evaluation::BeforeSaIterations:viscosity::Evaluation::Full);
#endif
    };
    world_.local([&] {
        try { compute(); }
        catch (const BoundaryFluxFailure& e) {
            if (!trial && !retry_boundary) throw;
            if (!trial) failed.push_back(surfaces_[e.face].begin);
            physical = false;
        }
        catch (const std::domain_error&) {
            if (!trial) throw;
            physical = false;
        }
    });
    // При пробе предиктора достаточно вернуть false. После поправки Ньютона
    // или на стадии SSPRK3 нужен откат всего шага до обмена потоками между рангами.
    if (retry_boundary && !trial) {
        const auto total = world_.size == 1 ? failed.size() : world_.sum(failed.size());
        if (total) throw RejectedGasState{std::move(failed), total};
    }
#ifdef AERO_HAS_TURBULENCE
    const auto update_sa=[&] {
        if (!sa_active()) return;
        sa_prediction_pending_=frozen_sa;
        if (!frozen_sa) evaluate_sa(u,timing);
    };
#endif
    if (world_.size == 1) {
#ifdef AERO_HAS_TURBULENCE
        if (physical) update_sa();
#endif
        return physical;
    }
    // До обмена потоками все процессы должны принять одно решение об откате.
    if (trial && world_.sum(physical ? 0 : 1)) return false;
    // На стыке сосед получает тот же интегральный поток. Знак задаст его узловой список при сборке.
    EulerFluxResult* flux = flux_.data();
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) flux = gpu_->active_grid().flux;
#endif
    edges_.transfer(world_, {width_, nullptr, flux}, device(),
                           timing.counter(SolverTimings::FluxExchange));
#ifdef AERO_HAS_TURBULENCE
    update_sa();
#endif
    return true;
}

void PartitionOperator::retry_auto_step(double failed_dt) {
    // У BDF газ уже восстановлен из U_n. SA имеет собственную копию начала этого шага.
#ifdef AERO_HAS_TURBULENCE
    if (sa_active()) {
        sa_dt_ = 0.;
        sa_prediction_pending_ = false;
        world_.local([&] { sa_apply(sa_step(),turbulence::Operation::RestoreStart); });
        sa_exchange();
    }
#endif
    gas_current_ = false;
    if (++retries_ > 12) throw std::runtime_error("Auto LTS: BiCGSTAB failed after 12 step reductions");
    reduce_auto_attempt(failed_dt, .5);
}

// Ищем владельца один раз. Для отчёта читаем только данные выбранного узла.
void PartitionOperator::set_rate_probe(MeshIndex id) {
    for (MeshIndex row=0; row<owned.size(); ++row)
        if (global_node(owned[row]) == id) { probe_row_ = row; break; }
    const bool owner = probe_row_ != ~MeshIndex(0);
    if (world_.sum(owner ? 1 : 0) != 1)
        throw std::invalid_argument("local_time_step_probe_node has no unique owner");
    probe_owner_ = static_cast<int>(world_.sum(owner ? static_cast<unsigned>(world_.rank) : 0));
}

void PartitionOperator::read_rate_probe(double accepted_dt, double* values) const {
    world_.local([&] {
        if (probe_row_ == ~MeshIndex(0)) return;
        const auto node = owned[probe_row_];
        const auto read = [&](const double* p) {
#ifdef AERO_WITH_KOKKOS_GPU
            if (gpu_) {
                double value;
                gpu::checked(gpu::copy(&value,p,sizeof(value),gpu::Copy::ToHost));
                return value;
            }
#endif
            return *p;
        };
        values[0] = read(steady_steps_->values+node)*(accepted_dt/minimum_step_);
        values[1] = read(bdf_->spatial_rates()+probe_row_);
        values[3] = rate_control_ && rate_samples_ > 1 ? read(rate_history_ + LocalRateLimit::Width*probe_row_ + LocalRateLimit::CflFraction) : 1.;
#ifdef AERO_HAS_TURBULENCE
        if (sa_active()) values[2] = std::abs(read(sa_step().transport.sa.rate+node));
#endif
    });
    if (world_.size > 1) world_.broadcast(values, 4, probe_owner_);
}

void PartitionOperator::set_steady_steps(LocalStepMode mode, double max_ratio) {
    // После переключения между глобальным и локальным шагами начинаем историю auto заново.
    accepted_field_ = nullptr;
    previous_cfl_ = retry_dt_ = 0.;
    retries_ = 0;
    recovered_step_ = false;
    reset_rate_history();
    rate_limited_ = 0;
    step_mode_ = mode;
    steady_steps_->local = mode != LocalStepMode::Off;
    steady_steps_->max_ratio = max_ratio;
}

void PartitionOperator::prepare_steady_steps(bool gpu, LocalStepMode mode, double max_ratio, bool residual_limit) {
    // Адрес описания остаётся постоянным для газа, SA и матрицы на всех шагах.
    steady_steps_.reset(new SteadyNodeSteps);
    rate_control_ = residual_limit;
    set_steady_steps(mode, max_ratio);
    if (!gpu) {
        host_steps_.resize(grid_.nodes);
        steady_steps_->values = host_steps_.data();
        if (rate_control_) {
            host_rate_history_.resize(LocalRateLimit::Width*owned.size());
            rate_history_ = host_rate_history_.data();
        }
    }
}

void PartitionOperator::prepare_neighbour_step_limit(double ratio, bool cuda) {
    if (ratio == 0.) return;
    Processes::Messages needed;
    world_.local([&] {
        neighbour_limit_.reset(new NeighbourStepLimit);
        auto& limit = *neighbour_limit_;
        limit.ratio = ratio;
        if (!cuda) {
            limit.host_scratch.resize(grid_.nodes);
            limit.scratch = limit.host_scratch.data();
        }
        if (world_.size == 1) return;
        limit.halo.reset(new PartitionTransfers(world_.size, 1));
        needed.resize(world_.size);
        for (int p = 0; p < world_.size; ++p) {
            // В обычном обмене газом есть ещё узлы шаблона и вязкости. Для dt
            // берём только удалённые концы настоящих рёбер, смежных с собственным узлом.
            for (Id n : nodes_.imports[p]) {
                const bool adjacent = n < grid_.nodes && offsets_[n] != offsets_[n+1];
                needed[p].push_back(adjacent ? 1 : 0);
                if (adjacent) limit.halo->imports[p].push_back(n);
            }
        }
    });
    if (world_.size == 1) return;
    // Списки nodes_ уже согласованы при connect(). Однократно передаём
    // владельцу маску нужных позиций, сохраняя одинаковый порядок у соседей.
    const auto requested = world_.exchange(needed);
    world_.local([&] {
        auto& halo = *neighbour_limit_->halo;
        for (int p = 0; p < world_.size; ++p) {
            if (requested[p].size() != nodes_.exports[p].size())
                throw MeshError("Invalid neighbouring time-step exchange");
            for (Id i = 0; i < requested[p].size(); ++i)
                if (requested[p][i]) halo.exports[p].push_back(nodes_.exports[p][i]);
        }
        halo.allocate(cuda);
    });
}

void PartitionOperator::limit_neighbour_steps() {
    auto& limit = *neighbour_limit_;
    if (maximum_step_ == minimum_step_) {
        limit.before = limit.after = 1.;
        return; // Все фактические шаги одинаковы, включая случай общего max_ratio=1.
    }
    const auto gas = cfl_grid();
    bool first = true;
    for (;;) {
        double*& current = steady_steps_->values;
        if (limit.halo) limit.halo->transfer(world_, {1,current,nullptr}, device());
        const residual_detail::NeighbourStepSweep sweep{gas.owned, gas.offsets, gas.speeds.incidence,
            gas.speeds.surfaces, grid_.edges, current, limit.scratch, limit.ratio, minimum_step_};
        residual_detail::NeighbourStepRange range{1., 0., 0};
        world_.local([&] {
            if (owned.empty()) return; // Пустой ранг всё равно участвует в обменах и проверке сходимости.
#ifdef AERO_WITH_KOKKOS_GPU
            if (gpu_) {
                device_work::check(device_work::aero_device_neighbour_steps(sweep, owned.size(), &range));
                return;
            }
#endif
            host_parallel::ensure_runtime();
            Kokkos::parallel_reduce("Aero: neighbouring local steps", host_parallel::Range(0,owned.size()), sweep, range);
        });
        if (first) limit.before = world_.reduce_double(range.ratio, true);
        // Минимальный узловой dt не меняется: каждому соседу разрешён как минимум
        // этот шаг. Пропорциональное уменьшение всей попытки тоже сохраняет все отношения.
        if (limit.ratio == 1.) {
            std::swap(current, limit.scratch);
            maximum_step_ = minimum_step_;
            limit.after = 1.;
            return;
        }
        const bool changed = world_.size == 1 ? range.changed != 0 : world_.sum(range.changed) != 0;
        if (!changed) {
            // Записанные значения уже совпадают с исходными. Оставляем поле с заполненными halo.
            maximum_step_ = world_.reduce_double(range.maximum, true);
            limit.after = first ? limit.before : world_.reduce_double(range.ratio, true);
            return;
        }
        // Одновременно переходим к новым шагам на ВСЕХ разделах, даже если
        // на этом ранге пока не было уменьшения. Следующий проход увидит изменения соседа.
        std::swap(current, limit.scratch);
        first = false;
    }
}

// Для общего и локального шага используем те же геометрию, газ и вязкие ограничения.
residual_detail::FluxCfl PartitionOperator::cfl_grid() const {
    const double* sa_rate = nullptr;
#ifdef AERO_HAS_TURBULENCE
    if (sa_active()) sa_rate = sa_step().transport.sa.diagonal;
#endif
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        const auto& g = gpu_->active_grid();
        return {g.geometry, g.flux, g.volumes, g.owned, g.owned_count, sa_rate, device_heights_,g.state};
    }
#endif
    return {grid_, flux_.data(), volumes_, world_.size == 1 ? nullptr : owned.data(), owned.size(), sa_rate, heights_};
}

double PartitionOperator::prepare_local_steps(double cfl) {
    // После отказа значения в узловом массиве остаются прежними. Отношение
    // dt/minimum_step_ в интеграторе уменьшит сразу все строки газа и SA.
    if (retry_dt_ > 0.) {
        current_step_cfl_ = cfl; // После отката газа CFL тоже мог уменьшиться.
        return retry_dt_;
    }
    const bool adapting = step_mode_ == LocalStepMode::Auto && previous_cfl_ > 0.;
    residual_detail::StepRange range{max_finite_double, 0., 0};
    world_.local([&] {
        if (owned.empty()) return;
        AdaptiveStepChange change{};
        if (adapting) {
            change.old_field = bdf_->start_field();
            change.field = accepted_field_;
            change.dim = grid_.dim;
            change.gamma = grid_.options.gamma;
            change.background = grid_.options.background;
            // После тяжёлого решения временно удерживаем именно принятый dt:
            // одновременная прибавка CFL не должна увеличить его в обход запрета.
            const bool hold = recovered_step_;
            const double cfl_change = cfl / previous_cfl_;
            change.fraction = accepted_fraction_ * (hold ? std::min(cfl_change, 1.) : cfl_change);
            change.ceiling = hold ? 1. : 1.25;
#ifdef AERO_HAS_TURBULENCE
            if (sa_active()) {
                const auto sa = sa_step().transport.sa;
                change.old_sa = sa.old;
                change.sa = sa.w;
                change.material = sa.nodal_material;
            }
#endif
        }
        LocalRateLimit feedback{};
        if (adapting && rate_samples_) {
            feedback = {rate_history_, bdf_->spatial_rates(), nullptr, gas_rate_max_, sa_rate_max_,
                        minimum_step_*change.fraction, rate_samples_ % LocalRateLimit::interval == 0};
#ifdef AERO_HAS_TURBULENCE
            if (sa_active()) feedback.sa_rate = sa_step().transport.sa.rate;
#endif
        }
        const residual_detail::LocalSteadySteps steps{cfl_grid(), cfl, change, steady_steps_->values,
                                                    feedback, rate_samples_ == 1};
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            device_work::check(device_work::aero_device_local_steps(steps, &range));
            return;
        }
#endif
        host_parallel::ensure_runtime();
        Kokkos::parallel_reduce("Aero: local steady steps", host_parallel::Range(0, owned.size()), steps, range);
    });
    rate_limited_ = range.limited;
    minimum_step_ = world_.minimum(range.minimum);
    maximum_step_ = world_.reduce_double(range.maximum, true);
    if (!(minimum_step_ > 0. && maximum_step_ >= minimum_step_))
        throw std::domain_error("Invalid local steady time step");
    // Минимум уже учитывает и газ, и SA на всех процессах. Ограничиваем поле один
    // раз перед Ньютоном: матрица, невязка и перенос SA затем читают один шаг узла.
    // В auto ограничиваем уже адаптированный шаг относительно минимума текущего слоя.
    if (steady_steps_->max_ratio >= 1. && minimum_step_ <= max_finite_double / steady_steps_->max_ratio) {
        const double ceiling = minimum_step_ * steady_steps_->max_ratio;
        if (maximum_step_ > ceiling) {
            world_.local([&] {
                if (range.maximum <= ceiling) return;
                const residual_detail::LimitLocalSteps limit{cfl_grid().owned, steady_steps_->values, ceiling};
#ifdef AERO_WITH_KOKKOS_GPU
                if (gpu_) {
                    device_work::check(device_work::aero_device_limit_local_steps(limit, owned.size()));
                    return;
                }
#endif
                Kokkos::parallel_for("Aero: limit local steady steps", host_parallel::Range(0, owned.size()), limit);
            });
            maximum_step_ = ceiling; // В report попадает фактически применённый максимум.
        }
    }
    if (neighbour_limit_) limit_neighbour_steps();
    current_step_cfl_ = cfl;
    return minimum_step_;
}

double PartitionOperator::cfl_step(double cfl) {
    if (steady_steps_ && steady_steps_->local) return prepare_local_steps(cfl);
    // Пустая подобласть не ограничивает шаг. Минимум по процессам выбирает конечное значение.
    double local = std::numeric_limits<double>::infinity();
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            local = residual_detail::gpu_flux_cfl(cfl_grid(), cfl, nullptr);
            return;
        }
#endif
        local = residual_detail::cpu_flux_cfl(cfl_grid(), cfl, ids_.data());
    });
    const double dt = world_.minimum(local);
    if (!std::isfinite(dt)) {
        throw std::domain_error("No finite global CFL step");
    }
    return dt;
}

// Раскладываем только собственные компоненты x, затем используем обычный обмен узлами.
 // На GPU и рабочее поле, и индексы упаковки остаются в памяти выбранной карты.
const double* PartitionOperator::extend_linear_vector(const double* compact) {
    double* field = linear_input_.data();
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            field = gpu_->expand_owned(compact);
            return;
        }
#endif
        const partition_detail::RecordField values{width_, field, nullptr};
        parallel_for(owned.size(), [&](MeshIndex k) { values.copy_record(owned[k], k, compact, nullptr); });
    });
    nodes_.transfer(world_, {width_, field, nullptr}, device());
    return field;
}

implicit::Partition PartitionOperator::linear_partition() {
    implicit::Partition part;
#ifdef AERO_WITH_KOKKOS_GPU
    part.shared_csgs = shared_csgs_;
    part.gpu_matrix_plan = matrix_plan_;
#endif
    part.steady_steps = steady_steps_.get();
#ifdef AERO_HAS_VISCOSITY
    if (aes_) {
        part.host_aes = aes_->view();
        part.compact_jacobian = compact_jacobian_;
        part.host_aes.node_count = state_nodes();
        part.active_aes = part.host_aes;
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) {
            part.active_aes = gpu_->viscosity_view();
            if (world_.size==1 && part.compact_jacobian && (part.active_aes.affine || part.active_aes.tetra.cells)) {
                part.krylov_scratch=krylov_workspace(grid_,aes_->uses_implicit_coordinates(),gpu_->active_grid().flux,
                                                     part.active_aes.gas,part.active_aes.rows);
            }
        }
#endif
    }
#endif
    if (world_.size == 1) return part;
    part.rows = owned.size();
    part.faces = compute_.size();
    part.host_owned = part.owned = owned.data();
    part.computed = compute_.data();
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        part.owned = gpu_->active_grid().owned;
        part.computed = gpu_->active_grid().computed;
    }
#endif
    part.checked = [this](const std::function<void()>& work) { world_.local(work); };
    part.reduce = [this](double value, bool maximum) { return world_.reduce_double(value, maximum); };
    part.extend_vector = [this](const double* x) { return extend_linear_vector(x); };
    part.share_state = [this](double* u) { nodes_.transfer(world_, {width_, u, nullptr}, device()); };
    part.share_blocks = [this](double* blocks) {
        derivative_exchange_->transfer(world_, {2 * width_ * width_, blocks, nullptr}, device());
    };
    return part;
}

implicit::LinearGrid PartitionOperator::linear_grid() {
    int id = -1;
    const auto* active = &grid_;
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        gpu::checked(gpu::current(&id));
        active = &gpu_->active_grid().geometry;
    }
#endif
    return {grid_, *active, id, linear_partition()};
}

implicit::BdfStep& PartitionOperator::implicit_step() {
    if (!bdf_) {
        const auto grid = linear_grid();
        bdf_.reset(new implicit::BdfStep(grid.host, grid.active, grid.device, grid.partition));
    }
    return *bdf_;
}

PartitionOperator::ImplicitFields PartitionOperator::implicit_fields(
        std::vector<double>& state, const std::vector<double>& initial) {
    if (!bdf_) world_.local([&] { implicit_step(); });
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        const auto& fields = gpu_->active_grid();
        return {fields.state, fields.initial, fields.volumes, fields.flux};
    }
#endif
    return {state.data(), initial.data(), volumes_, flux_.data()};
}

double PartitionOperator::steady_residual(std::vector<double>& state, const std::vector<double>& initial, bool accepted) {
    const auto fields = implicit_fields(state, initial);
    const bool sample = accepted && step_mode_ == LocalStepMode::Auto;
    const bool keep_rows = sample && (rate_control_ || probe_row_ != ~MeshIndex(0));
    const double gas = bdf_->spatial_rate_norm(fields.volumes, fields.flux, keep_rows);
    double sa = 0.;
#ifdef AERO_HAS_TURBULENCE
    if (sa_active()) sa = sa_max(sa_step(),turbulence::Norm::Rate);
#endif
    if (sample && rate_control_) {
        gas_rate_max_ = gas;
        sa_rate_max_ = sa;
        ++rate_samples_;
    }
    return std::max(gas, sa);
}

implicit::NewtonResult PartitionOperator::advance_bdf(std::vector<double>& state, const std::vector<double>& initial,
        double dt, const implicit::NewtonSettings& settings, SolverTimings& timing, const std::function<void()>& validate) {
    const auto fields = implicit_fields(state, initial);
    // При попадании в t_fo или срок вывода сокращаем все узловые шаги в одной пропорции.
    // На протяжении Ньютона, включая пробные поправки и обновления SA, шаги уже зафиксированы.
    const double dt_or_fraction = steady_steps_ && steady_steps_->local ? dt / minimum_step_ : dt;
    // При каждой поправке используем тот же поток и ту же реконструкцию из main.txt.
    // В матрицу поправки входят производные выбранного потока первого порядка.
    const auto refresh = [&] {
        if (validate) validate();
        assemble_fluxes(state, initial, timing, false, bool(validate));
    };
    const auto trial = [&] { if (validate) validate(); return assemble_fluxes(state, initial, timing, true); };
    std::function<bool()> confirm_prediction;
#ifdef AERO_HAS_TURBULENCE
    if(sa_active()) {
        sa_prediction_pending_=false;
        confirm_prediction=[&] {
            if (!sa_prediction_pending_) return false;
            // Потоки массы уже построены для нынешнего газа. Доводим SA до заданного
            // допуска, обновляем вязкие напряжения, затем BDF проверяет невязку газа.
            if (aes_->view().c_des>0.) update_les(state,timing,viscosity::Evaluation::FilterOnly);
            evaluate_sa(state,timing,false,true);
            sa_prediction_pending_=false;
            return true;
        };
        sa_dt_=dt_or_fraction; sa_bdf2_=settings.second_order; sa_tol_=settings.newton_relative*.1;
        sa_iterations_=std::max(200u,settings.max_iterations);
        sa_target_=0.;
        sa_apply(sa_step(),turbulence::Operation::SaveStart);
        evaluate_sa(state,timing,true); // Первый остаток Ньютона тоже должен учитывать новый слой SA.
    }
#endif
    const auto result=bdf_->advance(fields.state, fields.initial, fields.volumes, fields.flux, dt_or_fraction,
                                    settings, refresh, trial, confirm_prediction, timing.bdf_phases());
#ifdef AERO_HAS_TURBULENCE
    if(sa_active()) {
        if(sa_bdf2_) sa_apply(sa_step(),turbulence::Operation::CommitHistory);
        sa_previous_dt_=sa_bdf2_?dt:0.; sa_dt_=0.;
    }
#endif
    if (step_mode_ == LocalStepMode::Auto) {
        accepted_field_ = fields.state;
        previous_cfl_ = current_step_cfl_;
        accepted_fraction_ = dt_or_fraction;
        recovered_step_ = retry_dt_ > 0. || result.linear_iterations >= 100;
        retries_ = 0;
        retry_dt_ = 0.;
    }
    return result;
}

void PartitionOperator::update_owned(explicit_detail::Stage stage, std::vector<double>& start, double dt,
                                     std::vector<double>& state, bool save_original) {
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
#ifdef AERO_HAS_TURBULENCE
        if (sa_active() && !gpu_->has_stage_material()) {
            auto step=sa_step(); step.stage=stage; step.dt=dt; step.save_stage_start=save_original;
            sa_apply(step,turbulence::Operation::Explicit);
        }
#endif
        gpu_->update(stage, dt, save_original);
        return;
    }
#endif
    const Id* nodes = world_.size == 1 ? nullptr : owned.data();
    // На всех узлах одно число уравнений. Выбираем перед стадией тот же вариант
    // сборки строки, что и на GPU: длина суммы потоков известна при компиляции.
#ifdef AERO_HAS_VISCOSITY
    // На одном процессе обновлены все узлы. Вычисляем вязкость, пока новое состояние
    // ещё в кэше. Для MPI ждём обмена и готовим вязкость сразу для своих и соседних узлов.
    const auto* transport = !nodes && grid_.packed.gas && aes_ ? &aes_->view() : nullptr;
#endif
#ifdef AERO_HAS_TURBULENCE
    const auto* sa=sa_active() ? &aes_->view().sa : nullptr;
#endif
    const auto update = width_ == 5 ? partition_detail::update_node<5> :
                        width_ == 4 ? partition_detail::update_node<4> : partition_detail::update_node<3>;
    parallel_for(
        owned.size(),
        [&](std::size_t k) {
#ifdef AERO_HAS_TURBULENCE
            if (sa) sa->advance_explicit(nodes ? nodes[k] : k,stage,dt,save_original);
#endif
            if (!update(grid_, flux_.data(), volumes_, nodes, state.data(),
                        start.data(), k, stage, dt, save_original)) {
                throw std::domain_error("Invalid Euler stage state at global node " + std::to_string(ids_[owned[k]]));
            }
#ifdef AERO_HAS_VISCOSITY
            if (transport) transport->prepare_gas(k, state.data(), grid_.packed.gas);
#endif
        },
        256);
    gas_current_ = world_.size == 1;
}

PartitionGpu* PartitionOperator::device() const {
#ifdef AERO_WITH_KOKKOS_GPU
    return gpu_.get();
#else
    return nullptr;
#endif
}

}
