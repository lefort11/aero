#include "partition_operator.h"
#include "partition_kernels.h"
#include "../parallel/parallel_for.h"
#include "../residual/flux_cfl.h"
#include <cmath>
#include <limits>

namespace aero {
using Id = MeshIndex;
using residual_detail::Surface;

PartitionOperator::PartitionOperator(const MeshPartition& part, CvView cv, const PeriodicPartition* periodic,
                                     const BoundaryPatches& patches, ResidualOptions options,
                                     const ResidualFrame& frame, const Processes& world,
                                     StencilRequests* requests, bool cuda)
    : nodes_(world.size, cv.dimension + 2), edges_(world.size, cv.dimension + 3), width_(cv.dimension + 2),
      volumes_(periodic ? periodic->volumes.data() : cv.volumes), ids_(periodic ? periodic->ids : part.nodes),
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
    add_boundary_surfaces(cv, periodic, part.physical_patches, patches, frame, requests);

    // Ядра ожидают сначала все рёбра в порядке сетки, затем части физических границ.
    residual_detail::build_incidence(ids_.size(), edge_count, surfaces_, offsets_, incidence_);
    grid_ = {
        mesh.dimension,    ids_.size(),    edge_count, surfaces_.size(), surfaces_.data(), offsets_.data(),
        incidence_.data(), patches.view(), {},         options, {}};
    if (!cuda) {
        flux_.resize(surfaces_.size());
        normals_.resize(edge_count);
        parallel_for(edge_count, [&](Id i) {
            if (euler_unit_direction(mesh.dimension, surfaces_[i].area, normals_[i].axis) != EulerStatus::Success)
                throw std::domain_error("Invalid edge normal");
        });
        grid_.packed.normals = normals_.data();
    }
    prepare_nodal_gas();
}

void PartitionOperator::add_edge_surfaces(MeshView mesh, CvView cv, const PeriodicPartition* periodic,
                                          const ResidualFrame& frame, StencilRequests* requests) {
    const Id count = periodic ? periodic->edges.size() : mesh.edge_count;
    for (Id edge = 0; edge < count; ++edge) {
        const Id begin = periodic ? periodic->edges[edge].begin : mesh.edge_begin[edge];
        const Id end = periodic ? periodic->edges[edge].end : mesh.edge_end[edge];
        const int sender = owners_[begin];
        const int receiver = owners_[end];
        if (sender != world_.rank && receiver != world_.rank) {
            continue;
        }

        const Id index = surfaces_.size();
        Surface surface{};
        surface.begin = begin;
        surface.end = end;
        surface.use_nodal_traces = true;
        const auto& area = periodic ? periodic->edges[edge].area : cv.edge_areas[edge];
        residual_detail::project_surface(surface, area, mesh.dimension, frame);
        surfaces_.push_back(surface);

        // Поток вычисляется ровно один раз на владельце начала ребра.
        if (sender != world_.rank) {
            edges_.imports[sender].push_back(index);
            continue;
        }
        compute_.push_back(index);
        if (requests) {
            const auto original =
                periodic ? periodic->edges[edge].original : std::make_pair(ids_[begin], ids_[end]);
            requests->edges.push_back({index, begin, end, original});
        }
        if (receiver != sender) {
            edges_.exports[receiver].push_back(index);
        }
    }
}

void PartitionOperator::add_boundary_surfaces(CvView cv, const PeriodicPartition* periodic,
                                              Id physical_patches, const BoundaryPatches& patches,
                                              const ResidualFrame& frame, StencilRequests* requests) {
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
using residual_detail::WenoEdgeTile;

// Упаковываем общий центральный и два ближних наклона. Пятый порядок добавляет два дальних.
// Для неполного шаблона солвер читает исходные разреженные строки.
template <unsigned scheme_order>
bool pack_weno_edge(const PartitionStencil& stencil, MeshIndex edge, WenoEdgeTile<scheme_order>& tile, unsigned lane) {
    if (stencil.weno_order[2 * edge] != scheme_order || stencil.weno_order[2 * edge + 1] != scheme_order)
        return false;
    const auto& rows = stencil.slopes;
    // В исходной записи по четыре наклона с каждой стороны. Ближний правый наклон при WENO5
    // доступен как противоположный у левой стороны (2), при WENO3 — как собственный у правой (4 + 1).
    const unsigned row_number[] = {0, 1, scheme_order == 5 ? 2u : 5u, 3, 7};
    unsigned term = 0, weight = 0;
    for (unsigned r = 0; r < tile.rows_per_edge; ++r) {
        const auto row = 8 * edge + row_number[r];
        const auto first = rows.offsets[row], end = rows.offsets[row + 1];
        if (end - first != residual_detail::slope_terms(r))
            return false;
        for (auto i = first; i < end; ++i, ++term) {
                // Сохраняем индекс прямо у слагаемого, даже если узел встречался в соседней строке.
                // Так при каждом расчёте потока читается один индекс вместо двух зависимых обращений.
            tile.nodes[term][lane] = static_cast<unsigned>(rows.nodes[i]);
            if (i != first)
                tile.weights[weight++][lane] = rows.weights[i];
        }
    }
    return true;
}
}

void PartitionOperator::prepare_nodal_gas() {
    grid_.packed.state_nodes = state_nodes();
    if (!flux_.empty()) {
        nodal_gas_.resize(state_nodes());
        grid_.packed.gas = nodal_gas_.data();
    }
}

template <unsigned scheme_order>
void PartitionOperator::prepare_weno_tiles(std::vector<WenoEdgeTile<scheme_order>>& tiles) {
    auto& g = grid_;
    if (g.dim != 2 || g.options.variables != ResidualVariables::Characteristic ||
        g.options.solver != ResidualSolver::Roe || stencil_.weno_order.empty() ||
        state_nodes() > std::numeric_limits<unsigned>::max())
        return;
    // На CPU потоки площадок хранятся в flux_. На GPU для малой сетки один общий запуск
    // быстрее двух запусков для полных и остальных шаблонов. Веса реконструкции те же.
    const MeshIndex small_gpu_edge_limit = 8192;
    if (flux_.empty() && g.edges <= small_gpu_edge_limit)
        return;
    const auto count = (g.edges + 31) / 32;
    tiles.resize(count);
    parallel_for(count, [&](MeshIndex t) {
        auto& tile = tiles[t];
        for (unsigned lane = 0; lane < 32; ++lane) {
            const auto edge = 32 * t + lane;
            tile.complete[lane] = edge < g.edges && !g.surfaces[edge].use_nodal_traces &&
                                  pack_weno_edge(stencil_, edge, tile, lane);
        }
    });
    for (auto f : compute_)
        if (f >= g.edges || !tiles[f / 32].complete[f % 32])
            other_faces_.push_back(f);
    if (other_faces_.size() == compute_.size())
        tiles.clear();

    if (tiles.empty())
        return;
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
        for (const auto& edge : requests.edges) {
            surfaces_[edge.surface].use_nodal_traces = stencil_.nodal_traces[edge.surface] != 0;
        }
        for (Id j = 0; j < stencil_.extra_nodes.size(); ++j) {
            nodes_.imports[stencil_.extra_nodes[j].second].push_back(ids_.size() + j);
        }
        grid_.stencil = stencil_.view(state_nodes());
        prepare_nodal_gas();
        if (options.scheme == ReconstructionScheme::Weno5)
            prepare_weno_tiles(weno5_tiles_);
        else if (options.scheme == ReconstructionScheme::Weno3)
            prepare_weno_tiles(weno3_tiles_);
    });
    return world_.sum(stencil_.incomplete);
}

void PartitionOperator::connect(bool cuda) {
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
    });
}

#ifdef AERO_WITH_CUDA
void PartitionOperator::layout_gpu(bool compact_buffer) {
    gpu_->layout(grid_, stencil_, compute_, owned, volumes_, state_nodes(), compact_buffer, world_.size == 1);
    nodes_.prepare_gpu(*gpu_);
    edges_.prepare_gpu(*gpu_);
}

void PartitionOperator::reserve_gpu(bool compact_buffer, GpuMemoryFit& fit) {
    gpu_.reset(new PartitionGpu);
    layout_gpu(compact_buffer);
    fit.failed = !gpu_->reserve(fit);
    // На тесной карте расчёт может поместиться с исходными разреженными строками, без дополнительных буферов.
    if (fit.failed && grid_.packed.state_nodes) {
        grid_.packed = {};
        gpu_.reset(new PartitionGpu);
        layout_gpu(compact_buffer);
        fit.failed = !gpu_->reserve(fit);
    }
}

void PartitionOperator::upload_state(const std::vector<double>& state, const std::vector<double>& initial) {
    gpu_->upload_state(state, initial);
}
#endif

void PartitionOperator::copy_owned(const std::vector<double>& state, std::vector<double>& compact) const {
#ifdef AERO_WITH_CUDA
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
#ifdef AERO_WITH_CUDA
    if (gpu_) {
        gpu_->download(state);
    }
#else
    (void)state;
#endif
}

void PartitionOperator::halo(std::vector<double>& u, double* seconds) {
    nodes_.exchange_states(world_, width_, u.data(), device(), seconds);
}

void PartitionOperator::halo_host(std::vector<double>& u) {
    nodes_.exchange_states(world_, width_, u.data());
}

void PartitionOperator::assemble_fluxes(const std::vector<double>& u, const std::vector<double>& initial,
                                        SolverTimings& timing) {
    world_.local([&] {
        WallTimer timer(timing.counter(SolverTimings::Flux));
#ifdef AERO_WITH_CUDA
        if (gpu_) {
            gpu_->fluxes();
            return;
        }
#endif
        // Каждый процесс вычисляет назначенные ему поверхности, включая физические площадки своих узлов.
        const Id* faces = world_.size == 1 ? nullptr : compute_.data();
        residual_detail::cpu_face_fluxes(
            {grid_, faces, compute_.size(), u.data(), initial.data(), flux_.data(), nullptr, nullptr},
            ids_.data());
    });
    if (world_.size == 1) {
        return;
    }
    // На стыке сосед получает тот же интегральный поток. Знак задаст его узловой список при сборке.
    edges_.exchange_fluxes(world_, width_, flux_.data(), device(),
                           timing.counter(SolverTimings::FluxExchange));
}

double PartitionOperator::cfl_step(double cfl) {
    // Пустая подобласть не ограничивает шаг. Минимум по процессам выбирает конечное значение.
    double local = std::numeric_limits<double>::infinity();
    world_.local([&] {
#ifdef AERO_WITH_CUDA
        if (gpu_) {
            local = gpu_->cfl(cfl);
            return;
        }
#endif
        const Id* nodes = world_.size == 1 ? nullptr : owned.data();
        local = residual_detail::cpu_flux_cfl({grid_, flux_.data(), volumes_, nodes, owned.size()}, cfl,
                                              ids_.data());
    });
    const double dt = world_.minimum(local);
    if (!std::isfinite(dt)) {
        throw std::domain_error("No finite global CFL step");
    }
    return dt;
}

void PartitionOperator::update_owned(explicit_detail::Stage stage, std::vector<double>& start, double dt,
                                     std::vector<double>& state, bool save_original) const {
#ifdef AERO_WITH_CUDA
    if (gpu_) {
        gpu_->update(stage, dt, save_original);
        return;
    }
#endif
    const Id* nodes = world_.size == 1 ? nullptr : owned.data();
    parallel_for(
        owned.size(),
        [&](std::size_t k) {
            if (!partition_detail::update_node(grid_, flux_.data(), volumes_, nodes, state.data(),
                                               start.data(), k, stage, dt, save_original)) {
                throw std::domain_error("Nonfinite stage at global node " + std::to_string(ids_[owned[k]]));
            }
        },
        256);
}

PartitionGpu* PartitionOperator::device() const {
#ifdef AERO_WITH_CUDA
    return gpu_.get();
#else
    return nullptr;
#endif
}

}
