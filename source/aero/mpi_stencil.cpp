#include "mpi_stencil.h"
#include "../parallel/parallel_for.h"
#include <limits>
#include "../reconstruction/curved_stencil.h"
#include "mpi_periodic_detail.h"
#include <numeric>

namespace aero {
using namespace periodic_mpi;
namespace {
using Ids = std::vector<Id>;
Id index_of(const Ids& ids, Id id) {
    const auto it = std::lower_bound(ids.begin(), ids.end(), id);
    if (it == ids.end() || *it != id)
        throw MeshError("MPI stencil: missing node " + std::to_string(id));
    return static_cast<Id>(it - ids.begin());
}
Ref canonical(const MeshPartition& part, const PeriodicPartition* seam, Id n) {
    if (!seam)
        return vertex(part, n);
    const Id local = seam->node_dof[n], id = seam->ids[local];
    return {{id, id}, seam->owners[local]};
}

// Сведения о периодическом классе хранят владельцы всех его исходных вершин.
// Поэтому запрос второго слоя можно начать с номера вершины до склейки.
using Copies = std::map<Id, Ids>;
Copies periodic_copies(const MeshPartition& part, const PeriodicPartition& seam, const Processes& w) {
    std::map<Id, std::vector<Ref>> groups;
    exchange(
        w,
        [&](Messages& out) {
            for (Id n = 0; n < part.nodes.size(); ++n)
                if (part.node_owners[n] == w.rank) {
                    const Ref rep = canonical(part, &seam, n);
                    if (rep.key.first != part.nodes[n]) {
                        put(out[rep.rank], rep.key.first);
                        put(out[rep.rank], vertex(part, n));
                    }
                }
        },
        [&](Reader& r, int) {
            const Id id = r.id();
            groups[id].push_back(r.ref());
        });
    Copies result;
    exchange(
        w,
        [&](Messages& out) {
            for (auto& g : groups) {
                auto& members = g.second;
                members.push_back({{g.first, g.first}, w.rank});
                for (const auto& n : members) {
                    auto& s = out[n.rank];
                    put(s, n.key.first);
                    put(s, members.size());
                    for (const auto& m : members)
                        put(s, m.key.first);
                }
            }
        },
        [&](Reader& r, int) {
            auto& row = result[r.id()];
            for (Id k = r.id(); k; --k)
                row.push_back(r.id());
        });
    return result;
}

struct BoundarySupport {
    std::set<Id> paired, nodal, originals;
};
BoundarySupport boundary_support(const PartitionInput::NodeOwners& centers, const Copies& copies,
                                 const std::set<Id>& inlet_nodes, const Processes& w) {
    BoundarySupport result;
    Messages replies;
    exchange(
        w,
        [&](Messages& out) {
            replies.resize(w.size);
            for (const auto& n : centers) {
                const Id id = n.first;
                if (n.second != w.rank) {
                    put(out[n.second], id);
                    continue;
                }
                if (inlet_nodes.count(id))
                    result.nodal.insert(id);
                const auto it = copies.find(id);
                if (it != copies.end()) {
                    result.paired.insert(id);
                    result.originals.insert(it->second.begin(), it->second.end());
                }
            }
        },
        [&](Reader& r, int p) {
            const Id id = r.id();
            const auto it = copies.find(id);
            if (it == copies.end() && !inlet_nodes.count(id))
                return;
            auto& s = replies[p];
            put(s, id);
            put(s, inlet_nodes.count(id));
            put(s, it == copies.end() ? 0 : it->second.size());
            if (it != copies.end())
                for (Id copy : it->second)
                    put(s, copy);
        });
    exchange(
        w, [&](Messages& out) { out.swap(replies); },
        [&](Reader& r, int) {
            const Id id = r.id();
            if (r.id())
                result.nodal.insert(id);
            const Id count = r.id();
            if (count)
                result.paired.insert(id);
            for (Id k = 0; k < count; ++k)
                result.originals.insert(r.id());
        });
    return result;
}

// Сначала добавляем исходные копии, затем строим сдвинутые образы. Ячейка первого слоя,
// порождающая образ второго слоя, может находиться за разрезом. Её соседей берём из граничной полосы архива.
bool add_periodic_context(PartitionInput& input, const Copies& copies, const Processes& w, bool active,
                          bool first_ring_cells) {
    PartitionInput::NodeOwners boundary;
    std::set<Id> partial;
    w.local([&] {
        if (active)
            boundary = input.boundary_nodes(first_ring_cells ? &partial : nullptr);
    });
    const auto support = boundary_support(boundary, copies, {}, w);
    w.local([&] {
        input.add_stars(support.originals);
        input.add_stars(partial);
    });
    return !support.originals.empty();
}

std::vector<Ref> support_states(const MeshPartition& part, const PeriodicPartition* seam,
                                const MeshPartition& halo, const Processes& w) {
    std::vector<Ref> states;
    Messages replies;
    w.local([&] {
        states.reserve(halo.nodes.size());
        for (Id n = 0; n < halo.nodes.size(); ++n)
            states.push_back(vertex(halo, n));
    });
    if (!seam)
        return states;
    exchange(
        w,
        [&](Messages& out) {
            replies.resize(w.size);
            for (Id n = 0; n < states.size(); ++n) {
                const Id id = halo.nodes[n];
                const auto it = std::lower_bound(part.nodes.begin(), part.nodes.end(), id);
                if (it != part.nodes.end() && *it == id)
                    states[n] = canonical(part, seam, it - part.nodes.begin());
                else {
                    auto& s = out[states[n].rank];
                    put(s, n);
                    put(s, id);
                }
            }
        },
        [&](Reader& r, int p) {
            const Id slot = r.id(), id = r.id();
            put(replies[p], slot);
            put(replies[p], canonical(part, seam, index_of(part.nodes, id)));
        });
    exchange(
        w, [&](Messages& out) { out.swap(replies); },
        [&](Reader& r, int) {
            const Id n = r.id();
            states.at(n) = r.ref();
        });
    return states;
}

// Привязываем каждую общую грань к процессам, содержащим её призмы.
// Граница раздела MPI пропускает фронт нумерации так же, как внутренняя грань.
std::vector<int> partition_layer_depths(const MeshPartition& part, const MeshPartition& halo,
                                        const ReconstructionOptions& options, const Processes& w) {
    detail::PrismLayers graph(part.mesh, options.wall_labels);
    if (w.size == 1 && &part == &halo)
        return graph.local_depths(part.mesh.view());
    std::map<Id, std::vector<std::pair<int, Id>>> directory;
    exchange(
        w,
        [&](Messages& out) {
            const auto mesh = part.mesh.view();
            // Только грани на стыках требуют обмена: обе призмы внутренней грани
            // уже известны этому процессу, а у физической границы соседа нет.
            for (Id i = 0; i < mesh.boundary_face_count; ++i) {
                const Id f = mesh.boundary_faces[i];
                if (mesh.boundary_patch[i] < part.physical_patches || graph.opposite[f].empty())
                    continue;
                const Id global = part.faces.at(f);
                auto& packet = out[global % static_cast<Id>(w.size)];
                put(packet, global);
                put(packet, f);
            }
        },
        [&](Reader& r, int rank) {
            const Id global = r.id(), local = r.id();
            directory[global].emplace_back(rank, local);
        });
    std::map<Id, std::vector<std::pair<int, Id>>> peers;
    exchange(
        w,
        [&](Messages& out) {
            for (const auto& group : directory)
                for (const auto& a : group.second)
                    for (const auto& b : group.second)
                        if (a.first != b.first) {
                            auto& packet = out[a.first];
                            put(packet, a.second);
                            put(packet, static_cast<Id>(b.first));
                            put(packet, b.second);
                        }
        },
        [&](Reader& r, int) {
            const Id f = r.id(), rank = r.id(), remote = r.id();
            peers[f].emplace_back(static_cast<int>(rank), remote);
        });
    const auto local = graph.local_depths(part.mesh.view(), [&](Ids& frontier) {
        if (!w.sum(frontier.size()))
            return false;
        exchange(
            w,
            [&](Messages& out) {
                for (Id f : frontier) {
                    const auto found = peers.find(f);
                    if (found != peers.end())
                        for (const auto& p : found->second)
                            put(out[p.first], p.second);
                }
            },
            [&](Reader& r, int) { frontier.push_back(r.id()); });
        return true;
    });
    // На стыке стенок узел может входить в несколько столбцов призм. Его
    // владелец выбирает минимальный номер слоя среди всех копий, в том числе
    // если в его собственную часть не попало ни одной пристенной призмы.
    auto minimum = local;
    exchange(
        w,
        [&](Messages& out) {
            for (Id n = 0; n < local.size(); ++n) {
                const int owner = part.node_owners[n];
                if (local[n] < 0 || owner == w.rank)
                    continue;
                put(out[owner], part.nodes[n]);
                put(out[owner], static_cast<Id>(local[n]));
            }
        },
        [&](Reader& r, int) {
            const Id n = index_of(part.nodes, r.id());
            const int level = static_cast<int>(r.id());
            if (minimum[n] < 0 || level < minimum[n])
                minimum[n] = level;
        });
    // Для вспомогательных узлов шаблона запрашиваем окончательный номер у владельца.
    std::vector<int> depth(halo.nodes.size(), -1);
    Messages replies(w.size);
    exchange(
        w,
        [&](Messages& out) {
            for (Id n = 0; n < halo.nodes.size(); ++n) {
                auto& packet = out[halo.node_owners[n]];
                put(packet, n);
                put(packet, halo.nodes[n]);
            }
        },
        [&](Reader& r, int p) {
            const Id slot = r.id(), n = index_of(part.nodes, r.id());
            put(replies[p], slot);
            put(replies[p], static_cast<Id>(minimum[n] + 1));
        });
    exchange(
        w, [&](Messages& out) { out.swap(replies); },
        [&](Reader& r, int) {
            const Id slot = r.id();
            depth[slot] = static_cast<int>(r.id()) - 1;
        });
    return depth;
}

Id find_edge(MeshView m, Id a, Id b) {
    const auto first = std::lower_bound(m.edge_begin, m.edge_begin + m.edge_count, a) - m.edge_begin;
    const auto last = std::upper_bound(m.edge_begin + first, m.edge_begin + m.edge_count, a) - m.edge_begin;
    const auto it = std::lower_bound(m.edge_end + first, m.edge_end + last, b);
    if (it == m.edge_end + last || *it != b)
        throw MeshError("MPI stencil: missing representative edge");
    return static_cast<Id>(it - m.edge_end);
}
}

ReconstructionStencilView PartitionStencil::view(MeshIndex state_nodes) const {
    return {state_nodes,           row_owner.size() / 2,
            linear.offsets.data(), linear.nodes.data(),
            row_owner.data(),      linear.weights.data(),
            slopes.offsets.data(), slopes.nodes.data(),
            slopes.weights.data(), weno_order.empty() ? nullptr : weno_order.data(),
            epsilon_scale.data(), mp5,nullptr,false,nullptr,false};
}

namespace {
// Добавляет к ядру разбиения ячейки, нужные лучам реконструкции. В распределённом расчёте
// эти данные требуются лишь на этапе подготовки. Владельцы удалённых узлов позже пришлют значения поля.
MeshPartition build_stencil_support(const MeshPartition& part, const PeriodicPartition* seam,
                                    PartitionInput& input, const StencilRequests& requests, bool fifth,
                                    const Processes& w, std::set<Id>& nodal) {
    const auto& ids = seam ? seam->ids : part.nodes;
    const auto& owners = seam ? seam->owners : part.node_owners;
    const Copies copies = seam ? periodic_copies(part, *seam, w) : Copies();
    PartitionInput::NodeOwners centers;
    w.local([&] {
        std::set<Id> ends;
        for (const auto& e : requests.edges) {
            ends.insert(e.begin);
            ends.insert(e.end);
        }
        for (Id i : ends)
            centers.emplace_back(ids[i], owners[i]);
    });
    auto boundary = boundary_support(centers, copies, requests.inlet_nodes, w);
    PartitionInput::NodeOwners near_vertices;
    w.local([&] {
        for (const auto& e : requests.edges)
            if (boundary.paired.count(ids[e.begin]) || boundary.paired.count(ids[e.end])) {
                boundary.originals.insert(e.original.first);
                boundary.originals.insert(e.original.second);
            }
        input.prepare_stencil(boundary.originals, fifth, seam && fifth ? &near_vertices : nullptr);
    });
    std::set<Id> far_copies;
    if (seam && fifth)
        far_copies = boundary_support(near_vertices, copies, {}, w).originals;
    w.local([&] { input.add_stars(far_copies); });
    // В одном измерении единственная вершина слоя уже задаёт всю грань.
    if (part.mesh.view().dimension > 1) {
        const bool crosses_seam = seam && add_periodic_context(input, copies, w, true, false);
        w.local([&] { input.complete_layer_faces(); });
        if (seam) {
            if (fifth)
                add_periodic_context(input, copies, w, crosses_seam, true);
            add_periodic_context(input, copies, w, crosses_seam, false);
        }
    }
    nodal.swap(boundary.nodal);
    MeshPartition support;
    w.local([&] { support = input.stencil(); });
    return support;
}

// Каждое ребро выбирает исходные строки для своих двух сторон. По их длинам
// заранее выделяем непересекающиеся участки в общих массивах узлов и весов.
class RowTransfer {
    using Indices = std::vector<ReconstructionIndex>;
    PartitionStencil::Rows& dst_;
    const Indices &offsets_, &nodes_;
    const std::vector<double>& weights_;
    Indices source_; // Номер исходной строки для каждой строки приёмника.
public:
    RowTransfer(PartitionStencil::Rows& dst, const Indices& offsets, const Indices& nodes,
                const std::vector<double>& weights)
        : dst_(dst), offsets_(offsets), nodes_(nodes), weights_(weights),
          source_(dst.offsets.empty() ? 0 : dst.offsets.size() - 1) {}
    void select(Id row, Id dst) {
        source_[dst] = row;
        dst_.offsets[dst + 1] = offsets_[row + 1] - offsets_[row];
    }
    void copy(const Ids& field) {
        if (source_.empty()) return;
        auto& offsets = dst_.offsets;
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        dst_.nodes.resize(offsets.back());
        dst_.weights.resize(offsets.back());
        parallel_for(source_.size(), [&](std::size_t row) {
            const auto out = offsets[row], count = offsets[row + 1] - out;
            if (!count) return;
            const auto in = offsets_[source_[row]];
            std::copy_n(weights_.data() + in, count, dst_.weights.data() + out);
            // На целой сетке номера вершин уже совпадают с номерами состояний.
            if (field.empty())
                std::copy_n(nodes_.data() + in, count, dst_.nodes.data() + out);
            else
                for (Id k = 0; k < count; ++k)
                    dst_.nodes[out + k] = field[nodes_[in + k]];
        });
    }
};

// Каждый используемый узел получает один номер на подобласть. Удалённые
// узлы, не вошедшие ни в один шаблон, не увеличивают объём обмена MPI.
void assign_import_slots(PartitionStencil& result, const std::vector<Ref>& states, const Ids& ids, int rank) {
    const Id unused = std::numeric_limits<Id>::max();
    Ids slots(states.size(), unused);
    for (auto* rows : {&result.linear, &result.slopes})
        for (const auto state : rows->nodes)
            slots[state] = 0;
    for (Id i = 0; i < states.size(); ++i) {
        if (slots[i] == unused) continue;
        const auto& node = states[i];
        const auto it = std::lower_bound(ids.begin(), ids.end(), node.key.first);
        if (it != ids.end() && *it == node.key.first) {
            slots[i] = static_cast<Id>(it - ids.begin());
        } else {
            if (node.rank == rank) throw MeshError("MPI stencil: missing owned state");
            slots[i] = ids.size() + result.extra_nodes.size();
            result.extra_nodes.emplace_back(node.key.first, node.rank);
        }
    }
    for (auto* rows : {&result.linear, &result.slopes})
        parallel_for(rows->nodes.size(), [&](std::size_t k) {
            rows->nodes[k] = slots[rows->nodes[k]];
        });
}

// Линейной схеме нужны две строки на ребро. В WENO каждая сторона может иметь четыре строки наклонов.
// Передаёт солверу готовую реконструкцию целой сетки без перенумерации узлов.
void take_native_stencil(PartitionStencil& result, ReconstructionStencilData src,
                         const StencilRequests& requests, const Ids& ids, Id surfaces, bool weno) {
    // На целой сетке нумерация рёбер и узлов уже совпадает с нумерацией солвера.
    // Передаём готовые массивы без второй копии всех строк интерполяции.
    result.incomplete = std::count(src.complete.begin(), src.complete.end(), 0);
    result.linear = {std::move(src.offsets), std::move(src.nodes), std::move(src.weights)};
    if (weno) result.slopes = {std::move(src.slope_offsets), std::move(src.slope_nodes), std::move(src.slope_weights)};
    result.row_owner.swap(src.row_owner);
    result.weno_order.swap(src.weno_order);
    result.epsilon_scale.swap(src.weno_epsilon_scale);
    result.nodal_traces.resize(surfaces);
    std::vector<unsigned char> inlet(ids.size());
    // Обходим только узлы открытой границы. В таблице целой сетки глобальный
    // номер обычно совпадает с позицией; для иной нумерации ищем его в ids.
    for (Id n : requests.inlet_nodes)
        inlet[n < ids.size() && ids[n] == n ? n : index_of(ids, n)] = 1;
    parallel_for(requests.edges.size(), [&](Id k) {
        const auto& e = requests.edges[k];
        if (!inlet[e.begin] && !inlet[e.end]) return;
        result.nodal_traces[e.surface] = 1;
        if (weno) {
            result.weno_order[2*e.surface] = result.weno_order[2*e.surface+1] = 0;
        }
    });
}

void allocate_stencil_rows(PartitionStencil& result, Id surfaces, bool weno) {
    const Id rows_per_edge = weno ? 8 : 2;
    if (surfaces > (result.linear.offsets.max_size() - 1) / rows_per_edge)
        throw MeshError("MPI stencil: too many surfaces");
    result.linear.offsets.resize(2 * surfaces + 1);
    if (weno) {
        result.slopes.offsets.resize(8 * surfaces + 1);
        result.weno_order.resize(2 * surfaces);
        result.epsilon_scale.resize(2 * surfaces);
    }
    result.row_owner.resize(2 * surfaces);
    result.nodal_traces.resize(surfaces);
}
}

PartitionStencil prepare_partition_stencil(const MeshPartition& part, const PeriodicPartition* seam,
                                           const std::vector<PeriodicPair>& pairs, PartitionInput* input,
                                           const StencilRequests& requests, Id surfaces,
                                           const ReconstructionOptions& options, const Processes& w) {
    const auto& computed = requests.edges;
    const auto& ids = seam ? seam->ids : part.nodes;
    const auto scheme = reconstruction_scheme(options.scheme);
    const bool fifth = scheme.layers == 2;
    const bool weno = uses_slope_rows(scheme.family);
    MeshPartition support;
    std::set<Id> nodal;
    if (input) {
        support = build_stencil_support(part, seam, *input, requests, fifth, w, nodal);
    } else
        nodal = requests.inlet_nodes;
    const MeshPartition& halo = input ? support : part;
    const auto native_states = input ? support_states(part, seam, halo, w) : std::vector<Ref>();
    const bool pl = options.geometry == StencilGeometry::Pl || options.geometry == StencilGeometry::Auto;
    std::vector<int> depth;
    if (pl && w.sum(options.wall_labels.size()))
        depth = partition_layer_depths(part, halo, options, w);
    PartitionStencil result;
    result.mp5 = {options.scheme == ReconstructionScheme::Mp5, options.scheme == ReconstructionScheme::MPWENO5,
                  options.mpweno_ad_corr, options.mp5_alpha};
    w.local([&] {
        if (computed.empty()) {
            allocate_stencil_rows(result, surfaces, weno);
            return;
        }
        const auto& originals = halo.nodes;
        Ids field;
        std::map<Ref, Id> numbering;
        std::vector<Ref> states;
        for (Ref n : native_states) {
            const auto it = numbering.emplace(n, states.size());
            if (it.second)
                states.push_back(n);
            field.push_back(it.first->second);
        }
        Mesh lifted;
        if (seam) {
            const double tol = periodic_detail::match_tolerance(pairs);
            Ids image_field;
            const auto& dofs = input ? field : seam->node_dof;
            lifted = periodic_stencil_cover(halo.mesh, dofs, input ? states.size() : ids.size(), tol,
                                            scheme.layers, image_field);
            field.swap(image_field);
        }
        const Mesh& geometry = seam ? lifted : halo.mesh;
        auto local_options = options;
        local_options.solver_rows_only = true;
        if (!depth.empty() && seam) {
            // Поле периодического образа ссылается на представителя. Номер
            // пристенного слоя переносим и на все его геометрические копии.
            std::vector<int> compact(input ? states.size() : ids.size(), -1);
            for (Id n = 0; n < depth.size(); ++n)
                if (depth[n] >= 0) {
                    auto& level = compact[input ? numbering.at(native_states[n]) : seam->node_dof[n]];
                    if (level < 0 || depth[n] < level)
                        level = depth[n];
                }
            for (Id slot : field)
                local_options.node_layers.push_back(compact[slot]);
        } else
            local_options.node_layers.swap(depth);
        auto prepared = prepare_reconstruction(geometry, local_options);
        if (w.size == 1 && !input && !seam) {
            take_native_stencil(result, std::move(prepared).release_data(), requests, ids, surfaces, weno);
            return;
        }
        allocate_stencil_rows(result, surfaces, weno);
        const auto& src = prepared.data();
        // Строки геометрии используют номера вершин вспомогательной сетки. Солверу нужны смещения в массиве состояний.
        RowTransfer linear(result.linear, src.offsets, src.nodes, src.weights);
        RowTransfer slopes(result.slopes, src.slope_offsets, src.slope_nodes, src.slope_weights);
        std::vector<unsigned char> missing(computed.size(), 0);
        parallel_for(computed.size(), [&](std::size_t k) {
            const auto& e = computed[k];
            // Для целой сетки без склейки номера исходных рёбер уже известны.
            const Id edge = !input && !seam ? e.surface : find_edge(geometry.view(),
                    index_of(originals, e.original.first), index_of(originals, e.original.second));
            const bool first = nodal.count(ids[e.begin]) || nodal.count(ids[e.end]);
            result.nodal_traces[e.surface] = first;
            for (Id side = 0; side < 2; ++side) {
                const Id row = 2 * edge + side, dst = 2 * e.surface + side;
                const Id endpoint = side ? e.end : e.begin;
                const Id vertex = src.row_owner[row];
                const Id owner = (input || seam) ? field[vertex] : vertex;
                if (input ? states[owner].key.first != ids[endpoint] : owner != endpoint)
                    throw MeshError("MPI stencil: endpoint mismatch");
                result.row_owner[dst] = endpoint;
                if (!src.complete[row])
                    ++missing[k];
                if (first)
                    continue;
                if (weno && src.weno_order[row]) {
                    result.weno_order[dst] = src.weno_order[row];
                    result.epsilon_scale[dst] = src.weno_epsilon_scale[row];
                    for (Id j = 0; j < 4; ++j)
                        slopes.select(4 * row + j, 4 * dst + j);
                } else
                    linear.select(row, dst);
            }
        });
        result.incomplete = std::accumulate(missing.begin(), missing.end(), Id(0));
        linear.copy(field);
        slopes.copy(field);
        if (input) assign_import_slots(result, states, ids, w.rank);
    });
    return result;
}
}
