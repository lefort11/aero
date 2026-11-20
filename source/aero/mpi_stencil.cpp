#include "mpi_stencil.h"
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
            epsilon_scale.data(), mp5};
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

// Линейной схеме нужны две строки на ребро. В WENO каждая сторона может иметь четыре строки наклонов.
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
    PartitionStencil result;
    result.mp5 = {options.scheme == ReconstructionScheme::Mp5, options.scheme == ReconstructionScheme::MPWENO5,
                  options.mpweno_ad_corr, options.mp5_alpha};
    w.local([&] {
        allocate_stencil_rows(result, surfaces, weno);
        if (computed.empty())
            return;
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
        const auto prepared = prepare_reconstruction(geometry, options);
        const auto& src = prepared.data();
        // Строки геометрии используют номера вершин вспомогательной сетки. Солверу нужны смещения в массиве состояний.
        std::map<Ref, Id> extra;
        const auto slot = [&](Id vertex) -> Id {
            // У целой сетки нумерация уже окончательная, в том числе для периодических образов.
            if (!input)
                return seam ? field[vertex] : vertex;
            const Ref n = states[field[vertex]];
            const auto it = std::lower_bound(ids.begin(), ids.end(), n.key.first);
            if (it != ids.end() && *it == n.key.first)
                return static_cast<Id>(it - ids.begin());
            if (n.rank == w.rank)
                throw MeshError("MPI stencil: missing owned state");
            const auto add = extra.emplace(n, ids.size() + extra.size());
            if (add.second)
                result.extra_nodes.emplace_back(n.key.first, n.rank);
            return add.first->second;
        };
        using Indices = std::vector<ReconstructionIndex>;
        const auto copy_row = [&](Id row, Id dst, const Indices& offsets, const Indices& nodes,
                                  const std::vector<double>& weights, PartitionStencil::Rows& out) {
            for (Id k = offsets[row]; k < offsets[row + 1]; ++k) {
                out.nodes.push_back(slot(nodes[k]));
                out.weights.push_back(weights[k]);
            }
            // Пока храним длину строки; префиксная сумма после обхода превратит её в смещение CSR.
            out.offsets[dst + 1] = offsets[row + 1] - offsets[row];
        };
        for (const auto& e : computed) {
            const Id i = index_of(originals, e.original.first), j = index_of(originals, e.original.second);
            const Id edge = find_edge(geometry.view(), i, j);
            const bool first = nodal.count(ids[e.begin]) || nodal.count(ids[e.end]);
            result.nodal_traces[e.surface] = first;
            for (Id side = 0; side < 2; ++side) {
                const Id row = 2 * edge + side, dst = 2 * e.surface + side;
                const Id endpoint = side ? e.end : e.begin;
                if (slot(src.row_owner[row]) != endpoint)
                    throw MeshError("MPI stencil: endpoint mismatch");
                result.row_owner[dst] = endpoint;
                if (!src.complete[row])
                    ++result.incomplete;
                if (first)
                    continue;
                if (weno && src.weno_order[row]) {
                    result.weno_order[dst] = src.weno_order[row];
                    result.epsilon_scale[dst] = src.weno_epsilon_scale[row];
                    for (Id k = 0; k < 4; ++k)
                        copy_row(4 * row + k, 4 * dst + k, src.slope_offsets, src.slope_nodes,
                                 src.slope_weights, result.slopes);
                } else
                    copy_row(row, dst, src.offsets, src.nodes, src.weights, result.linear);
            }
        }
        for (auto* rows : {&result.linear, &result.slopes})
            std::partial_sum(rows->offsets.begin(), rows->offsets.end(), rows->offsets.begin());
    });
    return result;
}
}
