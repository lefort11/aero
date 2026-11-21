#include "mpi_periodic_detail.h"

namespace aero {
using namespace periodic_mpi;
namespace {
struct EdgeData {
    Key key;
    Ref a, b;
    Coordinate area;
};
struct Member {
    EdgeData edge;
    Coordinate direction;
};
struct Volume {
    Ref node;
    Id original;
    double value;
};
struct Piece {
    Key order;
    Ref node;
    Id patch;
    Coordinate area, position;
};

void put_edge(std::string& s, const EdgeData& e) {
    put(s, e.key);
    put(s, e.a);
    put(s, e.b);
    put(s, e.area);
}
EdgeData read_edge(Reader& r) {
    return {r.key(), r.ref(), r.ref(), r.point()};
}

std::vector<Ref> node_representatives(const MeshPartition& part, const Labels& labels, const Processes& w) {
    std::vector<Ref> result;
    Messages replies;
    exchange(
        w,
        [&](Messages& out) {
            result.resize(part.nodes.size());
            replies.resize(w.size);
            for (Id i = 0; i < part.nodes.size(); ++i) {
                const Ref n = vertex(part, i);
                if (n.rank == w.rank)
                    result[i] = representative(labels, n);
                else {
                    put(out[n.rank], i);
                    put(out[n.rank], n.key);
                }
            }
        },
        [&](Reader& r, int peer) {
            const Id i = r.id();
            const Key key = r.key();
            put(replies[peer], i);
            put(replies[peer], representative(labels, {key, w.rank}));
        });
    exchange(
        w, [&](Messages& out) { out.swap(replies); },
        [&](Reader& r, int) {
            const Id i = r.id();
            result.at(i) = r.ref();
        });
    return result;
}

std::vector<EdgeData> collect_edges(const MeshPartition& part, CvView cv, const Labels& labels,
                                    const std::vector<Ref>& nodes, const Processes& w) {
    const auto m = part.mesh.view();
    std::map<Key, std::vector<Member>> groups;
    std::vector<EdgeData> incident;
    exchange(
        w,
        [&](Messages& out) {
            for (Id e = 0; e < m.edge_count; ++e) {
                const Id a = m.edge_begin[e], b = m.edge_end[e];
                if (part.node_owners[a] != w.rank)
                    continue;
                const Key key = {part.nodes[a], part.nodes[b]};
                const auto it = labels.find(key);
                if (it == labels.end())
                    continue;
                auto& s = out[it->second.rank];
                put(s, it->second.key);
                put_edge(s, {key, nodes[a], nodes[b], point(cv.edge_areas[e])});
                put(s, (m.coordinate(b) - m.coordinate(a)));
            }
        },
        [&](Reader& r, int) {
            const Key root = r.key();
            groups[root].push_back({read_edge(r), r.point()});
        });
    exchange(
        w,
        [&](Messages& out) {
            const auto deliver = [&](const EdgeData& e) {
                if (e.a.rank == w.rank || e.b.rank == w.rank)
                    incident.push_back(e);
                if (e.a.rank != w.rank)
                    put_edge(out[e.a.rank], e);
                if (e.b.rank != w.rank && e.a.rank != e.b.rank)
                    put_edge(out[e.b.rank], e);
            };
            for (Id e = 0; e < m.edge_count; ++e) {
                const Id a = m.edge_begin[e], b = m.edge_end[e];
                const Key key = {part.nodes[a], part.nodes[b]};
                if (part.node_owners[a] == w.rank && !labels.count(key))
                    deliver({key, nodes[a], nodes[b], point(cv.edge_areas[e])});
            }
            for (auto& kv : groups) {
                auto& members = kv.second;
                std::sort(members.begin(), members.end(),
                          [](const Member& a, const Member& b) { return a.edge.key < b.edge.key; });
                EdgeData e = members.front().edge;
                if (e.key != kv.first)
                    throw MeshError("periodic: missing representative edge");
                e.area = {{0, 0, 0}};
                const Coordinate dir = members.front().direction;
                for (const auto& member : members) {
                    // Если периодическое ребро замкнулось на себя, номеров концов недостаточно для определения направления.
                    const bool forward = coordinate_length(member.direction - dir) <= coordinate_length(member.direction + dir);
                    e.area = forward ? (e.area + member.edge.area) : (e.area - member.edge.area);
                }
                deliver(e);
            }
        },
        [&](Reader& r, int) { incident.push_back(read_edge(r)); });
    return incident;
}
// Каждый исходный узел имеет одного владельца. Только он посылает вклад в объём представителя.
std::vector<Volume> collect_volumes(const MeshPartition& part, CvView cv, const std::vector<Ref>& nodes,
                                    const Processes& w) {
    std::vector<Volume> volumes;
    exchange(
        w,
        [&](Messages& out) {
            for (Id i = 0; i < nodes.size(); ++i)
                if (part.node_owners[i] == w.rank) {
                    const Volume v = {nodes[i], part.nodes[i], cv.volumes[i]};
                    if (v.node.rank == w.rank)
                        volumes.push_back(v);
                    else {
                        auto& s = out[v.node.rank];
                        put(s, v.node);
                        put(s, v.original);
                        put_real(s, v.value);
                    }
                }
        },
        [&](Reader& r, int) { volumes.push_back({r.ref(), r.id(), r.real()}); });
    return volumes;
}

// Швы уходят во внутренние потоки. Физические площадки пересылаем владельцам оставшихся степеней свободы.
std::vector<Piece> collect_physical_boundary(const MeshPartition& part, CvView cv,
                                             const std::vector<Ref>& nodes,
                                             const std::vector<PeriodicPair>& pairs, const Processes& w) {
    std::vector<Piece> boundary;
    exchange(
        w,
        [&](Messages& out) {
            std::set<std::string> paired;
            for (const auto& p : pairs) {
                paired.insert(p.patch);
                paired.insert(p.pair);
            }
            const auto m = part.mesh.view();
            Id k = 0;
            for (Id b = 0; b < m.boundary_face_count; ++b) {
                const Id patch = m.boundary_patch[b];
                const auto corners = m.face_nodes(m.boundary_faces[b]);
                if (patch < part.physical_patches && !paired.count(part.mesh.data().patch_names[patch]))
                    for (Id j = 0; j < corners.size; ++j) {
                        const Id n = corners[j];
                        if (part.node_owners[n] != w.rank)
                            continue;
                        const Piece piece = {
                            {part.boundary_order[b], j}, nodes[n], patch, point(cv.boundary[k + j].area), m.coordinate(n)};
                        if (piece.node.rank == w.rank)
                            boundary.push_back(piece);
                        else {
                            auto& s = out[piece.node.rank];
                            put(s, piece.order);
                            put(s, piece.node);
                            put(s, patch);
                            put(s, piece.area);
                            put(s, piece.position);
                        }
                    }
                // В двойственной сетке площадки следуют в порядке узлов граней, включая швы.
                k += corners.size;
            }
        },
        [&](Reader& r, int) { boundary.push_back({r.key(), r.ref(), r.id(), r.point(), r.point()}); });
    return boundary;
}
}

// Строит компактные данные для тех же узловых сумм, что и в обычной периодической области.
PeriodicPartition partition_periodic(const MeshPartition& part, CvView cv,
                                     const std::vector<PeriodicPair>& pairs, const Processes& w) {
    const auto labels = match_periodic(part, cv, pairs, w);
    const auto nodes = node_representatives(part, labels, w);
    auto edges = collect_edges(part, cv, labels, nodes, w);
    auto volumes = collect_volumes(part, cv, nodes, w);
    auto boundary = collect_physical_boundary(part, cv, nodes, pairs, w);
    PeriodicPartition grid;
    w.local([&] {
        // Уплотняем номера представителей, нужных этому процессу для собственных узлов и концов рёбер.
        std::map<Id, int> needed;
        for (Ref n : nodes)
            needed.emplace(n.key.first, n.rank);
        for (const auto& e : edges) {
            needed.emplace(e.a.key.first, e.a.rank);
            needed.emplace(e.b.key.first, e.b.rank);
        }
        for (const auto& n : needed) {
            grid.ids.push_back(n.first);
            grid.owners.push_back(n.second);
        }
        const auto local = [&](Ref n) -> Id {
            const auto it = std::lower_bound(grid.ids.begin(), grid.ids.end(), n.key.first);
            if (it == grid.ids.end() || *it != n.key.first)
                throw MeshError("periodic: missing local node");
            return static_cast<Id>(it - grid.ids.begin());
        };
        for (Ref n : nodes)
            grid.node_dof.push_back(local(n));
        grid.volumes.resize(grid.ids.size());
        // Складываем вклады в порядке обычного расчёта независимо от очередности прихода сообщений.
        std::sort(volumes.begin(), volumes.end(),
                  [](const Volume& a, const Volume& b) { return a.original < b.original; });
        for (const auto& v : volumes)
            grid.volumes[local(v.node)] += v.value;
        std::sort(edges.begin(), edges.end(),
                  [](const EdgeData& a, const EdgeData& b) { return a.key < b.key; });
        for (const auto& e : edges)
            grid.edges.push_back({local(e.a), local(e.b), vector(e.area), e.key});
        std::sort(boundary.begin(), boundary.end(),
                  [](const Piece& a, const Piece& b) { return a.order < b.order; });
        for (const auto& p : boundary)
            grid.boundary.push_back({local(p.node), p.order.first, p.patch, vector(p.area), p.position});
    });
    return grid;
}
}
