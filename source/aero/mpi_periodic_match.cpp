#include "mpi_periodic_detail.h"
#include <limits>

namespace aero {
namespace periodic_mpi {
namespace {
// Узел периодической грани и относящийся к нему вектор площади.
struct Corner {
    // Номер вершины сопоставляемой грани.
    Id node;
    // Её вклад в вектор площади.
    Coordinate area;
};
using Face = std::vector<Corner>;
using Ids = std::vector<Id>;
// Собственные узлы и грани одного периодического участка.
struct Patch {
    // Узлы патча во владении процесса.
    std::set<Id> owned;
    // Грани для поиска соответствий.
    std::vector<Face> faces;
};
using Links = std::set<std::pair<Ref, Ref>>;

// Собирает локальные грани и собственные узлы периодических участков.
std::map<std::string, Patch> collect(const MeshPartition& part, CvView cv,
                                     const std::vector<PeriodicPair>& pairs, int rank) {
    std::map<std::string, Patch> patches;
    for (const auto& p : pairs) {
        patches.emplace(p.patch, Patch());
        patches.emplace(p.pair, Patch());
    }
    const auto m = part.mesh.view();
    Id k = 0;
    for (Id b = 0; b < m.boundary_face_count; ++b) {
        const auto nodes = m.face_nodes(m.boundary_faces[b]);
        const auto it = patches.find(part.mesh.data().patch_names[m.boundary_patch[b]]);
        if (it != patches.end()) {
            auto& patch = it->second;
            for (Id n : nodes)
                if (part.node_owners[n] == rank)
                    patch.owned.insert(n);
            // Владелец видит всю примыкающую ячейку и может вычислить площади всех её граней.
            if (part.node_owners[*std::min_element(nodes.begin(), nodes.end())] == rank) {
                Face face;
                for (Id j = 0; j < nodes.size; ++j)
                    face.push_back({nodes[j], point(cv.boundary[k + j].area)});
                patch.faces.push_back(std::move(face));
            }
        }
        k += nodes.size;
    }
    return patches;
}

// Ограничивающий параллелепипед для отсева кандидатов периодического сопоставления.
struct Box {
    // Покоординатные нижняя и верхняя границы области поиска.
    Coordinate lo, hi;
    // Инициализирует пустые границы для последующего расширения точками.
    Box() {
        lo.fill(std::numeric_limits<double>::infinity());
        hi.fill(-std::numeric_limits<double>::infinity());
    }
    // Расширяет границы параллелепипеда до включения заданной точки.
    void include(Coordinate p) {
        for (unsigned i = 0; i < 3; ++i) {
            lo[i] = std::min(lo[i], p[i]);
            hi[i] = std::max(hi[i], p[i]);
        }
    }
    // Проверяет попадание точки в ограничивающий параллелепипед с допуском.
    bool contains(Coordinate p, double tol) const {
        for (unsigned i = 0; i < 3; ++i)
            if (p[i] + tol < lo[i] || p[i] - tol > hi[i])
                return false;
        return true;
    }
    // Проверяет пересечение ограничивающих параллелепипедов с допуском.
    bool overlaps(const Box& b, double tol) const {
        for (unsigned i = 0; i < 3; ++i)
            if (hi[i] + tol < b.lo[i] || lo[i] - tol > b.hi[i])
                return false;
        return true;
    }
};
// Сопоставляет узлы периодических участков после заданного сдвига.
std::map<Id, Ref> match_nodes(const MeshPartition& part, const Patch& source, const Patch& target,
                              Coordinate shift, double tol, const std::string& label, const Processes& w,
                              Links& links) {
    const auto m = part.mesh.view();
    // Сначала ограничиваем поиск пар пространственным диапазоном локального участка.
    Box target_box;
    std::vector<double> bounds;
    // Индекс вершин по наиболее протяжённой оси ускоряет поиск кандидатов в пределах допуска.
    std::multimap<double, Id> bins;
    std::set<Id> queries;
    unsigned axis = 0;
    w.local([&] {
        for (Id i : target.owned)
            target_box.include(m.coordinate(i));
        const auto span = target_box.hi - target_box.lo;
        axis = static_cast<unsigned>(std::max_element(span.begin(), span.end()) - span.begin());
        for (Id i : target.owned)
            bins.emplace(m.coordinate(i)[axis], i);
        bounds.resize(6 * static_cast<std::size_t>(w.size));
        queries = source.owned;
        for (const auto& f : source.faces)
            for (const auto& c : f)
                queries.insert(c.node);
    });
    if (w.minimum(target.owned.empty() ? 1.0 : 0.0))
        throw MeshError(label + "empty target patch");
    const double own[] = {target_box.lo[0], target_box.lo[1], target_box.lo[2],
                          target_box.hi[0], target_box.hi[1], target_box.hi[2]};
    w.allgather(own, 6, bounds.data());
    Messages replies;
    exchange(
        w,
        [&](Messages& out) {
            replies.resize(w.size);
            Box region;
            for (Id i : queries)
                region.include(m.coordinate(i) + shift);
            std::vector<std::pair<int, Box>> candidates;
            for (int p = 0; p < w.size; ++p) {
                const auto* b = bounds.data() + 6 * static_cast<std::size_t>(p);
                Box box;
                std::copy_n(b, 3, box.lo.begin());
                std::copy_n(b + 3, 3, box.hi.begin());
                if (region.overlaps(box, tol))
                    candidates.emplace_back(p, box);
            }
            for (Id i : queries) {
                const Coordinate x = m.coordinate(i) + shift;
                for (const auto& peer : candidates)
                    if (peer.second.contains(x, tol)) {
                        put(out[peer.first], i);
                        put(out[peer.first], x);
                    }
            }
        },
        [&](Reader& r, int p) {
            const Id i = r.id();
            const Coordinate x = r.point();
            for (auto it = bins.lower_bound(x[axis] - tol); it != bins.end() && it->first <= x[axis] + tol;
                 ++it) {
                if (coordinate_length(m.coordinate(it->second) - x) <= tol) {
                    put(replies[p], i);
                    put(replies[p], vertex(part, it->second));
                }
            }
        });
    std::map<Id, Ref> partners;
    exchange_replies(w, replies, [&](Reader& r, int) {
            const Id i = r.id();
            const Ref other = r.ref();
            if (!partners.emplace(i, other).second)
                throw MeshError(label + "ambiguous node " + std::to_string(part.nodes.at(i)));
        });
    std::set<Id> hits;
    exchange(
        w,
        [&](Messages& out) {
            for (Id i : queries)
                if (!partners.count(i))
                    throw MeshError(label + "unmatched node " + std::to_string(part.nodes[i]));
            for (Id i : source.owned) {
                const Ref other = partners.at(i);
                put(out[other.rank], other.key.first);
                links.emplace(vertex(part, i), other);
            }
        },
        [&](Reader& r, int) {
            if (!hits.insert(r.id()).second)
                throw MeshError(label + "target node matched more than once");
        });
    w.local([&] {
        if (hits.size() != target.owned.size())
            throw MeshError(label + "unmatched target nodes");
    });
    return partners;
}

// Проверяет соответствие граней по сопоставленным периодическим узлам.
void match_faces(const MeshPartition& part, const Patch& source, const Patch& target,
                 const std::map<Id, Ref>& partners, const std::string& label, const Processes& w,
                 Links& links) {
    std::map<Ids, const Face*> faces;
    exchange(
        w,
        [&](Messages& out) {
            for (const auto& f : target.faces) {
                Ids key;
                for (const Corner& c : f)
                    key.push_back(part.nodes[c.node]);
                std::sort(key.begin(), key.end());
                if (!faces.emplace(key, &f).second)
                    throw MeshError(label + "duplicate target face");
            }
            for (const auto& f : source.faces) {
                const auto first =
                    std::min_element(f.begin(), f.end(), [&](const Corner& a, const Corner& b) {
                        return partners.at(a.node) < partners.at(b.node);
                    });
                auto& msg = out[partners.at(first->node).rank];
                put(msg, f.size());
                for (std::size_t j = 0; j < f.size(); ++j) {
                    const auto& c = f[j];
                    const Ref other = partners.at(c.node);
                    put(msg, other.key.first);
                    put(msg, c.area);
                    if (f.size() > 1) {
                        const Id next = f[(j + 1) % f.size()].node;
                        links.emplace(edge(vertex(part, c.node), vertex(part, next)),
                                      edge(other, partners.at(next)));
                    }
                }
            }
        },
        [&](Reader& r, int) {
            const Id size = r.id();
            Ids cycle;
            std::map<Id, Coordinate> areas;
            for (Id j = 0; j < size; ++j) {
                const Id n = r.id();
                cycle.push_back(n);
                areas.emplace(n, r.point());
            }
            Ids key = cycle;
            std::sort(key.begin(), key.end());
            const auto it = faces.find(key);
            if (it == faces.end())
                throw MeshError(label + "unmatched face");
            const auto& face = *it->second;
            std::set<Key> edges;
            for (std::size_t j = 0; j < face.size(); ++j) {
                const Id n = part.nodes[face[j].node], next = part.nodes[face[(j + 1) % face.size()].node];
                edges.insert(std::minmax(n, next));
                const Coordinate a = areas.at(n), b = face[j].area;
                if (!periodic_detail::compatible_areas(a, b))
                    throw MeshError(label + "incompatible face areas");
            }
            for (std::size_t j = 0; j < cycle.size(); ++j)
                if (!edges.count(std::minmax(cycle[j], cycle[(j + 1) % cycle.size()])))
                    throw MeshError(label + "unmatched face edge");
            faces.erase(it);
        });
    w.local([&] {
        if (!faces.empty())
            throw MeshError(label + "unmatched target face");
    });
}
}

// Сопоставляет периодические участки разных рангов и строит общие метки узлов.
Labels match_periodic(const MeshPartition& part, CvView cv, const std::vector<PeriodicPair>& pairs,
                      const Processes& w) {
    std::map<std::string, Patch> patches;
    double tol = 0;
    w.local([&] {
        patches = collect(part, cv, pairs, w.rank);
        tol = periodic_detail::match_tolerance(pairs);
    });
    Links links;
    for (const auto& p : pairs) {
        const auto& a = patches.at(p.patch);
        const auto& b = patches.at(p.pair);
        const auto label = "periodic " + p.patch + " -> " + p.pair + ": ";
        const auto partners = match_nodes(
            part, a, b, p.translation, tol, label, w, links);
        match_faces(part, a, b, partners, label, w, links);
    }
    // Передаём наименьший исходный номер по связям между совпавшими участками границы.
    // Так угол или ребро нескольких периодических пар получает одного представителя.
    std::map<Key, std::set<Ref>> adjacent;
    Labels labels;
    exchange(
        w,
        [&](Messages& out) {
            for (const auto& link : links) {
                const Ref a = link.first, b = link.second;
                put(out[a.rank], a.key);
                put(out[a.rank], b);
                put(out[b.rank], b.key);
                put(out[b.rank], a);
            }
        },
        [&](Reader& r, int) {
            const Key key = r.key();
            const Ref other = r.ref();
            adjacent[key].insert(other);
            labels.emplace(key, Ref{key, w.rank});
        });
    bool changed;
    do {
        changed = false;
        exchange(
            w,
            [&](Messages& out) {
                for (const auto& v : adjacent)
                    for (Ref other : v.second) {
                        put(out[other.rank], other.key);
                        put(out[other.rank], labels.at(v.first));
                    }
            },
            [&](Reader& r, int) {
                const Key key = r.key();
                const Ref candidate = r.ref();
                auto& current = labels.at(key);
                if (candidate < current) {
                    current = candidate;
                    changed = true;
                }
            });
    } while (w.minimum(changed ? 0.0 : 1.0) == 0);
    return labels;
}
}
}
