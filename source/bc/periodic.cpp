#include "periodic.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace aero {
namespace {
using Id = MeshIndex;
using Ids = std::vector<Id>;
// Преобразует координаты в геометрический вектор контрольного объёма.
Coordinate vector_of(CvVector v) {
    return {{v.x, v.y, v.z}};
}
// Для каждой группы периодических копий представителем служит узел или ребро с наименьшим номером.
struct Groups {
    Ids parent;
    // Инициализирует отдельное множество для каждого элемента.
    explicit Groups(Id count) : parent(count) { std::iota(parent.begin(), parent.end(), Id(0)); }
    // Находит представителя группы периодически объединяемых узлов.
    Id root(Id i) {
        while (parent[i] != i) {
            // Укорачиваем путь до представителя при каждом следующем поиске.
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }
    // Объединяет группы двух сопряжённых периодических узлов.
    void join(Id a, Id b) {
        a = root(a);
        b = root(b);
        parent[std::max(a, b)] = std::min(a, b);
    }
};
// Узлы, грани и рёбра одного участка для периодического сопоставления.
struct Patch {
    std::set<Id> nodes;
    std::map<Ids, Id> faces;
    std::map<std::pair<Id, Id>, Id> edges;
};
// Добавляет грань и её связность в описание периодического участка.
void add_face(Patch& patch, MeshFace f) {
    const MeshIndexSpan nodes = f.node_ids();
    Ids key(nodes.begin(), nodes.end());
    patch.nodes.insert(key.begin(), key.end());
    std::sort(key.begin(), key.end());
    patch.faces[key] = f.id();
    for (Id j = 0; j < f.edge_count(); ++j)
        patch.edges[std::minmax(nodes[j], nodes[(j + 1) % nodes.size])] = f.edge_id(j);
}
// Сопоставляет узлы сопряжённых участков после геометрического сдвига.
std::map<Id, Id> match_nodes(MeshView m, const Patch& a, const Patch& b, Coordinate shift, double tol,
                             const std::string& label) {
    if (a.nodes.empty() || a.nodes.size() != b.nodes.size() || a.faces.size() != b.faces.size())
        throw MeshError(label + "different patch sizes or empty patch");
    // Широкая координатная ось лучше всего разделяет узлы в одномерном поисковом индексе.
    Coordinate low = m.coordinate(*b.nodes.begin()), high = low;
    for (Id n : b.nodes) {
        const Coordinate p = m.coordinate(n);
        for (unsigned i = 0; i < 3; ++i) {
            low[i] = std::min(low[i], p[i]);
            high[i] = std::max(high[i], p[i]);
        }
    }
    Coordinate span = high - low;
    size_t axis = static_cast<size_t>(std::max_element(span.begin(), span.end()) - span.begin());
    std::multimap<double, Id> bins;
    for (Id n : b.nodes)
        bins.emplace(m.coordinate(n)[axis], n);
    std::map<Id, Id> matches;
    std::set<Id> used;
    const Ids sources(a.nodes.begin(), a.nodes.end());
    Ids partners(sources.size());
    parallel_for(sources.size(), [&](size_t k) {
        const Id n = sources[k];
        bool found = false;
        const Coordinate wanted = m.coordinate(n) + shift;
        // Сначала отбираем кандидатов по одной оси, затем проверяем полное расстояние.
        for (auto i = bins.lower_bound(wanted[axis] - tol); i != bins.end() && i->first <= wanted[axis] + tol;
             ++i) {
            if (coordinate_length(m.coordinate(i->second) - wanted) > tol)
                continue;
            if (found)
                throw MeshError(label + "ambiguous match at node " + std::to_string(n));
            partners[k] = i->second;
            found = true;
        }
        if (!found)
            throw MeshError(label + "unmatched node " + std::to_string(n));
    });
    // Два исходных узла не могут претендовать на одну и ту же копию.
    for (size_t k = 0; k < sources.size(); ++k) {
        if (!used.insert(partners[k]).second)
            throw MeshError(label + "ambiguous match at node " + std::to_string(sources[k]));
        matches.emplace(sources[k], partners[k]);
    }
    return matches;
}

// Переводит соответствие координат двух границ в группы общих узлов и рёбер.
void join_patch_topology(const Patch& a, const Patch& b, const std::map<Id, Id>& match,
                         const std::map<std::pair<Id, Id>, Coordinate>& areas, const std::string& label,
                         Groups& nodes, Groups& edges) {
    for (const auto& n : match)
        nodes.join(n.first, n.second);
    // У грани порядок обхода может отличаться. Ищем её по отсортированному набору узлов.
    for (const auto& face : a.faces) {
        Ids key;
        for (Id n : face.first)
            key.push_back(match.at(n));
        std::sort(key.begin(), key.end());
        const auto found = b.faces.find(key);
        if (found == b.faces.end())
            throw MeshError(label + "unmatched face " + std::to_string(face.second));
        // Внешние площадки на двух сторонах шва должны взаимно погашаться в каждом узле.
        for (Id n : face.first) {
            const Coordinate u = areas.at({face.second, n}), v = areas.at({found->second, match.at(n)});
            if (!periodic_detail::compatible_areas(u, v))
                throw MeshError(label + "incompatible area at face " + std::to_string(face.second));
        }
    }
    // Для каждого ребра объединяем и его образ. Знак площади определим при сборке.
    for (const auto& edge : a.edges) {
        auto found = b.edges.find(std::minmax(match.at(edge.first.first), match.at(edge.first.second)));
        if (found == b.edges.end())
            throw MeshError(label + "unmatched edge " + std::to_string(edge.second));
        edges.join(edge.second, found->second);
    }
}
}
// Выбирает геометрический допуск сопоставления периодических участков.
double periodic_detail::match_tolerance(const std::vector<PeriodicPair>& pairs, double tol) {
    if (pairs.empty() || !std::isfinite(tol) || tol < 0)
        throw MeshError("invalid periodic pairs or tolerance");
    double shortest = std::numeric_limits<double>::max();
    for (const auto& p : pairs) {
        const double L = coordinate_length(p.translation);
        if (!(L > 0) || !std::isfinite(L))
            throw MeshError("invalid periodic translation");
        shortest = std::min(shortest, L);
    }
    if (!tol)
        tol = shortest * 1e-10;
    if (!(tol > 0 && tol < 0.5 * shortest))
        throw MeshError("unresolved periodic translation");
    return tol;
}
// Проверяет согласованность площадей сопряжённых периодических граней.
bool periodic_detail::compatible_areas(const Coordinate& a, const Coordinate& b) {
    return coordinate_length(a + b) <= 1e-8 * std::max(coordinate_length(a), coordinate_length(b));
}
// Сопоставляет периодические участки и строит общие степени свободы.
PeriodicDomain::PeriodicDomain(const Mesh& mesh, const std::vector<PeriodicPair>& pairs, double tol)
    : mesh_(mesh), tolerance_(periodic_detail::match_tolerance(pairs, tol)) {
    const MeshView m = mesh.view();
    const MedianDual dual(mesh);
    const CvView cv = dual.view();
    std::set<std::string> paired;
    std::map<std::string, Patch> patches;
    for (const PeriodicPair& p : pairs) {
        if (!paired.insert(p.patch).second || !paired.insert(p.pair).second)
            throw MeshError("repeated periodic patch: " + p.patch + " / " + p.pair);
    }
    for (Id k = 0; k < m.boundary_face_count; ++k) {
        const std::string& name = mesh.data().patch_names[m.boundary_patch[k]];
        if (paired.count(name))
            add_face(patches[name], m.face(m.boundary_faces[k]));
    }
    Groups nodes(m.node_count), edges(m.edge_count);
    std::map<std::pair<Id, Id>, Coordinate> areas;
    for (Id k = 0; k < cv.boundary_count; ++k) {
        const CvBoundary& s = cv.boundary[k];
        if (paired.count(mesh.data().patch_names[s.patch]))
            areas[{s.face, s.node}] = vector_of(s.area);
    }
    for (const PeriodicPair& p : pairs) {
        const Patch &a = patches[p.patch], &b = patches[p.pair];
        const std::string label = "periodic " + p.patch + " -> " + p.pair + ": ";
        auto match =
            match_nodes(m, a, b, p.translation, tolerance_, label);
        join_patch_topology(a, b, match, areas, label, nodes, edges);
    }
    // Номер представителя минимален, поэтому при обходе копии его расчётный индекс уже известен.
    node_dof_.resize(m.node_count);
    for (Id n = 0; n < m.node_count; ++n) {
        const Id root = nodes.root(n);
        if (root == n) {
            node_dof_[n] = volumes_.size();
            volumes_.push_back(0);
        } else
            node_dof_[n] = node_dof_[root];
        volumes_[node_dof_[n]] += cv.volumes[n];
    }
    Ids edge_index(m.edge_count);
    for (Id e = 0; e < m.edge_count; ++e) {
        const Id root = edges.root(e);
        if (root == e) {
            edge_index[e] = edges_.size();
            edges_.push_back({node_dof_[m.edge_begin[e]], node_dof_[m.edge_end[e]], e, {}});
        }
        PeriodicEdge& entry = edges_[edge_index[root]];
        // Направление сохраняется и у ребра, оба конца которого периодичность свела к одному узлу.
        const Coordinate d = m.coordinate(m.edge_end[e]) - m.coordinate(m.edge_begin[e]);
        const Coordinate ref = m.coordinate(m.edge_end[root]) - m.coordinate(m.edge_begin[root]);
        const int sign = coordinate_length(d - ref) <= coordinate_length(d + ref) ? 1 : -1;
        entry.area.x += sign * cv.edge_areas[e].x;
        entry.area.y += sign * cv.edge_areas[e].y;
        entry.area.z += sign * cv.edge_areas[e].z;
    }
    // После склейки швы стали внутренними. В граничный поток попадут только физические границы.
    for (Id k = 0; k < cv.boundary_count; ++k) {
        CvBoundary piece = cv.boundary[k];
        if (paired.count(mesh.data().patch_names[piece.patch]))
            continue;
        piece.node = node_dof_[piece.node];
        boundary_.push_back(piece);
    }
}
// Возвращает невладеющее представление склеенной периодической геометрии.
PeriodicView PeriodicDomain::view() const {
    return {volumes_.size(),  edges_.size(),   boundary_.size(), node_dof_.size(),
            node_dof_.data(), volumes_.data(), edges_.data(),    boundary_.data()};
}
} // namespace aero
