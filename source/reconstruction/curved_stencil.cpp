#include "curved_stencil.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <set>
#include <cmath>

namespace aero {
namespace detail {
namespace {
using Id = MeshIndex;
using List = std::vector<Id>;
double dot(Coordinate a, Coordinate b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Coordinate cross(Coordinate a, Coordinate b) {
    return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
}
Coordinate scaled(Coordinate a, double s) {
    for (double& value : a)
        value *= s;
    return a;
}
Coordinate unit(Coordinate a) {
    const double length = coordinate_length(a);
    return length > 0. ? scaled(a, 1. / length) : Coordinate{};
}
Id other(MeshView m, Id edge, Id node) {
    return m.edge_begin[edge] == node ? m.edge_end[edge] : m.edge_begin[edge];
}
bool contains(MeshIndexSpan nodes, Id node) {
    return std::find(nodes.begin(), nodes.end(), node) != nodes.end();
}
bool structured_cell(MeshView m, Id cell) {
    if (m.dimension == 2)
        return m.cell_faces(cell).size == 4;
    if (m.dimension != 3 || m.cell_nodes(cell).size != 8 || m.cell_faces(cell).size != 6)
        return false;
    for (Id face : m.cell_faces(cell))
        if (m.face_nodes(face).size != 4)
            return false;
    return true;
}
// Находим пары оснований по связности. Боковые грани треугольной призмы не могут начать новый слой.
template <class Visit> void for_bases(MeshView m, Id cell, Visit visit) {
    const auto faces = m.cell_faces(cell);
    const auto count = m.cell_nodes(cell).size;
    if (m.dimension == 2
            ? faces.size != 4
            : !(m.dimension == 3 && ((count == 6 && faces.size == 5) || (count == 8 && faces.size == 6))))
        return;
    for (Id i = 0; i < faces.size; ++i) {
        const auto a = m.face_nodes(faces[i]);
        if (2 * a.size != count)
            continue;
        for (Id j = i + 1; j < faces.size; ++j) {
            const auto b = m.face_nodes(faces[j]);
            if (a.size == b.size && std::none_of(a.begin(), a.end(), [&](Id n) { return contains(b, n); }))
                visit(faces[i], faces[j]);
        }
    }
}
Coordinate position(MeshView m, const RayPoint& point) {
    Coordinate x{};
    for (const auto& w : point.weights)
        x = x + scaled(m.coordinate(w.first), w.second);
    return x;
}
} // namespace

PrismLayers::PrismLayers(const Mesh& mesh, const std::vector<std::string>& labels) {
    const auto m = mesh.view();
    opposite.resize(m.face_count);
    for (Id c = 0; c < m.cell_count; ++c)
        for_bases(m, c, [&](Id a, Id b) {
            opposite[a].push_back(b);
            opposite[b].push_back(a);
        });
    for (Id i = 0; i < m.boundary_face_count; ++i) {
        const auto& name = mesh.data().patch_names[m.boundary_patch[i]];
        if (std::find(labels.begin(), labels.end(), name) != labels.end() &&
            !opposite[m.boundary_faces[i]].empty())
            wall_faces.push_back(m.boundary_faces[i]);
    }
}

std::vector<int> PrismLayers::local_depths(MeshView m, const Sync& sync) const {
    std::vector<int> depth(m.node_count, -1), visited(m.face_count, -1);
    List frontier = wall_faces;
    for (int level = 0; sync ? sync(frontier) : !frontier.empty(); ++level) {
        List next;
        for (Id face : frontier) {
            if (visited[face] >= 0)
                continue;
            visited[face] = level;
            for (Id n : m.face_nodes(face))
                if (depth[n] < 0)
                    depth[n] = level;
            for (Id f : opposite[face])
                if (visited[f] < 0)
                    next.push_back(f);
        }
        frontier.swap(next);
    }
    return depth;
}

CurvedStencil::CurvedStencil(const Mesh& mesh, const ReconstructionOptions& o)
    : mesh_(mesh.view()), cosine_(std::cos(o.max_turn_degrees * std::acos(-1.) / 180.)) {
    if (!(o.max_turn_degrees > 0. && o.max_turn_degrees < 90.))
        throw ReconstructionError("Invalid stencil_max_angle");
    if (o.geometry == StencilGeometry::Ijk || o.geometry == StencilGeometry::Auto)
        prepare_ijk();
    if (o.geometry == StencilGeometry::Pl || o.geometry == StencilGeometry::Auto)
        prepare_pl(mesh, o);
}

void CurvedStencil::prepare_ijk() {
    const auto m = mesh_;
    std::vector<List> edges(m.node_count), cells(m.edge_count);
    std::vector<char> regular(m.node_count, 1);
    for (Id c = 0; c < m.cell_count; ++c) {
        if (!structured_cell(m, c))
            for (Id n : m.cell_nodes(c))
                regular[n] = 0;
        for (Id e : m.cell_edges(c))
            cells[e].push_back(c);
    }
    for (Id e = 0; e < m.edge_count; ++e) {
        edges[m.edge_begin[e]].push_back(e);
        edges[m.edge_end[e]].push_back(e);
    }
    next_.resize(m.edge_count, {{m.edge_count, m.edge_count}});
    parallel_for(m.edge_count, [&](Id e) {
        for (unsigned side = 0; side < 2; ++side) {
            const Id n = side ? m.edge_end[e] : m.edge_begin[e];
            if (!regular[n] || edges[n].size() > 2 * m.dimension)
                continue;
            Id candidate = m.edge_count;
            for (Id next : edges[n]) {
                // Поперечное ребро лежит с исходным в общей ячейке. Продолжение
                // координатной линии — единственное оставшееся ребро звезды узла.
                const Id opposite = other(m, next, n);
                if (std::any_of(cells[e].begin(), cells[e].end(),
                                [&](Id c) { return contains(m.cell_nodes(c), opposite); }))
                    continue;
                if (candidate != m.edge_count) {
                    candidate = m.edge_count;
                    break;
                }
                candidate = next;
            }
            next_[e][side] = candidate;
        }
    });
}

// Если продолжение задано связностью, обе опорные точки — узлы сетки.
// В 2D пристенный слой сам является ломаной. Для него используем тот же
// обход, что и для IJK: пересечения с плоскостью нужны только для PL в 3D.
template <class Next>
bool CurvedStencil::follow_line(Id edge, bool fifth, bool limit_each_turn, Next next,
                                CurvedPoints& points) const {
    const auto m = mesh_;
    for (unsigned side = 0; side < 2; ++side) {
        const Id node = side ? m.edge_end[edge] : m.edge_begin[edge];
        const Id first_edge = next(edge, node);
        if (first_edge == m.edge_count)
            return false;
        const Id first = other(m, first_edge, node);
        const auto origin = m.coordinate(node);
        const auto forward = unit(origin - m.coordinate(other(m, edge, node)));
        const auto step = m.coordinate(first) - origin;
        const double length = coordinate_length(step);
        if (!(length > 0.) || dot(step, forward) < cosine_ * length)
            return false;
        points.near[side].distance = length;
        points.near[side].weights[first] = 1.;
        if (!fifth)
            continue;

        const Id second_edge = next(first_edge, first);
        if (second_edge == m.edge_count)
            continue;
        const Id second = other(m, second_edge, first);
        const auto along = m.coordinate(second) - origin;
        const auto last = m.coordinate(second) - m.coordinate(first);
        const double spacing = coordinate_length(last);
        if (!(spacing > 0.) || dot(along, forward) < cosine_ * coordinate_length(along))
            continue;
        if (limit_each_turn && dot(step, last) < cosine_ * length * spacing)
            continue;
        points.far[side].distance = length + spacing;
        points.far[side].weights[second] = 1.;
    }
    return true;
}

void CurvedStencil::prepare_pl(const Mesh& mesh, const ReconstructionOptions& o) {
    const auto m = mesh_;
    if (o.wall_labels.empty() && o.node_layers.empty())
        return;
    layer_ = o.node_layers.empty() ? PrismLayers(mesh, o.wall_labels).local_depths(m) : o.node_layers;
    if (layer_.size() != m.node_count)
        throw ReconstructionError("PL layer numbering does not match mesh");
    if (m.dimension == 3)
        layer_faces_.resize(m.node_count);
    layer_edges_.resize(m.node_count);
    if (m.dimension == 3)
        normal_.resize(m.edge_count);
    std::vector<unsigned> counts(m.edge_count);
    std::vector<char> marked(m.face_count);
    for (Id c = 0; c < m.cell_count; ++c)
        for_bases(m, c, [&](Id a, Id b) {
            const int depth = layer_[m.face_nodes(a)[0]];
            const int next = layer_[m.face_nodes(b)[0]];
            if (depth < 0 || next < 0 || std::abs(depth - next) != 1)
                return;
            for (Id n : m.face_nodes(a))
                if (layer_[n] != depth)
                    return;
            for (Id n : m.face_nodes(b))
                if (layer_[n] != next)
                    return;
            const auto growth =
                scaled(m.coordinate(m.face_nodes(b)[0]) - m.coordinate(m.face_nodes(a)[0]), next - depth);
            for (Id face : {a, b}) {
                if (marked[face])
                    continue;
                marked[face] = 1;
                const auto nodes = m.face_nodes(face);
                if (m.dimension == 3)
                    for (Id n : nodes)
                        layer_faces_[n].push_back(face);
                Coordinate area{};
                for (Id k = 1; k + 1 < nodes.size; ++k)
                    area = area + cross(m.coordinate(nodes[k]) - m.coordinate(nodes[0]),
                                        m.coordinate(nodes[k + 1]) - m.coordinate(nodes[0]));
                if (dot(area, growth) < 0.)
                    area = scaled(area, -1.);
                for (Id edge : m.face_edges(face)) {
                    if (!counts[edge]++) {
                        layer_edges_[m.edge_begin[edge]].push_back(edge);
                        layer_edges_[m.edge_end[edge]].push_back(edge);
                    }
                    if (m.dimension == 3)
                        normal_[edge] = normal_[edge] + unit(area);
                }
            }
        });
    for (Id e = 0; e < normal_.size(); ++e)
        normal_[e] = counts[e] <= 2 ? unit(normal_[e]) : Coordinate{};
}

bool CurvedStencil::pl_points(Id e, bool fifth, CurvedPoints& p) const {
    const auto m = mesh_;
    if (layer_.empty() || layer_[m.edge_begin[e]] < 0 || layer_[m.edge_begin[e]] != layer_[m.edge_end[e]])
        return false;
    if (m.dimension == 2) {
        const auto continuation = [&](Id edge, Id node) {
            const auto& incident = layer_edges_[node];
            if (incident.size() == 2) {
                if (incident[0] == edge)
                    return incident[1];
                if (incident[1] == edge)
                    return incident[0];
            }
            return m.edge_count;
        };
        return follow_line(e, fifth, false, continuation, p);
    }
    if (coordinate_length(normal_[e]) == 0.)
        return false;
    for (unsigned side = 0; side < 2; ++side) {
        const Id n = side ? m.edge_end[e] : m.edge_begin[e];
        const auto origin = m.coordinate(n);
        const auto direction = unit(origin - m.coordinate(other(m, e, n)));
        // Сечение содержит ребро и внешнюю нормаль слоя. Его пересечение с
        // контуром кольца даёт координату и два линейных веса опорной точки.
        const auto plane = unit(cross(direction, normal_[e]));
        if (coordinate_length(plane) == 0.)
            return false;
        std::set<Id> seen{n}, ring{n};
        for (unsigned level = 0; level < (fifth ? 2u : 1u); ++level) {
            std::set<Id> next;
            for (Id node : ring)
                for (Id face : layer_faces_[node])
                    for (Id v : m.face_nodes(face))
                        if (!seen.count(v))
                            next.insert(v);
            seen.insert(next.begin(), next.end());
            RayPoint best;
            double best_distance = 0.;
            for (Id v : next) {
                const auto delta = m.coordinate(v) - origin;
                auto consider = [&](Id w, double fraction) {
                    const auto offset = delta + scaled(m.coordinate(w) - m.coordinate(v), fraction);
                    const double distance = coordinate_length(offset);
                    if (!(distance > best_distance && dot(direction, offset) >= cosine_ * distance))
                        return;
                    best_distance = distance;
                    best.weights.clear();
                    if (fraction < 1.)
                        best.weights[v] = 1. - fraction;
                    if (fraction > 0.)
                        best.weights[w] = fraction;
                    best.distance = distance;
                };
                for (Id candidate : layer_edges_[v]) {
                    const Id w = other(m, candidate, v);
                    if (w < v || !next.count(w) || dot(normal_[candidate], normal_[e]) <= 0.)
                        continue;
                    const auto other_offset = m.coordinate(w) - origin;
                    // При повороте сетки вершина в плоскости сечения может
                    // отклониться от нуля на ошибку округления скалярного произведения.
                    const double tolerance = 32. * std::numeric_limits<double>::epsilon() *
                                             std::max(coordinate_length(delta), coordinate_length(other_offset));
                    double h1 = dot(plane, delta), h2 = dot(plane, other_offset);
                    if (std::abs(h1) <= tolerance) h1 = 0.;
                    if (std::abs(h2) <= tolerance) h2 = 0.;
                    if (h1 == 0.)
                        consider(w, 0.);
                    if (h2 == 0.)
                        consider(w, 1.);
                    if ((h1 < 0. && h2 > 0.) || (h1 > 0. && h2 < 0.))
                        consider(w, h1 / (h1 - h2));
                }
            }
            if (!level) {
                if (best.weights.empty())
                    return false;
                p.near[side] = best;
            } else if (!best.weights.empty() && best_distance > p.near[side].distance) {
                best.distance =
                    p.near[side].distance + coordinate_length(position(m, best) - position(m, p.near[side]));
                p.far[side] = best;
            }
            ring.swap(next);
        }
    }
    return true;
}

bool CurvedStencil::points(Id edge, bool fifth, CurvedPoints& out) const {
    if (!next_.empty()) {
        const auto continuation = [&](Id e, Id n) { return next_[e][mesh_.edge_end[e] == n]; };
        if (follow_line(edge, fifth, true, continuation, out))
            return true;
    }
    if (layer_.empty())
        return false;
    out = CurvedPoints{};
    return pl_points(edge, fifth, out);
}
} // namespace detail
} // namespace aero
