#include "cv.h"
#include "../parallel/parallel_for.h"
#include "../mesh/mesh.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aero {
namespace {
typedef CvVector V;
// Складывает компоненты геометрических векторов.
V operator+(V a, V b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
// Вычисляет разность геометрических векторов.
V operator-(V a, V b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
// Умножает все компоненты геометрического вектора на общий масштаб.
V operator*(V a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}
// Проверяет конечность всех проверяемых компонент.
bool finite(V a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

// Представление сетки и текущей ячейки для вычисления контрольных объёмов.
struct Geometry {
    MeshView mesh;
    MeshIndex cell;
    // Вычисляет разность координат двух узлов.
    V delta(MeshIndex a, MeshIndex b) const {
        return {mesh.x[a] - mesh.x[b], mesh.y[a] - mesh.y[b], mesh.z[a] - mesh.z[b]};
    }
    // Смещение среднего положения вершин относительно первой вершины в единицах сетки.
    V center(MeshIndexSpan nodes) const {
        V sum = {};
        for (MeshIndex id : nodes)
            sum = sum + delta(id, nodes[0]);
        return sum * (1.0 / static_cast<double>(nodes.size));
    }
    // Проверяет обязательное условие и сообщает ошибку при его нарушении.
    void require(bool valid, const char* reason) const {
        if (!valid)
            throw MeshError("median dual: cell " + std::to_string(cell) + ": " + reason);
    }
};

// Сумма ориентированных площадей треугольников определяет нормаль многоугольника
// при любом направлении обхода и любом повороте его плоскости.
V polygon_normal(const Geometry& g, MeshCell cell, V center) {
    V sum = {};
    const MeshIndex origin = cell.node_id(0);
    for (MeshIndex k = 0; k < cell.face_count(); ++k) {
        const MeshFace f = cell.face(k);
        const V arm = g.delta(f.node_id(0), origin) - center;
        sum = sum + arm.cross(g.delta(f.node_id(1), f.node_id(0))) * cell.face_orientation(k);
    }
    const double length = sum.length();
    g.require(length > 0 && std::isfinite(length), "degenerate polygon");
    return sum * (1 / length);
}

// Проверяет плоскостность контура относительно заданной нормали с масштабным допуском.
void check_plane(const Geometry& g, MeshIndexSpan ids, V normal) {
    double span = 0, deviation = 0;
    for (MeshIndex id : ids) {
        const V v = g.delta(id, ids[0]);
        span = std::max(span, v.length());
        deviation = std::max(deviation, std::fabs(v.dot(normal)));
    }
    g.require(deviation <= 128 * std::numeric_limits<double>::epsilon() * span, "nonplanar polygon");
}
// Вклад одной ячейки в объёмы концов ребра и в ориентированную поверхность между ними.
struct Contribution {
    MeshIndex a, b; // Концы ребра в порядке обхода данной грани.
    MeshIndex edge; // Номер ребра в общей сетке.
    double volume;  // Каждый из двух узлов получает такую долю объёма.
    V area;         // Вектор площадки, направленный от edge_begin к edge_end.
};
const MeshIndex no_boundary = std::numeric_limits<MeshIndex>::max();

// В одном измерении граница между двумя узлами проходит через середину отрезка.
double split_segment(const Geometry& g, MeshCell cell, const std::vector<MeshIndex>& boundary_start,
                   std::vector<CvBoundary>& boundary, std::vector<Contribution>& parts) {
    const MeshView& m = g.mesh;
    const MeshIndex e = cell.edge_id(0), a = m.edge_begin[e], b = m.edge_end[e];
    const V edge = g.delta(b, a);
    const double length = edge.length();
    g.require(length > 0 && std::isfinite(length), "invalid line segment");
    // У осевого отрезка ненулевая компонента равна ±length. Деление даёт
    // точно ±1 даже для неодинаковых длин. Так постоянное давление не создаёт
    // ложный отток при проверке стационарности на неравномерной сетке.
    const V area = {edge.x / length, edge.y / length, edge.z / length};
    parts.push_back({a, b, e, length / 2, area});
    for (MeshIndex f : cell.face_ids()) {
        if (boundary_start[f] != no_boundary)
            boundary[boundary_start[f]].area = area * (m.face_nodes(f)[0] == a ? -1 : 1);
    }
    return length;
}

// Ширина ячейки вдоль нормали к грани. На симплексе это обычная высота,
// на прямоугольнике или параллелепипеде — расстояние до противоположной грани.
double normal_width(const Geometry& g, MeshIndexSpan vertices, MeshIndex origin, V area) {
    const double length = area.length();
    g.require(length > 0 && std::isfinite(length), "invalid face normal for CFL");
    const V unit = {area.x / length, area.y / length, area.z / length};
    double low = 0., high = 0.;
    for (MeshIndex node : vertices) {
        const double projection = g.delta(node, origin).dot(unit);
        low = std::min(low, projection);
        high = std::max(high, projection);
    }
    return high - low;
}

// Делит ячейку на доли узлов. Внешняя грань принадлежит ровно одной ячейке,
// поэтому её площадки можно заполнять сразу даже при параллельном обходе.
double collect_cell_contributions(const MeshView& m, MeshIndex id, const std::vector<MeshIndex>& boundary_start,
                                std::vector<CvBoundary>& boundary, std::vector<Contribution>& parts) {
    const Geometry g = {m, id};
    const MeshCell cell = m.cell(id);
    if (m.dimension == 1) {
        return split_segment(g, cell, boundary_start, boundary, parts);
    }
    const MeshIndex origin = cell.node_id(0);
    const V center = g.center(cell.node_ids());
    V normal = {};
    if (m.dimension == 2) {
        normal = polygon_normal(g, cell, center);
        check_plane(g, cell.node_ids(), normal);
    }
    double height = std::numeric_limits<double>::max();
    for (MeshIndex k = 0; k < cell.face_count(); ++k) {
        const MeshFace face = cell.face(k);
        const MeshIndex start = boundary_start[face.id()];
        const int sign = cell.face_orientation(k);
        const MeshIndexSpan nodes = face.node_ids();
        V face_center = {}, face_area = {};
        if (m.dimension == 3) {
            // Вершины могут лежать на разных высотах. Каждое ребро вместе с общим
            // средним положением вершин образует свой треугольник поверхности.
            face_center = g.center(nodes);
        }
        for (MeshIndex j = 0; j < face.edge_count(); ++j) {
            const MeshIndex next = (j + 1) % nodes.size;
            const MeshIndex a = nodes[j], b = nodes[next], e = face.edge_id(j);
            const V edge = g.delta(b, a);
            // Вектор от середины ребра к центру ячейки задаёт одно направление двойственной площадки.
            const V to_cell = center - g.delta(a, origin) - edge * 0.5;
            V dual_area, exterior;
            if (m.dimension == 2) {
                dual_area = to_cell.cross(normal) * sign;
                exterior = edge.cross(normal) * (0.5 * sign);
            } else {
                // В 3D второе направление ведёт к центру грани; половина векторного произведения даёт площадь треугольника.
                const V to_face = face_center - g.delta(a, nodes[0]) - edge * 0.5;
                dual_area = to_cell.cross(to_face) * (0.5 * sign);
                exterior = edge.cross(to_face) * (0.25 * sign);
            }
            // Две равные части по сторонам двойственной площадки принадлежат концам одного ребра.
            const double share = dual_area.dot(edge) / (2 * m.dimension);
            face_area = face_area + exterior;
            g.require(share > 0 && std::isfinite(share), "non-positive or non-finite median subcell");
            parts.push_back({a, b, e, share, dual_area * (a == m.edge_begin[e] ? 1 : -1)});
            if (start != no_boundary) {
                boundary[start + j].area = boundary[start + j].area + exterior;
                boundary[start + next].area = boundary[start + next].area + exterior;
            }
        }
        height = std::min(height, normal_width(g, cell.node_ids(), nodes[0], face_area));
    }
    g.require(height > 0 && std::isfinite(height), "invalid cell height for CFL");
    return height;
}
}

// Строит медианные объёмы на неподвижной сетке. Центр равен среднему координат вершин.
// Каждая часть многоугольника или пирамиды должна иметь положительную ориентированную меру.
MedianDual::MedianDual(const Mesh& mesh) : dimension_(mesh.data().dimension) {
    const MeshView m = mesh.view();
    volumes_.resize(m.node_count);
    min_heights_.assign(m.node_count, std::numeric_limits<double>::max());
    edge_areas_.resize(m.edge_count);
    std::vector<MeshIndex> boundary_start(m.face_count, no_boundary);
    for (MeshIndex b = 0; b < m.boundary_face_count; ++b) {
        const MeshIndex face = m.boundary_faces[b];
        boundary_start[face] = boundary_.size();
        for (MeshIndex node : m.face_nodes(face))
            boundary_.push_back({node, face, m.boundary_patch[b], {}, m.coordinate(node)});
    }
    // Считаем геометрию ячеек параллельно, затем складываем в порядке сетки.
    // Номер потока не влияет на округление; пакет ограничивает временную память.
    const MeshIndex batch = 1024;
    std::vector<std::vector<Contribution>> parts(std::min(batch, m.cell_count));
    std::vector<double> heights(parts.size());
    for (MeshIndex first = 0; first < m.cell_count;) {
        const MeshIndex size = std::min(batch, m.cell_count - first);
        parallel_for(
            size,
            [&](MeshIndex i) {
                parts[i].clear();
                heights[i] = collect_cell_contributions(m, first + i, boundary_start, boundary_, parts[i]);
            },
            128);
        for (MeshIndex i = 0; i < size; ++i) {
            // Высота нужна на каждом шаге CFL, но на неподвижной сетке её вычисляем только здесь.
            for (MeshIndex node : m.cell_nodes(first + i))
                min_heights_[node] = std::min(min_heights_[node], heights[i]);
            for (const auto& v : parts[i]) {
                volumes_[v.a] += v.volume;
                volumes_[v.b] += v.volume;
                edge_areas_[v.edge] = edge_areas_[v.edge] + v.area;
            }
        }
        first += size;
    }
    parallel_for(m.node_count, [&](MeshIndex i) {
        if (!(volumes_[i] > 0 && std::isfinite(volumes_[i])))
            throw MeshError("median dual: invalid volume at node " + std::to_string(i));
    });
    for (const V& area : edge_areas_)
        if (!finite(area) || !(area.length() > 0))
            throw MeshError("median dual: invalid edge area");
    for (const CvBoundary& piece : boundary_)
        if (!finite(piece.area))
            throw MeshError("median dual: invalid boundary area");

    // У замкнутого объёма векторная сумма площадок равна нулю.
    // Одно внутреннее ребро входит в два узла с противоположными знаками.
    std::vector<V> closure(m.node_count);
    std::vector<double> surface(m.node_count);
    for (MeshIndex e = 0; e < m.edge_count; ++e) {
        const V area = edge_areas_[e];
        closure[m.edge_begin[e]] = closure[m.edge_begin[e]] + area;
        closure[m.edge_end[e]] = closure[m.edge_end[e]] - area;
        const double area_magnitude = area.length();
        surface[m.edge_begin[e]] += area_magnitude;
        surface[m.edge_end[e]] += area_magnitude;
    }
    for (const CvBoundary& piece : boundary_) {
        closure[piece.node] = closure[piece.node] + piece.area;
        surface[piece.node] += piece.area.length();
    }
    parallel_for(m.node_count, [&](MeshIndex i) {
        if (!std::isfinite(surface[i]) || !(closure[i].length() <= 1e-12 * surface[i]))
            throw MeshError("median dual: surface does not close at node " + std::to_string(i));
    });
}
// Даёт доступ к геометрии без копирования массивов MedianDual.
CvView MedianDual::view() const noexcept {
    return {dimension_,      volumes_.size(),    edge_areas_.size(), boundary_.size(),
            volumes_.data(), edge_areas_.data(), boundary_.data(), min_heights_.data()};
}
}
