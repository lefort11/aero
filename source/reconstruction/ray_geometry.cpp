#include "reconstruction.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <initializer_list>

namespace aero {
detail::RayPoint::RayPoint() : distance((std::numeric_limits<double>::infinity)()), outward(false) {}
namespace {
Coordinate operator*(Coordinate a, double t) {
    return {{a[0] * t, a[1] * t, a[2] * t}};
}
Coordinate operator/(Coordinate a, double t) {
    return a * (1.0 / t);
}
double dot(Coordinate a, Coordinate b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Coordinate cross(Coordinate a, Coordinate b) {
    return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
}
typedef std::array<MeshIndex, 3> Triangle;
struct Surface {
    Coordinate normal;
    std::vector<Triangle> triangles;
    // Для неплоской грани третьей вершиной каждого треугольника служит
    // общий центр. Его значение интерполируем поровну по узлам контура.
    std::vector<MeshIndex> center_nodes;
    Coordinate center;
};

double turn(Coordinate a, Coordinate b, Coordinate c, Coordinate normal) {
    return dot(cross(b - a, c - a), normal);
}

// Переносим начало в вершину грани и приводим координаты к её масштабу.
std::vector<Coordinate> normalized_polygon(MeshView mesh, const std::vector<MeshIndex>& ids) {
    if (ids.size() < 3) {
        throw ReconstructionError("Polygon has fewer than three nodes");
    }
    const Coordinate origin = mesh.coordinate(ids[0]);
    double scale = 0;
    for (MeshIndex id : ids) {
        scale = (std::max)(scale, coordinate_length(mesh.coordinate(id) - origin));
    }
    if (!(scale > 0) || !std::isfinite(scale)) {
        throw ReconstructionError("Degenerate polygon scale");
    }
    std::vector<Coordinate> p;
    p.reserve(ids.size());
    for (MeshIndex id : ids) {
        p.push_back((mesh.coordinate(id) - origin) / scale);
    }
    return p;
}

// Самопересечение или касание несмежных рёбер делает интерполяцию по грани неоднозначной.
void validate_polygon(const std::vector<Coordinate>& p, Coordinate normal, double eps) {
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (coordinate_length(p[(i + 1) % n] - p[i]) <= eps) {
            throw ReconstructionError("Zero or unresolved polygon edge");
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            if (j == i + 1 || (i == 0 && j == n - 1)) {
                continue;
            }
            const Coordinate a = p[i], b = p[(i + 1) % n], c = p[j], d = p[(j + 1) % n];
            const double ac = turn(a, b, c, normal), ad = turn(a, b, d, normal);
            const double ca = turn(c, d, a, normal), cb = turn(c, d, b, normal);
            if (((ac > eps && ad < -eps) || (ac < -eps && ad > eps)) &&
                ((ca > eps && cb < -eps) || (ca < -eps && cb > eps))) {
                throw ReconstructionError("Self-intersecting polygon");
            }
            const Coordinate ends[4] = {a, b, c, d};
            for (unsigned k = 0; k < 4; ++k) {
                const Coordinate q = ends[k], r = ends[k < 2 ? 2 : 0], t = ends[k < 2 ? 3 : 1];
                if (std::abs(turn(r, t, q, normal)) <= eps && dot(q - r, q - t) <= eps * eps) {
                    throw ReconstructionError("Touching or overlapping polygon edges");
                }
            }
        }
    }
}

// Отрезаем выпуклые треугольники по порядку вершин, пока в контуре не останутся три точки.
std::vector<Triangle> clip_ears(const std::vector<Coordinate>& p, const std::vector<MeshIndex>& ids, Coordinate normal,
                                double eps) {
    const auto n = p.size();
    std::vector<Triangle> triangles;
    triangles.reserve(n - 2);
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < n; ++i) {
        active.push_back(i);
    }
    while (active.size() > 3) {
        bool clipped = false;
        for (std::size_t j = 0; j < active.size(); ++j) {
            const std::size_t a = active[(j + active.size() - 1) % active.size()], b = active[j],
                              c = active[(j + 1) % active.size()];
            if (turn(p[a], p[b], p[c], normal) <= eps) {
                continue;
            }
            bool contains = false;
            for (std::size_t k : active) {
                if (k != a && k != b && k != c) {
                    if (turn(p[a], p[b], p[k], normal) >= -eps && turn(p[b], p[c], p[k], normal) >= -eps &&
                        turn(p[c], p[a], p[k], normal) >= -eps) {
                        contains = true;
                        break;
                    }
                }
            }
            if (contains) {
                continue;
            }
            triangles.push_back(Triangle{{ids[a], ids[b], ids[c]}});
            active.erase(active.begin() + static_cast<std::ptrdiff_t>(j));
            clipped = true;
            break;
        }
        if (!clipped) {
            throw ReconstructionError("Cannot triangulate polygon at selected tolerance");
        }
    }
    if (turn(p[active[0]], p[active[1]], p[active[2]], normal) <= eps) {
        throw ReconstructionError("Degenerate final polygon triangle");
    }
    triangles.push_back(Triangle{{ids[active[0]], ids[active[1]], ids[active[2]]}});
    return triangles;
}

// Треугольники нужны только для пересечения луча с гранью. Исходная связность сетки сохраняется.
Surface triangulate(MeshView mesh, const std::vector<MeshIndex>& ids, double eps) {
    const auto points = normalized_polygon(mesh, ids);
    Coordinate area{};
    for (std::size_t i = 0; i < points.size(); ++i) {
        area = area + cross(points[i], points[(i + 1) % points.size()]);
    }
    const double magnitude = coordinate_length(area);
    if (!(magnitude > eps)) {
        throw ReconstructionError("Degenerate or self-intersecting polygon");
    }
    const Coordinate normal = area / magnitude;
    validate_polygon(points, normal, eps);
    Surface surface = {normal, {}, {}, {}};
    for (Coordinate point : points) {
        if (std::abs(dot(point, normal)) <= eps)
            continue;
        if (mesh.dimension == 2)
            throw ReconstructionError("EBR requires planar 2D cells");

        // Центр совпадает с центром разбиения в MedianDual. Суммируем
        // смещения относительно одного узла для сеток с большим сдвигом.
        const Coordinate origin = mesh.coordinate(ids.front());
        Coordinate sum{};
        for (MeshIndex node : ids)
            sum = sum + (mesh.coordinate(node) - origin);
        surface.center = origin + sum / static_cast<double>(ids.size());
        surface.center_nodes = ids;
        for (std::size_t j = 0; j < ids.size(); ++j)
            surface.triangles.push_back({{ids[j], ids[(j + 1) % ids.size()], mesh.node_count}});
        return surface;
    }
    surface.triangles = clip_ears(points, ids, normal, eps);
    return surface;
}

std::vector<MeshIndex> cell_cycle(MeshCell cell) {
    std::map<MeshIndex, MeshIndex> next;
    for (MeshIndex i = 0; i < cell.face_count(); ++i) {
        const MeshFace face = cell.face(i);
        const bool forward = cell.face_orientation(i) > 0;
        next[face.node_id(forward ? 0 : 1)] = face.node_id(forward ? 1 : 0);
    }
    std::vector<MeshIndex> result;
    MeshIndex id = next.begin()->first;
    for (std::size_t i = 0; i < next.size(); ++i) {
        result.push_back(id);
        id = next.at(id);
    }
    return result;
}

// При обходе ячеек звезды считаем, сколько раз встретилась грань. Один раз означает внешнюю границу.
struct PatchFace {
    MeshIndex face;  // Номер грани.
    MeshIndex cell;  // Смежная ячейка, задающая нормаль в 2D.
    int orientation; // Ориентация грани относительно ячейки.
    bool boundary;   // Внешняя граница сетки. Через неё нельзя продолжать поиск опоры.
};

// Ищет одну опорную точку для заданного начала и направления луча.
class RaySearch {
public:
    RaySearch(Coordinate origin, Coordinate direction, double length, double eps)
        : origin_(origin), direction_(direction), length_(length), eps_(eps) { best_.distance = 0; }

    void begin_face(bool boundary) { boundary_face_ = boundary; }
    // Для невыпуклой области ближняя физическая граница может заслонить дальнюю грань слоя.
    bool needs_clipping() const {
        return best_.distance > boundary_distance_ + eps_ * length_ ||
               (!boundary_exit_ && best_.distance >= boundary_distance_ - eps_ * length_);
    }
    void clip_at_boundary() {
        clipped_ = true;
        best_.weights.clear();
        best_.distance = 0;
        best_.outward = false;
    }

    // В 1D опорная грань состоит из одной вершины с единичным весом.
    void test_point(MeshView mesh, MeshIndex node) {
        const Coordinate offset = mesh.coordinate(node) - origin_;
        const double distance = dot(offset, direction_);
        if (coordinate_length(offset - direction_ * distance) <= eps_ * (std::max)(length_, coordinate_length(offset))) {
            consider(distance, true, {{node, 1}});
        }
    }

    // В плоскости грани параметр u задаёт линейную интерполяцию между концами отрезка.
    void test_segment(MeshView mesh, MeshFace face, Coordinate normal, int orientation) {
        const MeshIndex a = face.node_id(0), b = face.node_id(1);
        const Coordinate edge = mesh.coordinate(b) - mesh.coordinate(a), from = mesh.coordinate(a) - origin_;
        const double det = dot(cross(direction_, edge), normal);
        if (std::abs(det) <= eps_ * coordinate_length(edge)) {
            return;
        }
        // На вытянутой ячейке малый вес длинного отрезка определяет положение точки.
        // Вычисляем его от ближайшего конца напрямую: разность 1-u теряет значащие цифры.
        const Coordinate to = mesh.coordinate(b) - origin_;
        const double weight_a = dot(cross(direction_, to), normal) / det;
        const double weight_b = dot(cross(from, direction_), normal) / det;
        if (weight_a < -eps_ || weight_b < -eps_) return;
        const bool near_a = weight_b < weight_a;
        const double small = (std::max)(0.0, near_a ? weight_b : weight_a);
        const Coordinate point = near_a ? from + edge * small : to - edge * small;
        const double distance = dot(point, direction_);
        // Проверяем уже интерполированную точку относительно длины от начала луча.
        // Если ближайшая вершина заметно в стороне, продолжаем поиск по соседним граням.
        if (coordinate_length(point - direction_ * distance) > eps_ * std::abs(distance)) return;
        const Coordinate outward = cross(edge, normal) * static_cast<double>(orientation);
        consider(distance, dot(direction_, outward) > 0,
                 {{a, near_a ? 1. - small : small}, {b, near_a ? small : 1. - small}});
    }

    // В 3D веса пересечения — барицентрические координаты на одном треугольнике грани.
    void test_triangle(MeshView mesh, const Triangle& tri, const Surface& surface, int orientation) {
        const Coordinate a = mesh.coordinate(tri[0]) - origin_;
        const Coordinate b = mesh.coordinate(tri[1]) - origin_, c = (surface.center_nodes.empty() ? mesh.coordinate(tri[2]) : surface.center) - origin_;
        // Локальный масштаб сохраняет смысл допуска для больших и маленьких ячеек.
        const double scale = (std::max)(coordinate_length(b - a), coordinate_length(c - a));
        const Coordinate e1 = (b - a) / scale, e2 = (c - a) / scale, r = a / scale;
        const Coordinate p = cross(direction_, e2);
        const double det = dot(e1, p);
        if (std::abs(det) <= eps_ * coordinate_length(e1) * coordinate_length(e2)) {
            return;
        }
        double u = dot(r * (-1), p) / det;
        const Coordinate q = cross(r * (-1), e1);
        double v = dot(direction_, q) / det;
        const double distance = dot(e2, q) / det * scale;
        if (u < -eps_ || v < -eps_ || u + v > 1 + eps_) {
            return;
        }
        u = (std::max)(0.0, u);
        v = (std::max)(0.0, v);
        if (u + v > 1) {
            const double sum = u + v;
            u /= sum;
            v /= sum;
        }
        // Знак определителя задаёт направление пересечения именно этого треугольника.
        if (consider(distance, det * orientation < 0,
                     {{tri[0], 1 - u - v}, {tri[1], u}, {tri[2], v}}) && !surface.center_nodes.empty()) {
            // Заменяем вес вспомогательного центра вкладами реальных узлов.
            best_.weights.erase(tri[2]);
            const double share = v / static_cast<double>(surface.center_nodes.size());
            for (MeshIndex node : surface.center_nodes)
                best_.weights[node] += share;
        }
    }

    detail::RayPoint finish() {
        return std::move(best_);
    }

private:
    // Выбираем наиболее удалённое пересечение этого слоя. При совпадении предпочитаем выход.
    bool consider(double distance, bool outward,
                  std::initializer_list<detail::SparseRow::value_type> weights) {
        const double tolerance = eps_ * length_;
        if (!(distance > tolerance) || !std::isfinite(distance)) {
            return false;
        }
        if (boundary_face_ && !clipped_) {
            if (distance < boundary_distance_ - tolerance) {
                boundary_distance_ = distance;
                boundary_exit_ = outward;
            } else if (std::abs(distance - boundary_distance_) <= tolerance)
                boundary_exit_ = boundary_exit_ || outward;
        }
        if (clipped_ && (distance > boundary_distance_ + tolerance ||
                         (!boundary_exit_ && distance >= boundary_distance_ - tolerance)))
            return false;
        const bool better = distance > best_.distance + tolerance;
        const bool tied_exit = std::abs(distance - best_.distance) <= tolerance && outward && !best_.outward;
        if (better || tied_exit) {
            best_.distance = distance;
            best_.outward = outward;
            best_.weights = weights;
            return true;
        }
        return false;
    }
    bool boundary_face_ = false, boundary_exit_ = false, clipped_ = false;
    double boundary_distance_ = std::numeric_limits<double>::infinity();
    Coordinate origin_;            // Координаты начала луча.
    Coordinate direction_;         // Единичный вектор продолжения ребра.
    double length_;         // Длина ребра, задающая масштаб допуска.
    double eps_;            // Относительный допуск геометрических проверок.
    detail::RayPoint best_; // Расстояние и веса лучшего пересечения.
};

// Переиспользуем построенную геометрию граней для всех лучей этого ребра.
template <class FaceAt>
detail::RayPoint outer_point(MeshView mesh, MeshIndex node, Coordinate direction, double length,
                             std::size_t face_count, const FaceAt& face_at, const std::vector<Surface>& faces,
                             const std::vector<Coordinate>& cell_normals, double eps) {
    RaySearch search(mesh.coordinate(node), direction, length, eps);
    const auto visit_faces = [&] {
        for (std::size_t i = 0; i < face_count; ++i) {
            const auto& entry = face_at(i);
            search.begin_face(entry.boundary);
            const auto face = mesh.face(entry.face);
            if (mesh.dimension == 1) {
                search.test_point(mesh, face.node_id(0));
            } else if (mesh.dimension == 2) {
                search.test_segment(mesh, face, cell_normals[entry.cell], entry.orientation);
            } else {
                const auto& surface = faces[entry.face];
                for (const auto& triangle : surface.triangles) {
                    search.test_triangle(mesh, triangle, surface, entry.orientation);
                }
            }
        }
    };
    visit_faces();
    if (search.needs_clipping()) {
        search.clip_at_boundary();
        visit_faces();
    }
    return search.finish();
}

} // namespace

struct detail::ReconstructionGeometry::Impl {
    MeshView m;                 // Заимствованные координаты и связность исходной сетки.
    double eps;                 // Относительный допуск для пересечений и проверок геометрии.
    std::vector<Surface> faces; // Разбиение 3D-граней на треугольники для поиска пересечений.
    std::vector<Coordinate> normals;   // Ориентированные нормали 2D-ячеек.
    std::vector<PatchFace> owners;               // Смежная ячейка и ориентация по номеру опорной грани.

    Impl(const Mesh& mesh, double tolerance)
        : m(mesh.view()), eps(tolerance), faces(m.dimension == 3 ? m.face_count : 0),
          normals(m.dimension == 2 ? m.cell_count : 0), owners(m.face_count) {
        if (!std::isfinite(eps) || eps <= 0 || eps >= 0.01) {
            throw ReconstructionError("Invalid geometry tolerance");
        }
        if (m.dimension == 3) {
            parallel_for(m.face_count, [&](MeshIndex face) {
                const auto nodes = m.face_nodes(face);
                faces[face] = triangulate(m, std::vector<MeshIndex>(nodes.begin(), nodes.end()), eps);
            });
        }
        for (MeshIndex index = 0; index < m.cell_count; ++index) {
            const auto cell = m.cell(index);
            for (MeshIndex k = 0; k < cell.face_count(); ++k) {
                const auto face = cell.face_id(k);
                owners[face] = PatchFace{face, index, cell.face_orientation(k), false};
            }
            if (m.dimension == 2) {
                normals[index] = triangulate(m, cell_cycle(cell), eps).normal;
            }
            if (m.dimension == 3) {
                check_cell_volume(cell);
            }
        }
        for (MeshIndex i = 0; i < m.boundary_face_count; ++i)
            owners[m.boundary_faces[i]].boundary = true;
        if (m.dimension == 2 && m.cell_count) {
            check_common_plane();
        }
    }

    // Сумма ориентированных тетраэдров равна шести объёмам нормированной ячейки.
    void check_cell_volume(MeshCell cell) const {
        const Coordinate origin = m.coordinate(cell.node_id(0));
        double scale = 0;
        for (auto node : cell.node_ids()) {
            scale = (std::max)(scale, coordinate_length(m.coordinate(node) - origin));
        }
        if (!(scale > 0) || !std::isfinite(scale)) {
            throw ReconstructionError("Degenerate cell scale");
        }
        double volume6 = 0;
        for (MeshIndex k = 0; k < cell.face_count(); ++k) {
            const Surface& surface = faces[cell.face_id(k)];
            for (const Triangle& t : surface.triangles) {
                const Coordinate third = surface.center_nodes.empty() ? m.coordinate(t[2]) : surface.center;
                volume6 += cell.face_orientation(k) *
                           dot((m.coordinate(t[0]) - origin) / scale,
                               cross((m.coordinate(t[1]) - origin) / scale, (third - origin) / scale));
            }
        }
        if (!(volume6 > eps)) {
            throw ReconstructionError("Inverted or degenerate 3D cell");
        }
    }

    // У всех ячеек двумерной сетки должна быть общая плоскость для продолжения ребра.
    void check_common_plane() const {
        const Coordinate origin = m.coordinate(m.cell(0).node_id(0));
        for (MeshIndex node = 0; node < m.node_count; ++node) {
            const Coordinate offset = m.coordinate(node) - origin;
            if (std::abs(dot(offset, normals[0])) > eps * coordinate_length(offset)) {
                throw ReconstructionError("2D EBR requires a planar mesh");
            }
        }
    }

    Coordinate edge(MeshIndex e) const { return m.coordinate(m.edge_end[e]) - m.coordinate(m.edge_begin[e]); }

    // Одна параметризация для EBR и WENO. Грани слоя читаем по индексам без копирования.
    template <class FaceAt>
    RayPoint intersect(MeshIndex node, MeshIndex edge_id, unsigned side, double length,
                       std::size_t count, const FaceAt& face_at) const {
        const Coordinate direction = edge(edge_id) * ((side ? 1.0 : -1.0) / length);
        return outer_point(m, node, direction, length, count, face_at, faces, normals, eps);
    }
};
detail::ReconstructionGeometry::ReconstructionGeometry(const Mesh& mesh, double tolerance)
    : impl_(new Impl(mesh, tolerance)) {}
detail::ReconstructionGeometry::~ReconstructionGeometry() {}
double detail::ReconstructionGeometry::edge_length(MeshIndex e) const {
    const double h = coordinate_length(impl_->edge(e));
    if (!(h > 0) || !std::isfinite(h)) {
        throw ReconstructionError("Zero or non-finite edge length");
    }
    return h;
}
detail::RayPoint detail::ReconstructionGeometry::layer_hit(MeshIndex e, unsigned side,
                                                           const std::vector<MeshIndex>& ids) const {
    const Impl& g = *impl_;
    const double h = edge_length(e);
    const MeshIndex node = side ? g.m.edge_end[e] : g.m.edge_begin[e];
    return g.intersect(node, e, side, h, ids.size(),
                       [&](std::size_t i) -> const PatchFace& { return g.owners[ids[i]]; });
}
void detail::ReconstructionGeometry::append_row(ReconstructionStencilData& data, const SparseRow& row,
                                                MeshIndex e, unsigned side) const {
    const Impl& g = *impl_;
    const double h = edge_length(e);
    const MeshIndex center = side ? g.m.edge_end[e] : g.m.edge_begin[e];
    double sum = 0;
    Coordinate moment{};
    for (const auto& w : row) {
        if (!std::isfinite(w.second)) {
            throw ReconstructionError("Non-finite reconstruction weight");
        }
        sum += w.second;
        moment = moment + ((g.m.coordinate(w.first) - g.m.coordinate(center)) / h) * w.second;
        if (w.second != 0) {
            data.nodes.push_back(w.first);
            data.weights.push_back(w.second);
        }
    }
    if (std::abs(sum) > 100 * g.eps || coordinate_length(moment - g.edge(e) / h) > 100 * g.eps) {
        throw ReconstructionError("EBR stencil failed constant/linear consistency check");
    }
    data.offsets.push_back(static_cast<ReconstructionIndex>(data.nodes.size()));
    data.row_owner.push_back(center);
}
void detail::add_scaled_row(SparseRow& destination, const SparseRow& source, double factor) {
    for (const auto& term : source) {
        destination[term.first] += factor * term.second;
    }
}

// На обеих сторонах ребра производная направлена от начала ребра к концу.
detail::SparseRow detail::ray_derivative(MeshIndex node, const RayPoint& hit, double length, unsigned side) {
    SparseRow row;
    const double factor = (side ? 1.0 : -1.0) * (length / hit.distance);
    row[node] = -factor;
    add_scaled_row(row, hit.weights, factor);
    return row;
}
} // namespace aero
