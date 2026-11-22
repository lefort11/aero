#include "storage.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace aero {
namespace turbulence {
namespace {
struct Face {
    Coordinate a, b, c;
}; // Повтор вершины обозначает отрезок или точку в 2D и 1D.
double segment(Coordinate x, Coordinate a, Coordinate b) {
    const auto edge = b - a;
    const double length = dot3(edge, edge);
    const double t = length ? std::max(0., std::min(1., dot3(x - a, edge) / length)) : 0.;
    const auto dx = x - a - edge * t;
    return dot3(dx, dx);
}
double squared_distance(Coordinate p, const Face &f) {
    // Если проекция лежит внутри треугольника, ближайшая точка находится в его плоскости.
    const auto e = f.b - f.a, h = f.c - f.a, x = p - f.a;
    const double ee = dot3(e, e), hh = dot3(h, h), eh = dot3(e, h), xe = dot3(x, e), xh = dot3(x, h);
    const double det = ee * hh - eh * eh;
    if (det > 0.) {
        const double s = (hh * xe - eh * xh) / det, t = (ee * xh - eh * xe) / det;
        if (s >= 0. && t >= 0. && s + t <= 1.) {
            const auto d = x - e * s - h * t;
            return dot3(d, d);
        }
    }
    return std::min(segment(p, f.a, f.b), std::min(segment(p, f.b, f.c), segment(p, f.c, f.a)));
}
// Дерево ограничивающих параллелепипедов строим один раз. При поиске отбрасываем
// целую ветвь, если до её ближайшей точки дальше, чем до уже найденной стенки.
struct WallTree {
    struct Box {
        Coordinate lo, hi;
        size_t first, end, left, right;
    };
    std::vector<Face> faces;
    std::vector<Box> boxes;
    size_t build(size_t first, size_t end) {
        Box box{faces[first].a, faces[first].a, first, end, 0, 0};
        for (size_t j = first; j < end; ++j)
            for (auto p : {faces[j].a, faces[j].b, faces[j].c})
                for (unsigned k = 0; k < 3; ++k) {
                    box.lo[k] = std::min(box.lo[k], p[k]);
                    box.hi[k] = std::max(box.hi[k], p[k]);
                }
        size_t i = boxes.size();
        boxes.push_back(box);
        if (end - first > 8) {
            auto span = box.hi - box.lo;
            const size_t k = static_cast<size_t>(std::max_element(span.begin(), span.end()) - span.begin());
            const size_t mid = first + (end - first) / 2;
            std::nth_element(faces.begin() + first, faces.begin() + mid, faces.begin() + end,
                             [&](const Face &a, const Face &b) {
                                 return a.a[k] + a.b[k] + a.c[k] < b.a[k] + b.b[k] + b.c[k];
                             });
            const auto left = build(first, mid), right = build(mid, end);
            boxes[i].left = left;
            boxes[i].right = right;
        }
        return i;
    }
    double lower(size_t i, Coordinate p) const {
        double d = 0.;
        for (unsigned k = 0; k < 3; ++k) {
            const double x = std::max(0., std::max(boxes[i].lo[k] - p[k], p[k] - boxes[i].hi[k]));
            d += x * x;
        }
        return d;
    }
    double nearest(size_t i, Coordinate p, double best) const {
        const auto &b = boxes[i];
        if (!b.left) {
            for (size_t j = b.first; j < b.end; ++j)
                best = std::min(best, squared_distance(p, faces[j]));
        } else {
            size_t a = b.left, c = b.right;
            double near_a = lower(a,p), near_c = lower(c,p);
            if (near_c < near_a) {
                std::swap(a,c);
                std::swap(near_a,near_c);
            }
            if (near_a < best) best = nearest(a,p,best);
            if (near_c < best) best = nearest(c,p,best);
        }
        return best;
    }
};
}
std::vector<double> wall_distances(const MeshPartition &part, const PeriodicPartition *seam,
                                   const BoundaryPatches &bc, const std::vector<PeriodicPair> &pairs,
                                   const Processes &world, MeshIndex nodes) {
    const auto mesh = part.mesh.view();
    WallTree tree;
    std::vector<unsigned char> wall(mesh.node_count, 0);
    std::string packet;
    world.local([&] {
        for (MeshIndex j = 0; j < mesh.boundary_face_count; ++j) {
            const auto patch = mesh.boundary_patch[j];
            if (patch >= part.physical_patches || bc.by_patch[patch].type() != BoundaryType::Wall)
                continue;
            const auto v = mesh.face_nodes(mesh.boundary_faces[j]);
            for (auto i : v)
                wall[i] = 1;
            if (part.node_owners[v[0]] != world.rank)
                continue;
            auto emit = [&](Coordinate a, Coordinate b, Coordinate c) {
                Face f{a, b, c};
                packet.append(reinterpret_cast<const char *>(&f), sizeof(f));
            };
            if (v.size <= 3)
                emit(mesh.coordinate(v[0]), mesh.coordinate(v[v.size / 2]), mesh.coordinate(v[v.size - 1]));
            else {
                Coordinate center{};
                for (auto i : v)
                    center = center + mesh.coordinate(i);
                center = center * (1. / static_cast<double>(v.size));
                for (MeshIndex k = 0; k < v.size; ++k)
                    emit(center, mesh.coordinate(v[k]), mesh.coordinate(v[(k + 1) % v.size]));
            }
        }
    });
    // Координаты содержат нулевые байты. Передаём длину и содержимое как двоичные
    // сообщения; обмен выполняется один раз, до построения дерева стенок.
    if (world.size > 1) {
        Processes::Messages outgoing;
        world.local([&] {
            outgoing.resize(world.size);
            outgoing[0].swap(packet);
        });
        const auto packets = world.exchange(outgoing);
        world.local([&] {
            if (!world.rank)
                for (const auto &p : packets)
                    packet += p;
        });
        world.broadcast(packet, 0);
    }
    std::vector<double> result(nodes, 0.);
    world.local([&] {
        if (packet.empty())
            return; // У полностью периодической задачи нет пристенной деструкции.
        if (packet.size() % sizeof(Face))
            throw MeshError("SA: incomplete wall geometry");
        tree.faces.resize(packet.size() / sizeof(Face));
        std::memcpy(tree.faces.data(), packet.data(), packet.size());
        tree.build(0, tree.faces.size());
        std::vector<Coordinate> shifts(1);
        std::vector<Coordinate> directions;
        for (const auto &pair : pairs) {
            bool seen = false;
            for (auto t : directions)
                if (coordinate_length(t - pair.translation) == 0. ||
                    coordinate_length(t + pair.translation) == 0.)
                    seen = true;
            if (seen)
                continue;
            directions.push_back(pair.translation);
            const size_t n = shifts.size();
            for (size_t k = 0; k < n; ++k) {
                const auto s = shifts[k];
                shifts.push_back(s + pair.translation);
                shifts.push_back(s - pair.translation);
            }
        }
        // Без периодической склейки узел записывает расстояние сразу на своё место.
        std::vector<double> periodic_distance(seam ? mesh.node_count : 0);
        auto& inv = seam ? periodic_distance : result;
        parallel_for(mesh.node_count, [&](MeshIndex i) {
            if (wall[i]) {
                inv[i] = -1.;
                return;
            }
            double best = std::numeric_limits<double>::infinity();
            for (auto s : shifts)
                best = tree.nearest(0, mesh.coordinate(i) + s, best);
            inv[i] = best == 0. ? -1. : 1. / std::sqrt(best);
        });
        if (!seam) return;
        for (MeshIndex i = 0; i < mesh.node_count; ++i) {
            auto &d = result[seam->node_dof[i]];
            if (d >= 0.)
                d = inv[i] < 0. ? -1. : std::max(d, inv[i]);
        }
    });
    return result;
}
}
}
