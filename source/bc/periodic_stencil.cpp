#include "periodic.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <set>

namespace aero {
namespace {
using Id = MeshIndex;
using Ids = std::vector<Id>;

// Для шаблонов строим временное покрытие из двух слоёв. Вершину задаёт пара из расчётного номера и образа.
// Сохраняем исходные узлы в начале массива, чтобы ориентация оставшихся рёбер совпала с исходной.
struct Cover {
    MeshView m;
    const Ids& dof;
    double tol;
    MeshData data;
    Ids field;
    std::vector<Ids> images;
    std::map<Ids, Id> faces;
    std::set<std::pair<Id, Id>> cells; // Исходная ячейка и пространственный образ её первой вершины.
    std::vector<Ids> incident, members;
    Cover(const Mesh& mesh, const Ids& node_dof, Id count, double tolerance)
        : m(mesh.view()), dof(node_dof), tol(tolerance), data(mesh.data()), field(dof), images(count),
          incident(m.node_count), members(count) {
        for (Id n = 0; n < m.node_count; ++n) {
            members[dof[n]].push_back(n);
            images[dof[n]].push_back(n);
        }
        for (Id f = 0; f < m.face_count; ++f) {
            MeshIndexSpan ns = m.face_nodes(f);
            Ids key(ns.begin(), ns.end());
            std::sort(key.begin(), key.end());
            faces[key] = f;
        }
        for (Id c = 0; c < m.cell_count; ++c) {
            cells.emplace(c, m.cell(c).node_id(0));
            for (Id n : m.cell_nodes(c))
                incident[n].push_back(c);
        }
    }
    Id image(Id n, Coordinate shift) {
        const Coordinate target = m.coordinate(n) + shift;
        auto& candidates = images[dof[n]];
        for (Id i : candidates)
            if (std::hypot(std::hypot(data.x[i] - target[0], data.y[i] - target[1]), data.z[i] - target[2]) <=
                tol)
                return i;
        Id id = data.x.size();
        data.x.push_back(target[0]);
        data.y.push_back(target[1]);
        data.z.push_back(target[2]);
        candidates.push_back(id);
        field.push_back(dof[n]);
        return id;
    }
    void add_cell(Id c, Coordinate shift) {
        MeshCell cell = m.cell(c);
        if (!cells.emplace(c, image(cell.node_id(0), shift)).second)
            return;
        for (Id k = 0; k < cell.face_count(); ++k) {
            Ids cycle;
            const MeshFace face = cell.face(k);
            for (Id n : face.node_ids())
                cycle.push_back(image(n, shift));
            Ids key = cycle;
            std::sort(key.begin(), key.end());
            auto insertion = faces.emplace(key, data.face_nodes.offsets.size() - 1);
            const Id f = insertion.first->second;
            int sign = cell.face_orientation(k);
            if (insertion.second) {
                data.face_nodes.indices.insert(data.face_nodes.indices.end(), cycle.begin(), cycle.end());
                data.face_nodes.offsets.push_back(data.face_nodes.indices.size());
            } else if (cycle.size() > 1) {
                auto first =
                    data.face_nodes.indices.begin() + static_cast<std::ptrdiff_t>(data.face_nodes.offsets[f]);
                if (cycle.size() == 2) {
                    if (*first != cycle[0])
                        sign = -sign;
                } else {
                    const size_t pos =
                        static_cast<size_t>(std::find(first, first + cycle.size(), cycle[0]) - first);
                    if (first[(pos + 1) % cycle.size()] != cycle[1])
                        sign = -sign;
                }
            }
            data.cell_faces.indices.push_back(f);
            data.cell_face_orientation.push_back(static_cast<std::int8_t>(sign));
        }
        data.cell_faces.offsets.push_back(data.cell_faces.indices.size());
    }
    void extend(unsigned rings) {
        Id begin = 0, end = m.node_count;
        for (unsigned r = 0; r < rings; ++r) {
            for (Id i = begin; i < end; ++i) {
                for (Id n : members[field[i]]) {
                    if (i == n)
                        continue;
                    Coordinate shift = {{data.x[i] - m.x[n], data.y[i] - m.y[n], data.z[i] - m.z[n]}};
                    for (Id c : incident[n])
                        add_cell(c, shift);
                }
            }
            begin = end;
            end = data.x.size();
        }
        Ids count(data.face_nodes.offsets.size() - 1, 0);
        for (Id f : data.cell_faces.indices)
            ++count[f];
        data.boundary_faces.clear();
        for (Id f = 0; f < count.size(); ++f)
            if (count[f] == 1)
                data.boundary_faces.push_back(f);
        data.patch_names = {"cover"};
        data.boundary_patch.assign(data.boundary_faces.size(), 0);
    }
};
using RI = ReconstructionIndex;
void copy_row(RI row, const Ids& field, const std::vector<RI>& offsets, const std::vector<RI>& nodes,
              const std::vector<double>& weights, std::vector<RI>& dst_offsets, std::vector<RI>& dst_nodes,
              std::vector<double>& dst_weights) {
    for (RI k = offsets[row]; k < offsets[row + 1]; ++k) {
        dst_nodes.push_back(field[nodes[k]]);
        dst_weights.push_back(weights[k]);
    }
    dst_offsets.push_back(dst_nodes.size());
}
}

Mesh periodic_stencil_cover(const Mesh& mesh, const Ids& dof, Id count, double tolerance, unsigned rings,
                            Ids& field) {
    Cover cover(mesh, dof, count, tolerance);
    cover.extend(rings);
    field = std::move(cover.field);
    return Mesh(std::move(cover.data));
}
ReconstructionStencil PeriodicDomain::prepare(const ReconstructionOptions& options) const {
    Ids field;
    Mesh lifted = periodic_stencil_cover(mesh_, node_dof_, volumes_.size(), tolerance_,
                                         reconstruction_scheme(options.scheme).layers, field);
    ReconstructionOptions local = options;
    local.require_complete_stencil =
        false; // Крайние строки вспомогательной сетки после подготовки отбрасываются.
    ReconstructionStencil all = prepare_reconstruction(lifted, local), result;
    const auto& src = all.data();
    auto& dst = result.data_;
    dst.scheme = options.scheme;
    dst.mp5_alpha = src.mp5_alpha;
    dst.mpweno_ad_corr = src.mpweno_ad_corr;
    dst.node_count = volumes_.size();
    dst.edge_count = edges_.size();
    const MeshView m = mesh_.view(), h = lifted.view();
    std::map<std::pair<Id, Id>, Id> index;
    for (Id e = 0; e < h.edge_count; ++e)
        index.emplace(std::make_pair(h.edge_begin[e], h.edge_end[e]), e);
    for (const PeriodicEdge& edge : edges_) {
        const Id e = index.at({m.edge_begin[edge.original], m.edge_end[edge.original]});
        for (RI side = 0; side < 2; ++side) {
            RI row = 2 * e + side;
            if (options.require_complete_stencil && !src.complete[row])
                throw ReconstructionError("Incomplete periodic-domain stencil on original edge " +
                                          std::to_string(edge.original));
            copy_row(row, field, src.offsets, src.nodes, src.weights, dst.offsets, dst.nodes, dst.weights);
            dst.row_owner.push_back(field[src.row_owner[row]]);
            dst.complete.push_back(src.complete[row]);
            dst.outer_distance.push_back(src.outer_distance[row]);
            dst.far_distance.push_back(src.far_distance[row]);
            dst.side_scheme.push_back(src.side_scheme[row]);
            if (!src.weno_order.empty()) {
                dst.weno_order.push_back(src.weno_order[row]);
                dst.weno_epsilon_scale.push_back(src.weno_epsilon_scale[row]);
                for (RI k = 0; k < 4; ++k)
                    copy_row(4 * row + k, field, src.slope_offsets, src.slope_nodes, src.slope_weights,
                             dst.slope_offsets, dst.slope_nodes, dst.slope_weights);
            }
        }
    }
    return result;
}
}
