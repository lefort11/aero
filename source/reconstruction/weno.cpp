#include "stencil_builder.h"
#include <set>

namespace aero {
namespace detail {
namespace {
using NodeSet = std::set<MeshIndex>;
using FaceLists = std::vector<std::vector<MeshIndex>>;

struct LayerFaces {
    FaceLists near; // Грани, у которых все вершины принадлежат первому слою узла.
    FaceLists far;  // Аналогичные грани второго слоя; нужны только WENO5.
};

// Проверяем все вершины грани. Одной общей вершины со слоем недостаточно.
std::vector<MeshIndex> faces_inside_layer(MeshView mesh, const NodeSet& layer, const FaceLists& node_faces) {
    std::set<MeshIndex> candidates;
    for (MeshIndex node : layer) {
        candidates.insert(node_faces[node].begin(), node_faces[node].end());
    }
    std::vector<MeshIndex> faces;
    for (MeshIndex face : candidates) {
        const auto vertices = mesh.face_nodes(face);
        if (std::all_of(vertices.begin(), vertices.end(),
                        [&](MeshIndex node) { return layer.count(node) != 0; })) {
            faces.push_back(face);
        }
    }
    return faces;
}

// Первый слой — все вершины общих ячеек. Второй — их соседи за вычетом центра и первого слоя.
LayerFaces build_layer_faces(MeshView mesh, bool need_far) {
    std::vector<NodeSet> neighbours(mesh.node_count);
    FaceLists node_faces(mesh.node_count);
    for (MeshIndex cell = 0; cell < mesh.cell_count; ++cell) {
        const auto nodes = mesh.cell(cell).node_ids();
        for (MeshIndex node : nodes) {
            for (MeshIndex other : nodes) {
                if (other != node) {
                    neighbours[node].insert(other);
                }
            }
        }
    }
    for (MeshIndex face = 0; face < mesh.face_count; ++face) {
        for (MeshIndex node : mesh.face_nodes(face)) {
            node_faces[node].push_back(face);
        }
    }

    LayerFaces layers;
    layers.near.resize(mesh.node_count);
    if (need_far) {
        layers.far.resize(mesh.node_count);
    }
    parallel_for(mesh.node_count, [&](MeshIndex node) {
        const auto& first = neighbours[node];
        layers.near[node] = faces_inside_layer(mesh, first, node_faces);
        if (!need_far) {
            return;
        }
        NodeSet second;
        for (MeshIndex adjacent : first) {
            for (MeshIndex candidate : neighbours[adjacent]) {
                if (candidate != node && !first.count(candidate)) {
                    second.insert(candidate);
                }
            }
        }
        layers.far[node] = faces_inside_layer(mesh, second, node_faces);
    });
    return layers;
}

// WENO5 берёт разность интерполированных значений в двух пересечениях одного луча.
void build_far_weno_slope(const RayPoint& far, unsigned side, double length, double tolerance,
                          SideSupport& support) {
    const double spacing = far.distance - support.near_point.distance;
    if (far.weights.empty() || !(spacing > tolerance * length)) {
        return;
    }
    const double factor = (side ? 1.0 : -1.0) * (length / spacing);
    add_scaled_row(support.far_slope, far.weights, factor);
    add_scaled_row(support.far_slope, support.near_point.weights, -factor);
    support.far_distance = far.distance;
}
}

void build_weno_stencil(const Mesh& mesh, const ReconstructionOptions& options,
                        const ReconstructionSchemeInfo& scheme, ReconstructionStencilData& data) {
    const ReconstructionGeometry geometry(mesh, options.relative_tolerance, false);
    const auto layers = build_layer_faces(mesh.view(), scheme.layers == 2);
    const auto find_near = [&](MeshIndex edge, unsigned side, MeshIndex node) {
        return geometry.layer_hit(edge, side, layers.near[node]);
    };
    const auto build_far = [&](MeshIndex edge, unsigned side, MeshIndex node, double length,
                               SideSupport& support) {
        const auto hit = geometry.layer_hit(edge, side, layers.far[node]);
        build_far_weno_slope(hit, side, length, options.relative_tolerance, support);
    };
    build_directional_stencil(data, mesh.view(), geometry, options, scheme, find_near, build_far);
}
}
}
