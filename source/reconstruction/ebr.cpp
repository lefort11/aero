#include "stencil_builder.h"

namespace aero {
namespace detail {
namespace {
// EBR5 интерполирует производные из вершин первой опорной грани. У каждой вершины своё расстояние.
void build_far_ebr_slope(const ReconstructionGeometry& geometry, MeshIndex edge, unsigned side, double length,
                         SideSupport& support) {
    SparseRow slope;
    double distance_from_near = 0;
    for (const auto& vertex : support.near_point.weights) {
        if (vertex.second == 0) {
            continue;
        }
        const auto hit = geometry.parallel_star_hit(vertex.first, edge, side);
        if (hit.weights.empty()) {
            return; // Весь дальний слой этой стороны неполон.
        }
        add_scaled_row(slope, ray_derivative(vertex.first, hit, length, side), vertex.second);
        distance_from_near += vertex.second * hit.distance;
    }
    support.far_slope = std::move(slope);
    support.far_distance = support.near_point.distance + distance_from_near;
}
}

void build_ebr_stencil(const Mesh& mesh, const ReconstructionOptions& options,
                       const ReconstructionSchemeInfo& scheme, ReconstructionStencilData& data) {
    const ReconstructionGeometry geometry(mesh, options.relative_tolerance, true);
    const auto find_near = [&](MeshIndex edge, unsigned side, MeshIndex node) {
        return geometry.parallel_star_hit(node, edge, side);
    };
    const auto build_far = [&](MeshIndex edge, unsigned side, MeshIndex, double length,
                               SideSupport& support) {
        build_far_ebr_slope(geometry, edge, side, length, support);
    };
    build_directional_stencil(data, mesh.view(), geometry, options, scheme, find_near, build_far);
}
}
}
