#ifndef AERO_STENCIL_BUILDER_H
#define AERO_STENCIL_BUILDER_H
#include "parallel_stencil.h"

namespace aero {
namespace detail {
// Все производные умножены на длину ребра и ориентированы от начала ребра к концу.
struct SideSupport {
    RayPoint near_point;     // Пересечение продолжения ребра с первым слоем.
    SparseRow near_slope;    // Производная между концом ребра и ближней точкой.
    SparseRow far_slope;     // Дальняя производная; пустая строка означает, что её построить не удалось.
    double far_distance = 0; // Расстояние для диагностики, в единицах координат сетки.
};

// Выбирает доступный порядок отдельно слева и справа и записывает обе строки в шаблон.
void append_stencil_sides(ReconstructionStencilData& rows, const ReconstructionGeometry& geometry,
                          const ReconstructionOptions& options, const ReconstructionSchemeInfo& scheme,
                          MeshIndex edge, double length, const SparseRow& central,
                          const SideSupport (&support)[2]);

// Общий ход построения. Семейство схем задаёт только поиск ближней точки и дальней производной.
template <class FindNear, class BuildFar>
void build_directional_stencil(ReconstructionStencilData& data, MeshView mesh,
                               const ReconstructionGeometry& geometry, const ReconstructionOptions& options,
                               const ReconstructionSchemeInfo& scheme, const FindNear& find_near,
                               const BuildFar& build_far) {
    build_edge_rows(data, [&](MeshIndex edge, ReconstructionStencilData& rows) {
        const MeshIndex endpoints[] = {mesh.edge_begin[edge], mesh.edge_end[edge]};
        const double length = geometry.edge_length(edge);
        const SparseRow central = {{endpoints[0], -1}, {endpoints[1], 1}}; // Q_j − Q_i.
        SideSupport support[2];
        for (unsigned side = 0; side < 2; ++side) {
            auto& s = support[side];
            s.near_point = find_near(edge, side, endpoints[side]);
            if (!s.near_point.weights.empty()) {
                s.near_slope = ray_derivative(endpoints[side], s.near_point, length, side);
            }
        }
        // Формула пятого порядка использует также ближнюю производную с противоположной стороны.
        if (scheme.layers == 2 && !support[0].near_slope.empty() && !support[1].near_slope.empty()) {
            for (unsigned side = 0; side < 2; ++side) {
                build_far(edge, side, endpoints[side], length, support[side]);
            }
        }
        append_stencil_sides(rows, geometry, options, scheme, edge, length, central, support);
    });
}

// Ближние и дальние точки EBR строятся по границам звёзд узлов.
void build_ebr_stencil(const Mesh& mesh, const ReconstructionOptions& options,
                       const ReconstructionSchemeInfo& scheme, ReconstructionStencilData& data);
// Для WENO опорные точки лежат на гранях первого и второго слоёв соседей.
void build_weno_stencil(const Mesh& mesh, const ReconstructionOptions& options,
                        const ReconstructionSchemeInfo& scheme, ReconstructionStencilData& data);
}
}
#endif
