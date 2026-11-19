#include "stencil_builder.h"
#include <cmath>

namespace aero {
namespace detail {
namespace {
// Весовая строка удвоенного приращения. Для WENO это оптимальная линейная комбинация.
SparseRow linear_increment(unsigned order, const SparseRow& central, const SideSupport& own,
                           const SideSupport& opposite) {
    SparseRow row;
    if (order == 5) {
        add_scaled_row(row, own.far_slope, -1.0 / 15);
        add_scaled_row(row, own.near_slope, 11.0 / 30);
        add_scaled_row(row, central, 4.0 / 5);
        add_scaled_row(row, opposite.near_slope, -1.0 / 10);
    } else if (order == 3) {
        add_scaled_row(row, central, 2.0 / 3);
        add_scaled_row(row, own.near_slope, 1.0 / 3);
    } else {
        // Без ближней опоры состояние на стороне ребра равно среднему двух концов.
        return central;
    }
    return row;
}

// Сохраняет геометрию четырёх производных в порядке, который ожидает weno_increment2.
// Нелинейные веса зависят от полей и вычисляются при применении шаблона.
void append_weno_slopes(ReconstructionStencilData& rows, unsigned char order, double epsilon_scale,
                        const SparseRow& central, const SideSupport& own, const SideSupport& opposite) {
    rows.weno_order.push_back(order);
    rows.weno_epsilon_scale.push_back(epsilon_scale);
    const SparseRow* slopes[] = {&central, &own.near_slope, &opposite.near_slope, &own.far_slope};
    const unsigned used = order == 5 ? 4 : (order == 3 ? 2 : 0);
    for (unsigned k = 0; k < 4; ++k) {
        if (k < used) {
            for (const auto& term : *slopes[k]) {
                if (!std::isfinite(term.second)) {
                    throw ReconstructionError("Non-finite WENO derivative weight");
                }
                if (term.second == 0) {
                    continue;
                }
                rows.slope_nodes.push_back(term.first);
                rows.slope_weights.push_back(term.second);
            }
        }
        // Даже пустая производная занимает строку: номер следующей стороны остаётся предсказуемым.
        rows.slope_offsets.push_back(rows.slope_nodes.size());
    }
}
}

void append_stencil_sides(ReconstructionStencilData& rows, const ReconstructionGeometry& geometry,
                          const ReconstructionOptions& options, const ReconstructionSchemeInfo& scheme,
                          MeshIndex edge, double length, const SparseRow& central,
                          const SideSupport (&support)[2]) {
    const bool weno = scheme.family == ReconstructionFamily::Weno;
    const unsigned requested_order = scheme.layers == 2 ? 5 : 3;
    for (unsigned side = 0; side < 2; ++side) {
        const auto& own = support[side];
        const auto& opposite = support[1 - side];
        // Понижаем только ту сторону, для которой не хватило опорных точек.
        unsigned char order = 0;
        ReconstructionSideScheme actual = ReconstructionSideScheme::EdgeAverage;
        if (!own.far_slope.empty()) {
            order = 5;
            actual = weno ? ReconstructionSideScheme::Weno5 : ReconstructionSideScheme::Ebr5;
        } else if (!own.near_slope.empty()) {
            order = 3;
            actual = weno ? ReconstructionSideScheme::Weno3 : ReconstructionSideScheme::Ebr3;
        }
        const bool complete = order == requested_order;
        if (!complete && options.require_complete_stencil) {
            const std::string family = weno ? "WENO" : "EBR";
            throw ReconstructionError("Incomplete requested " + family + " stencil at edge " +
                                      std::to_string(edge) + " side " + std::to_string(side));
        }
        geometry.append_row(rows, linear_increment(order, central, own, opposite), edge, side);
        rows.complete.push_back(complete ? 1 : 0);
        rows.side_scheme.push_back(actual);
        rows.outer_distance.push_back(order ? own.near_point.distance : 0);
        rows.far_distance.push_back(own.far_distance);
        if (weno) {
            const double scale = length * std::sqrt(options.weno_epsilon);
            if (!std::isfinite(scale)) {
                throw ReconstructionError("WENO epsilon/edge scale overflow");
            }
            append_weno_slopes(rows, order, scale, central, own, opposite);
        }
    }
}
}
}
