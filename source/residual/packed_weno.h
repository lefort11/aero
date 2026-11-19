#ifndef AERO_PACKED_WENO_H
#define AERO_PACKED_WENO_H
#include "residual_detail.h"

namespace aero {
namespace residual_detail {
// Нулевая плотность отмечает ошибку разбора. Диагностика укажет на ребро, использующее этот узел.
AERO_HOST_DEVICE inline void decode_nodal_gas(unsigned dimension, const double* field, double gamma, Index i, EulerState* gas) {
    if (euler_state(dimension, field + (dimension + 2) * i, gamma, gas + i) != EulerStatus::Success)
        gas[i].density = 0;
}

// Каждое слагаемое хранит номер своего узла в четырёх битах. На CPU значения одного ребра
// лежат подряд (stride = 1), на GPU одноимённые компоненты соседних потоков идут подряд (stride = 32).
template <unsigned stride>
AERO_HOST_DEVICE inline double wave_at_term(Index slots, unsigned term, const double* waves, unsigned component) {
    const unsigned slot = static_cast<unsigned>((slots >> (4 * term)) & 15);
    return waves[(4 * slot + component) * stride];
}

// Складываем значения относительно первого узла строки, сохраняя исходный порядок операций.
template <WenoSlopeRow row, unsigned stride, unsigned scheme_order>
AERO_HOST_DEVICE inline double packed_slope(const WenoEdgeTile<scheme_order>& tile, unsigned lane, Index slots,
                                               const double* q, unsigned component) {
    constexpr unsigned first = first_slope_term(row);
    // Коэффициент первого узла в разности не нужен, поэтому перед строкой пропущено ровно row весов.
    constexpr unsigned weight = first - row;
    const double reference = wave_at_term<stride>(slots, first, q, component);
    double sum = 0;
    for (unsigned j = 1; j < slope_terms(row); ++j) {
        const double value = wave_at_term<stride>(slots, first + j, q, component);
        sum += tile.weights[weight + j - 1][lane] * (value - reference);
    }
    return sum;
}

// Центральный наклон нужен обеим сторонам. На правой стороне ближние наклоны меняются местами.
template <unsigned stride, unsigned scheme_order>
AERO_HOST_DEVICE inline void reconstruct_wave(const Geometry& g, Index edge, const WenoEdgeTile<scheme_order>& tile,
                                               unsigned lane, Index slots, const double* q, unsigned c,
                                               double& left, double& right) {
    const double central = packed_slope<CentralSlope, stride>(tile, lane, slots, q, c);
    const double near_left = packed_slope<NearLeftSlope, stride>(tile, lane, slots, q, c);
    const double near_right = packed_slope<NearRightSlope, stride>(tile, lane, slots, q, c);
    double left_slopes[] = {central, near_left, near_right, 0};
    double right_slopes[] = {central, near_right, near_left, 0};
    if (scheme_order == 5) {
        left_slopes[3] = packed_slope<FarLeftSlope, stride>(tile, lane, slots, q, c);
        right_slopes[3] = packed_slope<FarRightSlope, stride>(tile, lane, slots, q, c);
    }
    left = weno_increment2(scheme_order, left_slopes, g.stencil.weno_epsilon_scale[2 * edge]);
    right = weno_increment2(scheme_order, right_slopes, g.stencil.weno_epsilon_scale[2 * edge + 1]);
}

// Обе стороны используют один базис Роу. Перевод приращения в состояние общий с разреженной реконструкцией.
AERO_HOST_DEVICE inline bool recover_traces(const EulerRoeBasis& basis, const double* left, const double* right, EdgeTraces& traces) {
    return trace_from_increment<4>(4, &basis, left, traces.left, false, traces.left) &&
           trace_from_increment<4>(4, &basis, right, traces.right, true, traces.right);
}

// Вычисляет поток для полного двумерного шаблона. Геометрия ребра была подготовлена до начала интегрирования.
template <unsigned stride, unsigned scheme_order>
AERO_HOST_DEVICE inline bool packed_edge_flux(Geometry g, const double* field, Index edge,
                                                 const WenoEdgeTile<scheme_order>& tile, unsigned lane, Index slots,
                                                 EulerFluxResult& out, double* q) {
    g.dim = 2;
    g.options.solver = ResidualSolver::Roe;
    const auto& surface = g.surfaces[edge];
    const double* left = g.node_row(field, surface.begin);
    const double* right = g.node_row(field, surface.end);
    EulerRoeBasis basis;
    euler_detail::InterfaceGas gas;
    const double* normal = gas.normal;
    if (!edge_gas(g, surface, left, right, gas) ||
        euler_detail::build_roe_basis(2, gas.left, gas.right, surface.area, normal, g.options.gamma, &basis)
            != EulerStatus::Success)
        return false;
    // Одна проекция на каждый различный узел. Тот же базис используется для возвращения к состояниям.
    for (unsigned j = 0; j < tile.nodes_per_edge; ++j) {
        double wave[4];
        if (euler_detail::project_characteristic<2>(basis, normal, g.options.gamma, basis.kinetic_energy(),
                g.node_row(field, tile.nodes[j][lane]), wave) != EulerStatus::Success)
            return false;
        for (unsigned c = 0; c < 4; ++c)
            q[(4 * j + c) * stride] = wave[c];
    }
    // Вес каждой волны зависит от её собственных наклонов на текущей стадии интегратора.
    EdgeTraces traces;
    for (unsigned c = 0; c < 4; ++c)
        reconstruct_wave<stride>(g, edge, tile, lane, slots, q, c, traces.left[c], traces.right[c]);
    if (!recover_traces(basis, left, right, traces))
        return false;
    return finish_edge_flux(g, surface, left, right, gas, traces, nullptr, out);
}

}
}
#endif
