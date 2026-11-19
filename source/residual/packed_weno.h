#ifndef AERO_PACKED_WENO_H
#define AERO_PACKED_WENO_H
#include "residual_detail.h"

namespace aero {
namespace residual_detail {
// Нулевая плотность отмечает ошибку разбора. Диагностика укажет на ребро, использующее этот узел.
AERO_HOST_DEVICE inline void decode_nodal_gas(unsigned dimension, const double* field, double gamma,
                                              Index i, EulerState* gas) {
    if (euler_state(dimension, field + (dimension + 2) * i, gamma, gas + i) != EulerStatus::Success)
        gas[i].density = 0;
}

// Возвращает четыре амплитуды волн для одной строки двумерного шаблона.
// В пределах ребра базис Роу фиксирован, поэтому сначала складываем разности состояний
// и только затем проецируем один готовый наклон.
template <WenoSlopeRow row, unsigned order>
AERO_HOST_DEVICE inline bool characteristic_slope(const WenoEdgeTile<order>& tile, unsigned lane,
                                                  const double* field, const EulerRoeBasis& basis,
                                                  const double* normal, double gamma, double inverse_sound,
                                                  double waves[][4]) {
    constexpr unsigned first = first_slope_term(row);
    const auto values = [&](Index i, double*) { return field + 4ull * tile.nodes[first + i][lane]; };
    // В каждой предыдущей строке коэффициент опорного узла пропущен.
    const auto weight = [&](Index i) { return tile.weights[first - row + i - 1][lane]; };
    double slope[4];
    sum_slope<4>(4, slope_terms(row), values, weight, slope);
    return euler_detail::project_characteristic<2>(basis, normal, gamma, basis.kinetic_energy(),
                                                    slope, waves[row], inverse_sound) == EulerStatus::Success;
}

// Разворачиваем ближние наклоны в порядке, принятом формулой WENO на выбранной стороне.
// Нелинейные веса считаются только после перехода в характеристики.
template <unsigned order>
AERO_HOST_DEVICE inline double wave_increment(const double slopes[][4], unsigned component,
                                               bool right, double epsilon_scale) {
    const double values[] = {slopes[CentralSlope][component],
                             slopes[right ? NearRightSlope : NearLeftSlope][component],
                             slopes[right ? NearLeftSlope : NearRightSlope][component],
                             order == 5 ? slopes[right ? FarRightSlope : FarLeftSlope][component] : 0};
    return weno_increment2(order, values, epsilon_scale);
}

// Обе стороны используют один базис Роу. Перевод приращения в состояние общий с разреженной реконструкцией.
AERO_HOST_DEVICE inline bool recover_traces(const EulerRoeBasis& basis, const double* left,
                                           const double* right, EdgeTraces& traces) {
    return trace_from_increment<4>(4, &basis, left, traces.left, false, traces.left) &&
           trace_from_increment<4>(4, &basis, right, traces.right, true, traces.right);
}

// Вычисляет поток для полного двумерного шаблона. Геометрия ребра была подготовлена до начала интегрирования.
template <unsigned scheme_order>
AERO_HOST_DEVICE inline bool packed_edge_flux(Geometry g, const double* field, Index edge,
                                             const WenoEdgeTile<scheme_order>& tile, unsigned lane,
                                             EulerFluxResult& out) {
    g.dim = 2;
    g.options.solver = ResidualSolver::Roe;
    const auto& surface = g.surfaces[edge];
    const double* left = g.node_row(field, surface.begin);
    const double* right = g.node_row(field, surface.end);
    EulerRoeBasis basis;
    euler_detail::InterfaceGas gas;
    const double* normal = gas.normal;
    if (!edge_gas(g, surface, left, right, gas) ||
        euler_detail::build_roe_basis<true, 2>(2, gas.left, gas.right, surface.area, normal, g.options.gamma, &basis)
            != EulerStatus::Success)
        return false;
    // Для WENO3 достаточно трёх проекций, для WENO5 нужны ещё два дальних наклона.
    double slopes[WenoEdgeTile<scheme_order>::rows_per_edge][4];
    const double gamma = g.options.gamma;
    // Скорость звука в базисе одна на ребро. Все проекции используют одно и то же обратное значение.
    const double inverse_sound = 1 / basis.sound_speed;
    if (!characteristic_slope<CentralSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes) ||
        !characteristic_slope<NearLeftSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes) ||
        !characteristic_slope<NearRightSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes))
        return false;
    if (scheme_order == 5 &&
        (!characteristic_slope<FarLeftSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes) ||
         !characteristic_slope<FarRightSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes)))
        return false;

    EdgeTraces traces;
    // Каждая волна получает свои нелинейные веса. Справа смотрим на тот же шаблон с другого конца.
    for (unsigned c = 0; c < 4; ++c) {
        traces.left[c] = wave_increment<scheme_order>(slopes, c, false, g.stencil.weno_epsilon_scale[2 * edge]);
        traces.right[c] = wave_increment<scheme_order>(slopes, c, true, g.stencil.weno_epsilon_scale[2 * edge + 1]);
    }
    if (!recover_traces(basis, left, right, traces))
        return false;
    return finish_edge_flux(g, surface, left, right, gas, traces, nullptr, out);
}

}
}
#endif
