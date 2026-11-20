#ifndef AERO_PACKED_WENO_H
#define AERO_PACKED_WENO_H
#include "residual_detail.h"

namespace aero {
namespace residual_detail {
// Возвращает четыре амплитуды волн для одной строки двумерного шаблона.
// В пределах ребра базис Роу фиксирован, поэтому сначала складываем разности состояний
// и только затем проецируем один готовый наклон.
template <WenoSlopeRow row, unsigned order>
KOKKOS_FORCEINLINE_FUNCTION bool characteristic_slope(const WenoEdgeTile<order>& tile, unsigned lane,
                                                  const double* field, const euler_detail::PlaneWaveBasis& basis,
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
KOKKOS_FORCEINLINE_FUNCTION double wave_increment(const double slopes[][4], unsigned component,
                                               bool right, double epsilon_scale) {
    const double values[] = {slopes[CentralSlope][component],
                             slopes[right ? NearRightSlope : NearLeftSlope][component],
                             slopes[right ? NearLeftSlope : NearRightSlope][component],
                             order == 5 ? slopes[right ? FarRightSlope : FarLeftSlope][component] : 0};
    return weno_increment2<order>(order, values, epsilon_scale);
}

// В плоском полном шаблоне центральный и ближние наклоны уже общие для двух сторон.
// У WENO5 и индикаторы этих наклонов можно посчитать за один проход.
template <unsigned order>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_wave_pair(const double slopes[][4], unsigned c,
                                              double left_epsilon, double right_epsilon,
                                              double& left, double& right) {
    const double center = slopes[CentralSlope][c];
    const double nl = slopes[NearLeftSlope][c], nr = slopes[NearRightSlope][c];
    const double fl = order == 5 ? slopes[FarLeftSlope][c] : 0;
    const double fr = order == 5 ? slopes[FarRightSlope][c] : 0;
    const double maximum = ::fmax(::fabs(center), ::fmax(::fmax(::fabs(nl), ::fabs(nr)),
                                                         ::fmax(::fabs(fl), ::fabs(fr))));
    if (!regular_weno_scale(maximum, left_epsilon) || !regular_weno_scale(maximum, right_epsilon)) {
        left = wave_increment<order>(slopes, c, false, left_epsilon);
        right = wave_increment<order>(slopes, c, true, right_epsilon);
        return;
    }
    const double el = left_epsilon * left_epsilon, er = right_epsilon * right_epsilon;
    if (order == 3) {
        left = weno_three<true>(center, nl, el);
        right = weno_three<true>(center, nr, er);
        return;
    }
    // Для левого следа правый наклон даёт внешний кандидат. Для правого следа
    // те же наклоны меняются ролями. Дальний кандидат у каждой стороны свой.
    const WenoIndicators l(center, nl), r(center, nr);
    left = weno_five_side<true>(center, nl, nr, fl, el, l, r);
    right = weno_five_side<true>(center, nr, nl, fr, er, r, l);
}

// Обе стороны используют один базис Роу. Перевод приращения в состояние общий с разреженной реконструкцией.
KOKKOS_FORCEINLINE_FUNCTION bool recover_traces(const euler_detail::PlaneWaveBasis& basis, const double* left,
                                           const double* right, EdgeTraces& traces) {
    return trace_from_increment<4>(4, &basis, left, traces.left, false, traces.left) &&
           trace_from_increment<4>(4, &basis, right, traces.right, true, traces.right);
}

// Начальные состояния и единичная нормаль задают общий плоский базис четырёх волн.
struct PlaneEdge {
    const double* left;  // Состояние в начале ориентированного ребра.
    const double* right; // Состояние в конце того же ребра.
    euler_detail::PlaneWaveBasis basis;
    euler_detail::InterfaceGas gas;

    // Узловой газ уже посчитан при обновлении поля. Здесь выбираем его для двух концов ребра.
    KOKKOS_FORCEINLINE_FUNCTION bool read_nodes(const Geometry& g, const double* field, Index edge) {
        const auto& surface = g.surfaces[edge];
        left = field + 4 * surface.begin;
        right = field + 4 * surface.end;
        return edge_gas<true>(g, surface, left, right, gas);
    }

    KOKKOS_FORCEINLINE_FUNCTION bool prepare(const Geometry& g, const double* field, Index edge) {
        return read_nodes(g, field, edge) && basis.prepare(gas, g.options.gamma);
    }

    // После восстановления волн возвращаемся к консервативным переменным и вычисляем поток Роу.
    KOKKOS_FORCEINLINE_FUNCTION bool flux(const Geometry& g, Index edge, EdgeTraces& traces,
                                                  EulerFluxResult& result) const {
        return recover_traces(basis, left, right, traces) &&
               finish_edge_flux<true, true>(g, g.surfaces[edge], left, right, gas, traces, nullptr, result);
    }
};

// Все наклоны считаются в одном базисе. Левый и правый след читают противоположные концы этой таблицы.
template <unsigned scheme_order>
KOKKOS_FORCEINLINE_FUNCTION bool prepare_edge_slopes(const Geometry& g, Index edge, const double* field,
                                            const WenoEdgeTile<scheme_order>& tile, unsigned lane,
                                            PlaneEdge& plane, double slopes[][4]) {
    // Обе стороны используют один базис Роу. Сразу возвращаем ошибку, если его нельзя построить.
    if (!plane.prepare(g, field, edge)) return false;
    const double gamma = g.options.gamma;
    const auto& basis = plane.basis;
    const double* normal = plane.gas.normal;
    // Для WENO3 достаточно трёх проекций, для WENO5 нужны ещё два дальних наклона.
    // Скорость звука в базисе одна на ребро. Все проекции используют одно и то же обратное значение.
    const double inverse_sound = inverse_value(basis.sound_speed);
    if (!characteristic_slope<CentralSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes) ||
        !characteristic_slope<NearLeftSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes) ||
        !characteristic_slope<NearRightSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes))
        return false;
    if (scheme_order == 5 &&
        (!characteristic_slope<FarLeftSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes) ||
         !characteristic_slope<FarRightSlope>(tile, lane, field, basis, normal, gamma, inverse_sound, slopes)))
        return false;

    return true;
}

// На входе пять наклонов (три для WENO3) в базисе ребра. На выходе в traces
// лежат ещё приращения волн. Преобразование в состояния выполняет PlaneEdge::flux.
template <unsigned scheme_order, bool use_mp5, bool use_mpweno, int hybrid_mode>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_packed_traces(const Geometry& g, Index edge,
                                                                   const double slopes[][4], EdgeTraces& traces) {
    // Гибриду нужны сразу все волны для общего ограничения средней поправки.
    const bool hybrid = hybrid_mode == 1 || (hybrid_mode < 0 && g.options.mp6_hybrid > 0);
    if (scheme_order == 5 && (use_mp5 || use_mpweno) && hybrid) {
        // Упаковка пятого порядка содержит все пять наклонов; для третьего эта ветка удаляется.
        mp6::reconstruct<use_mpweno>(slopes, 4, g.stencil.mp5, g.options.mp6_hybrid, traces.left, traces.right);
    } else {
        // Без гибрида пара следов каждой волны независима от остальных волн.
        for (unsigned c = 0; c < 4; ++c) {
            if (use_mpweno) {
                mpweno5::pair(slopes[FarLeftSlope][c], slopes[NearLeftSlope][c], slopes[CentralSlope][c],
                              slopes[NearRightSlope][c], slopes[FarRightSlope][c], g.stencil.mp5.alpha,
                              g.stencil.mp5.correction, traces.left[c], traces.right[c]);
                continue;
            }
            if (use_mp5) {
                mp5::pair(slopes[FarLeftSlope][c], slopes[NearLeftSlope][c], slopes[CentralSlope][c],
                          slopes[NearRightSlope][c], slopes[FarRightSlope][c], g.stencil.mp5.alpha,
                          traces.left[c], traces.right[c]);
                continue;
            }
            reconstruct_wave_pair<scheme_order>(slopes, c, g.stencil.weno_epsilon_scale[2 * edge],
                                                g.stencil.weno_epsilon_scale[2 * edge + 1], traces.left[c], traces.right[c]);
        }
    }
}

// Вычисляет поток для полного двумерного шаблона. Геометрия ребра была подготовлена до начала интегрирования.
template <unsigned scheme_order, bool use_mp5, bool use_mpweno, int hybrid_mode = -1>
KOKKOS_FORCEINLINE_FUNCTION bool packed_edge_flux(const Geometry& g, const double* field, Index edge,
                                             const WenoEdgeTile<scheme_order>& tile, unsigned lane,
                                             EulerFluxResult& out) {
    PlaneEdge plane;
    double slopes[WenoEdgeTile<scheme_order>::rows_per_edge][4];
    if (!prepare_edge_slopes(g, edge, field, tile, lane, plane, slopes))
        return false;
    EdgeTraces traces;
    reconstruct_packed_traces<scheme_order, use_mp5, use_mpweno, hybrid_mode>(g, edge, slopes, traces);
    return plane.flux(g, edge, traces, out);
}

// На границе и у неполного шаблона расчёт использует обычные строки реконструкции.
// CPU обрабатывает их сразу, GPU — отдельным обходом по заранее собранному списку.
template <unsigned order>
AERO_HOST_DEVICE inline const WenoEdgeTile<order>* complete_tile(const Geometry& g, Index face) {
    if (!g.packed.tiles || face >= g.edges)
        return nullptr;
    const auto* tile = static_cast<const WenoEdgeTile<order>*>(g.packed.tiles) + face / 32;
    return tile->order[face % 32].complete() ? tile : nullptr;
}

// Одна запись потока на ребро. Запись скорости нужна только вызывающей стороне, запросившей её.
template <bool use_mp5, bool use_mpweno = false, int hybrid_mode = -1, unsigned order>
AERO_HOST_DEVICE inline bool compute_packed_face(const FaceBatch& batch, Index face, const WenoEdgeTile<order>& tile) {
    if (!packed_edge_flux<order, use_mp5, use_mpweno, hybrid_mode>(
            batch.geometry, batch.state, face, tile, face % 32, batch.flux[face]))
        return false;
    batch.save_speed(face);
    return true;
}

}
}
#endif
