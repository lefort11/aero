#ifndef AERO_PACKED_WENO_H
#define AERO_PACKED_WENO_H
#include "residual_detail.h"

namespace aero {
namespace residual_detail {
// В пакете одинаковые слагаемые 32 соседних рёбер лежат подряд. Адресация и
// веса общие для состояний, характеристик и физических потоков. Читатель
// определяет, какую величину надо взять в узле.
template<WenoSlopeRow row, unsigned order, class Read>
KOKKOS_FORCEINLINE_FUNCTION bool packed_slope(const WenoEdgeTile<order>& tile, unsigned lane,
                                               const Read& read, double result[4]) {
    constexpr unsigned first = first_slope_term(row);
    // i нумерует слагаемые одной короткой строки, глобальный номер узла храним отдельно.
    const auto values = [&](Index i, double* buffer) { return read(tile.nodes(first + static_cast<unsigned>(i), lane), buffer); };
    const auto weight = [&](Index i) { return tile.weights(first - row + static_cast<unsigned>(i) - 1, lane); };
    return reconstruction_detail::sum_slope<4>(4, slope_terms(row), values, weight, result);
}

// Возвращает четыре амплитуды волн для одной строки двумерного шаблона.
// В пределах ребра базис Роу фиксирован, поэтому сначала складываем разности состояний
// и только затем проецируем один готовый наклон.
template <WenoSlopeRow row, unsigned order>
KOKKOS_FORCEINLINE_FUNCTION bool characteristic_slope(const WenoEdgeTile<order>& tile, unsigned lane,
                                                  const double* field, const euler_detail::PlaneWaveBasis& basis,
                                                  const double* normal, double gamma, double inverse_sound,
                                                  FixedMatrix<double,WenoEdgeTile<order>::rows_per_edge,4>& waves) {
    const auto state = [&](Index node, double*) { return field + 4 * node; };
    double slope[4];
    packed_slope<row>(tile, lane, state, slope);
    return euler_detail::project_characteristic<2>(basis, normal, gamma, basis.kinetic_energy(),
                                                    slope, waves.row(row), inverse_sound) == EulerStatus::Success;
}

// Разворачиваем ближние наклоны в порядке, принятом формулой WENO на выбранной стороне.
// Нелинейные веса считаются только после перехода в характеристики.
template <unsigned order, unsigned rows>
KOKKOS_FORCEINLINE_FUNCTION double wave_increment(const FixedMatrix<double,rows,4>& slopes, unsigned component,
                                               bool right, double epsilon_scale) {
    const double values[] = {slopes(CentralSlope, component),
                             slopes(right ? NearRightSlope : NearLeftSlope, component),
                             slopes(right ? NearLeftSlope : NearRightSlope, component),
                             order == 5 ? slopes(right ? FarRightSlope : FarLeftSlope, component) : 0};
    return weno_increment2<order>(order, values, epsilon_scale);
}

// В плоском полном шаблоне центральный и ближние наклоны уже общие для двух сторон.
// У WENO5 и индикаторы этих наклонов можно посчитать за один проход.
template <unsigned order>
KOKKOS_FORCEINLINE_FUNCTION void reconstruct_wave_pair(const FixedMatrix<double,WenoEdgeTile<order>::rows_per_edge,4>& slopes, unsigned c,
                                              double left_epsilon, double right_epsilon,
                                              double& left, double& right) {
    const double center = slopes(CentralSlope, c);
    const double nl = slopes(NearLeftSlope, c), nr = slopes(NearRightSlope, c);
    const double fl = order == 5 ? slopes(FarLeftSlope, c) : 0;
    const double fr = order == 5 ? slopes(FarRightSlope, c) : 0;
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
                                            PlaneEdge& plane, FixedMatrix<double,WenoEdgeTile<scheme_order>::rows_per_edge,4>& slopes) {
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
                                                                   const FixedMatrix<double,WenoEdgeTile<scheme_order>::rows_per_edge,4>& slopes, EdgeTraces& traces) {
    // Гибриду нужны сразу все волны для общего ограничения средней поправки.
    const bool hybrid = hybrid_mode == 1 || (hybrid_mode < 0 && g.options.mp6_hybrid > 0);
    if (scheme_order == 5 && (use_mp5 || use_mpweno) && hybrid) {
        // Упаковка пятого порядка содержит все пять наклонов; для третьего эта ветка удаляется.
        mp6::reconstruct<use_mpweno>(slopes, 4, g.stencil.mp5, g.options.mp6_hybrid, traces.left, traces.right);
    } else {
        // Без гибрида пара следов каждой волны независима от остальных волн.
        for (unsigned c = 0; c < 4; ++c) {
            if (use_mpweno) {
                mpweno5::pair(slopes(FarLeftSlope, c), slopes(NearLeftSlope, c), slopes(CentralSlope, c),
                              slopes(NearRightSlope, c), slopes(FarRightSlope, c), g.stencil.mp5.alpha,
                              g.stencil.mp5.correction, traces.left[c], traces.right[c]);
                continue;
            }
            if (use_mp5) {
                mp5::pair(slopes(FarLeftSlope, c), slopes(NearLeftSlope, c), slopes(CentralSlope, c),
                          slopes(NearRightSlope, c), slopes(FarRightSlope, c), g.stencil.mp5.alpha,
                          traces.left[c], traces.right[c]);
                continue;
            }
            reconstruct_wave_pair<scheme_order>(slopes, c, g.stencil.epsilon(2 * edge),
                                                g.stencil.epsilon(2 * edge + 1), traces.left[c], traces.right[c]);
        }
    }
}

// Вычисляет поток для полного двумерного шаблона. Геометрия ребра была подготовлена до начала интегрирования.
template <unsigned scheme_order, bool use_mp5, bool use_mpweno, int hybrid_mode = -1>
KOKKOS_FORCEINLINE_FUNCTION bool packed_edge_flux(const Geometry& g, const double* field, Index edge,
                                             const WenoEdgeTile<scheme_order>& tile, unsigned lane,
                                             EulerFluxResult& out) {
    PlaneEdge plane;
    FixedMatrix<double,WenoEdgeTile<scheme_order>::rows_per_edge,4> slopes;
    if (!prepare_edge_slopes(g, edge, field, tile, lane, plane, slopes))
        return false;
    EdgeTraces traces;
    reconstruct_packed_traces<scheme_order, use_mp5, use_mpweno, hybrid_mode>(g, edge, slopes, traces);
    return plane.flux(g, edge, traces, out);
}

// В декартовых переменных не нужен базис Роу для реконструкции. Развёртка
// трёх или пяти строк позволяет ядру хранить наклоны четырёх компонент в регистрах.
template<bool physical, unsigned order, bool mp, bool mpweno, bool hybrid>
KOKKOS_FORCEINLINE_FUNCTION bool cartesian_packed_values(const Geometry& g, Index edge, const double* field,
        const PlaneEdge& plane, const WenoEdgeTile<order>& tile, unsigned lane, EdgeTraces& traces) {
    const auto read = [&](Index node, double* buffer) -> const double* {
        const double* state = field + 4 * node;
        if (!physical) return state;
        // Примитивные переменные этой стадии уже готовы. Проекция зависит от
        // нормали текущего ребра, поэтому вычисляем её при чтении шаблона.
        const auto& gas = g.packed.gas[node];
        double speed;
        return gas.density > 0. && euler_detail::project_flux(2,state,gas,plane.gas.normal,buffer,&speed)
               == EulerStatus::Success ? buffer : nullptr;
    };
    FixedMatrix<double,WenoEdgeTile<order>::rows_per_edge,4> slopes;
    if (!packed_slope<CentralSlope>(tile,lane,read,slopes.row(CentralSlope)) ||
        !packed_slope<NearLeftSlope>(tile,lane,read,slopes.row(NearLeftSlope)) ||
        !packed_slope<NearRightSlope>(tile,lane,read,slopes.row(NearRightSlope))) return false;
    if (order == 5 && (!packed_slope<FarLeftSlope>(tile,lane,read,slopes.row(FarLeftSlope)) ||
                       !packed_slope<FarRightSlope>(tile,lane,read,slopes.row(FarRightSlope)))) return false;
    reconstruct_packed_traces<order,mp,mpweno,0>(g,edge,slopes,traces);
    const auto& surface = g.surfaces[edge];
    if (hybrid) {
        // Базовые приращения уже получены в осях сетки. Ограничиваем только
        // добавку MP6 в нормальном и касательном направлениях данного ребра.
        const FaceMomentumAxes axes(2, surface.area, plane.gas.normal);
        correct_cartesian_pair<mpweno>(axes, slopes, g.stencil.mp5, g.options.mp6_hybrid,
                                       traces.left, traces.right);
    }
    double fl[4], fr[4];
    const double* left = read(surface.begin,fl), *right = read(surface.end,fr);
    if (!left || !right) return false;
    add_half_increment(4,left,traces.left,false,traces.left);
    add_half_increment(4,right,traces.right,true,traces.right);
    return true;
}

template<bool physical, unsigned order, bool mp, bool mpweno, bool hybrid>
KOKKOS_FORCEINLINE_FUNCTION bool cartesian_packed_flux(const FaceBatch& batch, Index edge,
                                                        const WenoEdgeTile<order>& tile) {
    const auto& g = batch.geometry;
    const unsigned lane = edge_tile::lane(edge);
    PlaneEdge plane;
    EdgeTraces states, flux;
    if (!plane.read_nodes(g,batch.state,edge) ||
        !cartesian_packed_values<false,order,mp,mpweno,hybrid>(g,edge,batch.state,plane,tile,lane,states)) return false;
    // При реконструкции потоков диссипация использует восстановленные состояния,
    // а симметричная часть — восстановленную проекцию потока.
    if (physical && !cartesian_packed_values<true,order,mp,mpweno,hybrid>(g,edge,batch.state,plane,tile,lane,flux))
        return false;
    if (!finish_edge_flux<true,!physical>(g,g.surfaces[edge],plane.left,plane.right,plane.gas,
                                          states,physical ? &flux : nullptr,batch.flux[edge])) return false;
    batch.save_speed(edge);
    return true;
}

// На границе и у неполного шаблона расчёт использует обычные строки реконструкции.
// CPU обрабатывает их сразу, GPU — отдельным обходом по заранее собранному списку.
template <unsigned order>
AERO_HOST_DEVICE inline const WenoEdgeTile<order>* complete_tile(const Geometry& g, Index face) {
    if (!g.packed.tiles || face >= g.edges)
        return nullptr;
    const auto* tile = static_cast<const WenoEdgeTile<order>*>(g.packed.tiles) + edge_tile::index(face);
    return tile->order[edge_tile::lane(face)].complete() ? tile : nullptr;
}

// Одна запись потока на ребро. Запись скорости нужна только вызывающей стороне, запросившей её.
template <bool use_mp5, bool use_mpweno = false, int hybrid_mode = -1, unsigned order>
AERO_HOST_DEVICE inline bool compute_packed_face(const FaceBatch& batch, Index face, const WenoEdgeTile<order>& tile) {
    if (!packed_edge_flux<order, use_mp5, use_mpweno, hybrid_mode>(
            batch.geometry, batch.state, face, tile, edge_tile::lane(face), batch.flux[face]))
        return false;
    batch.save_speed(face);
    return true;
}

}
}
#endif
