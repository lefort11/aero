#ifndef AERO_MPWENO5_EDGES_H
#define AERO_MPWENO5_EDGES_H

#include <Kokkos_Core.hpp>
#include "../residual/packed_weno.h"

namespace aero {
namespace device_work {
namespace mpweno_edges {
using namespace residual_detail;

// Первый проход. Одно ребро имеет пять геометрических наклонов. Каждый переводим
// в четыре характеристики, используя один базис Роу для левого и правого следа.
KOKKOS_INLINE_FUNCTION void project(const FaceBatch& batch, Index face, MpEdgeTile& work,
                                    unsigned lane, Index* bad) {
    work.order[lane] = {};
    const auto& g = batch.geometry;
    if (face >= g.edges)
        return;
    const auto* stencil = static_cast<const WenoEdgeTile<5>*>(g.packed.tiles) + face / MpEdgeTile::width;
    const unsigned geometry_lane = face % MpEdgeTile::width;
    const auto order = stencil->order[geometry_lane];
    if (!order.ready())
        return; // Для неполных шаблонов и границ уже подготовлены отдельные списки.

    PlaneEdge edge;
    double slopes[5][4];
    if (!edge.prepare(g, batch.state, face) ||
        !project_stencil(*stencil, geometry_lane, batch.state, edge, g.options.gamma, slopes)) {
        Kokkos::atomic_fetch_min(bad, face);
        return;
    }
    work.store_basis(lane, edge.basis);
    for (unsigned row = 0; row < 5; ++row)
        for (unsigned wave = 0; wave < 4; ++wave)
            work.waves[row][wave][lane] = slopes[row][wave];
    if (!order.complete()) {
        // Дальний наклон укороченной стороне не нужен. В его ячейке сохраняем
        // её штатный результат WENO3, включая индивидуальную регуляризацию.
        for (unsigned side = 0; side < 2; ++side) {
            if (!order.shortened(side))
                continue;
            for (unsigned wave = 0; wave < 4; ++wave)
                work.waves[side ? FarRightSlope : FarLeftSlope][wave][lane] =
                    wave_increment<3>(slopes, wave, side != 0, g.stencil.weno_epsilon_scale[2 * face + side]);
        }
    }
    work.order[lane] = order;
}

// Один исполнитель сначала владеет одной парой следов. После срабатывания
// датчика свободные исполнители помогают вычислять веса других пар той же группы.
constexpr unsigned team_size = 320;
constexpr unsigned warp_size = MpEdgeTile::width;
using Teams = Kokkos::TeamPolicy<Kokkos::Cuda, Kokkos::LaunchBounds<team_size>>;

struct WeightQueue {
    // На пару приходится не больше двух нелинейных задач. Компоненты храним
    // отдельно: при обработке очереди соседние потоки читают соседние double.
    double beta[3][2 * team_size];
    double polynomial[3][2 * team_size];
    unsigned count[team_size / warp_size]; // Сколько задач прислала каждая группа из 32 потоков.

    // В очередь передаём только шесть чисел. Интервал MP остаётся у владельца стороны.
    KOKKOS_INLINE_FUNCTION void put(unsigned slot, const mpweno5::Side& side) {
        const auto values = side.polynomials();
        beta[0][slot] = side.indicators.first;
        beta[1][slot] = side.indicators.middle;
        beta[2][slot] = side.indicators.last;
        polynomial[0][slot] = values.first;
        polynomial[1][slot] = values.middle;
        polynomial[2][slot] = values.last;
    }

    // Вызывается после барьера, когда известны размеры очередей всех групп.
    // first указывает начало заданий этой группы, total ограничивает общий обход блока.
    KOKKOS_INLINE_FUNCTION void task_range(unsigned warp, unsigned& first, unsigned& total) const {
        first = total = 0;
        for (unsigned group = 0; group < team_size / warp_size; ++group) {
            if (group < warp)
                first += count[group];
            total += count[group];
        }
    }

    KOKKOS_INLINE_FUNCTION void solve(unsigned slot, bool correction) {
        const mpweno5::Indicators indicators = {beta[0][slot], beta[1][slot], beta[2][slot]};
        const mpweno5::Polynomials values = {polynomial[0][slot], polynomial[1][slot], polynomial[2][slot]};
        // Индикаторы этого задания больше никому не нужны. Его ответ занимает
        // первую ячейку той же записи и остаётся в общей памяти блока.
        beta[0][slot] = indicators.regular()
            ? mpweno5::weighted_value<mpweno5::WeightRange::ScaledProducts>(indicators, values, correction)
            : mpweno5::weighted_value<mpweno5::WeightRange::DividedWeights>(indicators, values, correction);
    }

    // После общего барьера ответ забирает тот поток, который поставил сторону в очередь.
    KOKKOS_INLINE_FUNCTION double result(unsigned slot) const { return beta[0][slot]; }
};

// Второй проход. Четыре последовательные группы по 32 потока разбирают одну
// плитку: в каждой группе одна волна, у каждого потока своё ребро.
template <bool correction>
inline void reconstruct_waves(const Kokkos::Cuda& exec, MpEdgeTile* tiles, Index faces, double alpha) {
    const Index tiles_count = (faces + warp_size - 1) / warp_size;
    const Index teams_count = (tiles_count * 4 * warp_size + team_size - 1) / team_size;
    Kokkos::parallel_for("Aero: MPWENO5 wave pairs", Teams(exec, teams_count, team_size, 1),
        KOKKOS_LAMBDA(const Teams::member_type& team) {
            __shared__ WeightQueue queue;
            const unsigned rank = team.team_rank(), warp = rank / warp_size, lane = rank % warp_size;
            const Index task = team_size * static_cast<Index>(team.league_rank()) + rank;
            const unsigned wave = (task / warp_size) % 4;
            const Index tile_id = task / (4 * warp_size);
            const bool present = tile_id * warp_size + lane < faces;
            // Последняя группа может быть неполной. Все её потоки участвуют в голосовании
            // и барьерах, но только действительные рёбра читают наклоны и пишут следы.
            auto& tile = tiles[present ? tile_id : 0];
            const auto order = present ? tile.order[lane] : PackedEdgeOrder{};
            const bool valid = order.ready();
            double slope[5];
            for (unsigned row = 0; row < 5; ++row)
                slope[row] = valid ? tile.waves[row][wave][lane] : 0.;
            mpweno5::Side sides[2];
            mpweno5::prepare_pair<true>(slope[FarLeftSlope], slope[NearLeftSlope], slope[CentralSlope],
                                       slope[NearRightSlope], slope[FarRightSlope], alpha, sides[0], sides[1]);

            // Один бит маски соответствует одному владельцу. Считаем задачи слева
            // и справа до выделения мест, чтобы не использовать глобальный атомарный счётчик.
            bool weighted[2];
            unsigned votes[2];
            for (unsigned side = 0; side < 2; ++side) {
                weighted[side] = valid && !order.shortened(side) && sides[side].needs_weights<true>();
                votes[side] = __ballot_sync(0xffffffffu, weighted[side]);
            }
            if (lane == 0)
                queue.count[warp] = __popc(votes[0]) + __popc(votes[1]);
            team.team_barrier();

            // Сумма размеров предыдущих групп даёт начало нашей очереди. Сумма
            // по всему блоку нужна ниже, когда свободные потоки разберут задания.
            unsigned first, total;
            queue.task_range(warp, first, total);
            unsigned slot[2];
            double answer[2];
            for (unsigned side = 0; side < 2; ++side) {
                // Младшие биты показывают, сколько задач той же стороны уже стоит перед нами.
                slot[side] = first + __popc(votes[side] & ((1u << lane) - 1u));
                first += __popc(votes[side]);
                if (weighted[side])
                    queue.put(slot[side], sides[side]);
                else if (order.shortened(side))
                    answer[side] = slope[side ? FarRightSlope : FarLeftSlope];
                else
                    answer[side] = sides[side].direct_trace<true>();
            }
            team.team_barrier();

            // Теперь задания лежат подряд. Каждый исполнитель считает одну и ту же
            // скалярную формулу MPWENO5; датчик, интервалы и положения рёбер уже известны.
            for (unsigned job = rank; job < total; job += team_size)
                queue.solve(job, correction);
            team.team_barrier();

            if (valid) {
                for (unsigned side = 0; side < 2; ++side) {
                    if (weighted[side])
                        answer[side] = sides[side].bounds.clamp<true>(queue.result(slot[side]));
                    // Пять исходных наклонов уже прочитаны их единственным владельцем.
                    // Первые две строки этой волны теперь содержат левый и правый след.
                    tile.waves[side][wave][lane] = answer[side];
                }
            }
            team.team_barrier(); // Kokkos может повторно использовать блок для следующей плитки.
        });
}

// Поправка задана для всего расчёта. Выбираем формулу один раз на пакет рёбер,
// чтобы каждой стороне не приходилось проверять один и тот же параметр.
inline void reconstruct(const Kokkos::Cuda& exec, MpEdgeTile* tiles, Index faces, Mp5Settings settings) {
    if (settings.correction)
        reconstruct_waves<true>(exec, tiles, faces, settings.alpha);
    else
        reconstruct_waves<false>(exec, tiles, faces, settings.alpha);
}

// Третий проход. Возвращаем обе стороны из характеристик в консервативные
// переменные и передаём их обычному решателю Роу. Оба узла получат один поток ребра.
KOKKOS_INLINE_FUNCTION void write_flux(const FaceBatch& batch, Index face, const MpEdgeTile& work,
                                       unsigned lane, Index* bad) {
    if (!work.order[lane].ready())
        return;
    PlaneEdge edge;
    if (edge.read_nodes(batch.geometry, batch.state, face)) {
        work.load_basis(lane, edge.gas.normal, edge.basis);
        EdgeTraces traces;
        for (unsigned wave = 0; wave < 4; ++wave) {
            traces.left[wave] = work.waves[0][wave][lane];
            traces.right[wave] = work.waves[1][wave][lane];
        }
        if (edge.flux(batch.geometry, face, traces, batch.flux[face])) {
            batch.save_speed(face);
            return;
        }
    }
    Kokkos::atomic_fetch_min(bad, face);
}
} // namespace mpweno_edges
} // namespace device_work
} // namespace aero
#endif
