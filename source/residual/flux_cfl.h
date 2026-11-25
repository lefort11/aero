#ifndef AERO_FLUX_CFL_H
#define AERO_FLUX_CFL_H
#include "../parallel/kernel_math.h"
#include "stencil_rates.h"
#include "../timestep/cfl_detail.h"
#include "../timestep/adaptive_step.h"
#include "../timestep/local_rate_limit.h"
#include "../mesh/mesh_index.h"

namespace aero {
namespace residual_detail {
// Связывает площадки узла с их площадями и оценками скоростей волн.
struct FaceSpeeds {
    FaceIncidence incidence; // Из записи списка узла получаем номер площадки.
    SurfaceTable surfaces;       // Модуль площади умножает скорость в сумме CFL.
    const EulerFluxResult* flux;   // Скорости, уже вычисленные при сборке потока.
    // Возвращает скорость волны и площадь по записи инцидентности узла.
    AERO_HOST_DEVICE cfl_detail::FaceSpeed operator()(Index j) const {
        const auto face = incidence[j].face();
        return {flux[face].max_wave_speed, surfaces[face].measure};
    }
};
// Без списка узлов обходим всю сетку. При разбиении процесс передаёт список собственных узлов.
struct FluxCfl {
    IndexArray<Index> offsets; // Диапазоны смежных площадок для каждого узла.
    FaceSpeeds speeds;    // Передаём в редукцию Kokkos только нужные ей массивы.
#ifdef AERO_HAS_VISCOSITY
    // Узловые строки вязкого ограничения шага.
    const viscosity::Row* diffusion;
#endif
    // Объёмы для перевода суммы скоростей на гранях в ограничение шага.
    const double* volumes;
    // Соответствие обрабатываемых строк собственным узлам.
    const MeshIndex* owned;
    // Число обрабатываемых строк.
    Index count;
    const double* sa_rate; // Диагональ переноса SA, в единицах обратного времени.
    const double* heights; // Минимум высот исходных ячеек вокруг узла.
    const EulerState* gas; // Уже вычисленные при сборке потоков узловые состояния Euler.
    const double* state; // Консервативное поле, если на GPU нет отдельного массива примитивных переменных.
    // Показатель адиабаты при оценке скорости волны.
    double gamma;
    // Размерность газового состояния.
    unsigned dimension;
    double linear_speed;   // Для Linear скорость определяется однородным фоном.
    // Связывает геометрию, скорости волн и объёмы для вычисления шагов CFL.
    FluxCfl(const Geometry& grid, const EulerFluxResult* flux, const double* node_volumes,
            const MeshIndex* owned_nodes, Index rows, const double* transport_rate = nullptr,
            const double* node_heights = nullptr,const double* node_state=nullptr)
        : offsets(grid.offsets), speeds{grid.incidence, grid.surfaces, flux},
#ifdef AERO_HAS_VISCOSITY
          diffusion(grid.options.diffusion),
#endif
          volumes(node_volumes),
          owned(owned_nodes), count(rows), sa_rate(transport_rate), heights(node_heights),
          gas(grid.packed.gas), state(node_state), gamma(grid.options.gamma), dimension(grid.dim), linear_speed(0.) {
        const auto& background = grid.options.background;
        if (background.active())
            linear_speed = wave_speed(dimension, background.velocity, background.sound);
    }
    // Возвращает узловую оценку скорости распространения волн для ограничения CFL.
    AERO_HOST_DEVICE static double wave_speed(unsigned dim, const double* velocity, double sound) {
        return ::sqrt(euler_detail::dot(dim, velocity, velocity)) + sound;
    }
    // min по ячейкам h / max(|u|+c) равен min по узлам h_min / (|u|+c).
    // Поэтому каждый шаг читает одну высоту на узел, без повторного обхода ячеек.
    AERO_HOST_DEVICE CflStep height_step(Index i, double courant, double viscous_rate) const {
        double speed = linear_speed;
        if (gas) {
            if (!(gas[i].density > 0.)) return {-1., i};
            speed = wave_speed(dimension, gas[i].velocity, gas[i].sound_speed);
        } else if (state && linear_speed==0.) {
            // При тесном размещении нет кэша газа. Высота и формула CFL те же:
            // восстанавливаем скорость и звук из состояния только этого узла.
            EulerState local{};
            if (euler_state(dimension,state+i*(dimension+2),gamma,&local)!=EulerStatus::Success) return {-1.,i};
            speed=wave_speed(dimension,local.velocity,local.sound_speed);
        }
        if (!(speed > 0. && speed <= max_finite_double &&
              viscous_rate >= 0. && viscous_rate <= max_finite_double)) return {-1., i};
        // Вязкая часть уже собрана как сумма по площадкам; переводим её в обратное время.
        const double rate = speed / heights[i] + (0.5 * viscous_rate) / volumes[i];
        const double dt = courant / rate;
        return {dt > 0. && dt <= max_finite_double ? dt : -1., i};
    }
    // Вычисляет допустимый локальный шаг по газовым и диффузионным скоростям.
    AERO_HOST_DEVICE CflStep local(Index k, double courant) const {
        const Index i = owned ? owned[k] : k;
        const double volume = volumes[i];
        if (!owned &&
            (!(volume > 0 && volume <= max_finite_double) || offsets[i] == offsets[i + 1]))
            return {-1, i};
        double viscous_rate = 0.;
#ifdef AERO_HAS_VISCOSITY
        if (diffusion) viscous_rate = diffusion[i].cfl_rate;
#endif
        const auto gas_step = heights ? height_step(i, courant, viscous_rate)
                                      : cfl_detail::local(offsets, volume, courant, i, speeds, viscous_rate);
        if (!sa_rate) return gas_step;
        // Ограничения газа и SA действуют на один шаг. Сравниваем их до
        // редукции, чтобы одного обхода узлов и одного минимума MPI хватило для обоих.
        const double rate = sa_rate[i];
        if (!(rate >= 0. && rate <= max_finite_double)) return {-1., i};
        if (rate == 0.) return gas_step;
        const double transport_dt = courant / rate;
        return cfl_detail::minimum(gas_step, {transport_dt, i});
    }
};
// Один обход сохраняет шаг каждого владельца и собирает границы диапазона для отчёта.
// Пустая подобласть участвует в MPI-редукции с нейтральными границами.
struct StepRange { double minimum, maximum; MeshIndex limited, capped; };
// Вычисляет локальные псевдошаги и сводит их диапазон и ограничения.
struct LocalSteadySteps {
    // Тип результата объединения локальных экстремумов шага.
    using value_type = StepRange;
    // Узловая оценка шага по потокам газа.
    FluxCfl gas;
    // Множитель числа Куранта.
    double courant;
    AdaptiveStepChange change; // Пустое поле означает обычный шаг CFL.
    double* steps;         // Общий шаг газа и SA в нумерации узлов подобласти.
    // История и ограничения по изменению невязки.
    LocalRateLimit feedback;
    // Указывает, когда нужно начать новую историю узловых значений.
    bool initialize_history;

    // Задаёт нейтральный результат редукции диапазона шагов и счётчиков.
    AERO_HOST_DEVICE void init(StepRange& range) const { range = {max_finite_double, 0., 0, 0}; }
    // Объединяет частичные диапазоны, максимумы и признаки ограничений шага.
    AERO_HOST_DEVICE void join(volatile StepRange& range, const volatile StepRange& next) const {
        range.limited += next.limited;
        range.capped += next.capped;
        if (next.minimum < range.minimum) range.minimum = next.minimum;
        if (next.maximum > range.maximum) range.maximum = next.maximum;
    }
    // Верхний предел узла определяет выбранный метод CFL: по высоте ячеек
    // или по скоростям на прилегающих площадках. Оба учитывают вязкость и SA.
    AERO_HOST_DEVICE double proposed_step(Index row, Index node, double ceiling) const {
        // Без истории после запуска/recovery доступны только газовый и турбулентный CFL.
        if (!change.field) return ceiling;
        const double previous = steps[node];
        const double next = change.next(row, node, previous);
        return next > ceiling ? ceiling : next;
    }
    // Вычисляет шаг собственного узла и обновляет диапазон и счётчики ограничений.
    AERO_HOST_DEVICE void operator()(Index row, StepRange& range) const {
        const Index node = gas.owned ? gas.owned[row] : row;
        const double ceiling = gas.local(row, courant).dt;
        const double proposed = proposed_step(row,node,ceiling);
        const double previous = feedback.history ? steps[node]*change.fraction : 0.;
        double dt = feedback.limit(row,node,ceiling,previous,proposed,initialize_history);
        if (dt < proposed) ++range.limited;
        // После повтора с половинным dt потолок может сохраняться, хотя уже не уменьшает шаг.
        if (proposed > 0. && feedback.has_cap(row)) ++range.capped;
        if (!(dt > 0. && dt < max_finite_double)) dt = -1.;
        steps[node] = dt;
        if (dt < range.minimum) range.minimum = dt;
        if (dt > range.maximum) range.maximum = dt;
    }
};
// После MPI-редукции всем процессам известен один предел. Меняем только свои узлы.
struct LimitLocalSteps {
    // Номера узлов, шаги которых обновляет этот процесс.
    const MeshIndex* owned;
    // Массив изменяемых локальных шагов.
    double* steps;
    // Общая верхняя граница шага.
    double ceiling;
    // Ограничивает локальный шаг узла заданным общим потолком.
    AERO_HOST_DEVICE void operator()(Index row) const {
        const Index node = owned ? owned[row] : row;
        if (steps[node] > ceiling) steps[node] = ceiling;
    }
};
// За проход измеряем действительный разброс на рёбрах и максимум шага.
// changed нужен отдельно: отношение двух double может округлиться до заданной границы.
struct NeighbourStepRange {
    // Отношение шагов и максимальное значение для редукции.
    double ratio, maximum;
    // Признак изменения узлового шага при проходе.
    unsigned changed;
};
// Уменьшает шаги узлов по ограничениям соседних рёбер.
struct NeighbourStepSweep {
    // Собственные строки текущего процесса.
    const MeshIndex* owned;
    // Границы списков инцидентных площадок.
    IndexArray<Index> offsets;
    // Ориентированные площадки в строке узла.
    FaceIncidence incidence;
    // Концы и тип каждой площадки.
    SurfaceTable surfaces;
    // Граница между внутренними рёбрами и граничными площадками.
    Index edges;
    const double* previous; // Все читают один слой, включая полученные по MPI узлы.
    double* next;           // Поток пишет только шаг своего узла.
    // Допустимое отношение соседних шагов и минимум по всей сетке.
    double ratio, global_minimum;

    // Результат редукции одного прохода по соседям.
    using value_type = NeighbourStepRange;
    // Задаёт нейтральный результат редукции диапазона шагов и счётчиков.
    AERO_HOST_DEVICE void init(value_type& result) const { result = {1., 0., 0}; }
    // Объединяет частичные диапазоны, максимумы и признаки ограничений шага.
    AERO_HOST_DEVICE void join(volatile value_type& a, const volatile value_type& b) const {
        if (b.ratio > a.ratio) a.ratio = b.ratio;
        if (b.maximum > a.maximum) a.maximum = b.maximum;
        if (b.changed) a.changed = 1;
    }
    // Уменьшает шаг узла по соседям и обновляет статистику прохода.
    AERO_HOST_DEVICE void operator()(Index row, value_type& result) const {
        const Index node = owned ? owned[row] : row;
        const double step = previous[node];
        double smallest = step;
        // У физической граничной площадки только один узел. Второй конец есть
        // только у ребра расчётной сетки (в том числе после склейки периодики).
        for (Index k = offsets[node]; k < offsets[node + 1]; ++k) {
            const Index face = incidence[k].face();
            if (face >= edges) continue;
            const auto& edge = surfaces[face];
            const Index other = edge.begin == node ? edge.end : edge.begin;
            if (previous[other] < smallest) smallest = previous[other];
        }
        const double edge_ratio = step / smallest;
        if (edge_ratio > result.ratio) result.ratio = edge_ratio;
        // При ratio=1 сразу ставим единый глобальный dt, без распространения
        // минимума через всю сетку. Во всех остальных случаях читаем только соседей.
        const double ceiling = ratio == 1. ? global_minimum : ratio * smallest;
        const double limited = step > ceiling ? ceiling : step;
        next[node] = limited;
        if (limited < step) result.changed = 1;
        if (limited > result.maximum) result.maximum = limited;
    }
};
// Результат редукции находится на CPU. Номер узла при ошибке переводим по таблице раздела.
double flux_cfl_result(CflStep limit, const MeshIndex* global_ids);
// Находит минимальный шаг по скоростям потоков на CPU.
double cpu_flux_cfl(const FluxCfl&, double courant, const MeshIndex* global_ids = nullptr);
// Находит минимальный шаг по скоростям потоков на GPU.
double gpu_flux_cfl(const FluxCfl&, double courant, void* stream, const MeshIndex* global_ids = nullptr);
}
}
#endif
