#ifndef AERO_CFL_DETAIL_H
#define AERO_CFL_DETAIL_H
#include "../parallel/kernel_math.h"
#include "timestep.h"
#ifndef KOKKOS_ENABLE_CUDA
#include "../parallel/kokkos_host.h"
#endif

namespace aero {
namespace cfl_detail {
// Номер площадки и её площадь в списке связей узла.
struct Entry {
    // Номер площадки, ограничивающей данный объём.
    size_t face;
    // Мера площадки, используемая в оценке CFL.
    double area;
};
// Невладеющая CSR-геометрия площадок и объёмов для оценки шага CFL.
struct Grid {
    // Количество узловых контрольных объёмов.
    size_t nodes;
    // Начала строк списка площадок.
    const size_t* offsets;
    // Площадки строк в порядке обхода.
    const Entry* entries;
    // Меры контрольных объёмов.
    const double* volumes;
};
// Возвращает нейтральное ограничение шага для начала минимизации.
AERO_HOST_DEVICE inline CflStep unconstrained() {
    return {max_finite_double, static_cast<size_t>(-1)};
}
// При одинаковом dt выбираем узел с меньшим номером. Результат одинаков при любом порядке редукции.
AERO_HOST_DEVICE inline CflStep minimum(CflStep a, CflStep b) {
    return (b.dt < a.dt || (b.dt == a.dt && b.node < a.node)) ? b : a;
}
// Переводит сумму спектральных скоростей площадок в допустимый узловой шаг.
AERO_HOST_DEVICE inline CflStep from_sum(double sum, double volume, double cfl, size_t node) {
    if (sum == 0)
        return unconstrained();
    if (sum > max_finite_double)
        return {-1, node};
    // В обычном диапазоне binary64 считаем 2·CFL·V / Σ напрямую. Перенос степени
    // двойки за умножение даёт то же округление, что и расчёт по мантиссам ниже.
    // У краёв диапазона оставляем запас: субнормальные промежуточные результаты могут терять биты.
    const double product = cfl * volume;
    const double normal_floor = 2 * 2.2250738585072014e-308;
    if (product >= normal_floor && product <= max_finite_double / 2) {
        const double dt = (2 * product) / sum;
        if (dt >= normal_floor && dt < max_finite_double)
            return {dt, node};
    }
    // Складываем показатели степеней отдельно, чтобы промежуточное произведение 2·CFL·V не переполнилось до деления.
    int ec, ev, es;
    const double mc = ::frexp(cfl, &ec), mv = ::frexp(volume, &ev), ms = ::frexp(sum, &es);
    const double dt = ::ldexp(mc * mv / ms, 1 + ec + ev - es);
    if (dt > max_finite_double)
        return unconstrained();
    return {dt > 0 ? dt : -1, node};
}
// Скорость волн на площадке и модуль её площади.
struct FaceSpeed {
    // Скорость волны и мера соответствующей площадки.
    double speed, area;
};
// Суммирует по всем площадкам узла произведения скорости на площадь. Ориентация потока для CFL роли не играет.
template <class Offset, class ReadFace>
AERO_HOST_DEVICE inline CflStep local(Offset offsets, double volume, double cfl, size_t node,
                                      ReadFace read_face, double sum = 0.) {
    if (!(sum >= 0. && sum <= max_finite_double)) return {-1, node};
    for (auto j = offsets[node]; j < offsets[node + 1]; ++j) {
        const FaceSpeed face = read_face(j);
        if (!(face.speed >= 0 && face.speed <= max_finite_double))
            return {-1, node};
        sum += face.speed * face.area;
    }
    return from_sum(sum, volume, cfl, node);
}
// Читает скорость волн и площадь по записи связи узла с площадкой.
struct GridSpeeds {
    // Список площадок, по которому читаются скорости.
    const Entry* entries;
    // Текущие волновые скорости газа.
    WaveSpeeds speeds;
    // Читает скорость волны внутренней или внешней площадки узла.
    AERO_HOST_DEVICE FaceSpeed operator()(size_t j) const {
        const Entry e = entries[j];
        return {e.face < speeds.edge_count ? speeds.edge[e.face]
                                           : speeds.boundary[e.face - speeds.edge_count],
                e.area};
    }
};
// Суммирует по всем площадкам узла произведения скорости на площадь. Ориентация потока для CFL роли не играет.
AERO_HOST_DEVICE inline CflStep local(Grid grid, WaveSpeeds speeds, double cfl, size_t node) {
    return local(grid.offsets, grid.volumes[node], cfl, node, GridSpeeds{grid.entries, speeds});
}
// Пара (dt, узел) остаётся целой при объединении результатов потоков.
// Тот же minimum выбирает узел при равенстве шагов и отдаёт приоритет ошибкам.
template <class ReadNode> struct MinimumStep {
    // Тип результата редукции минимального шага с номером узла.
    using value_type = CflStep;
    ReadNode read_node; // Возвращает ограничение CFL для одного узла текущей стадии.
    // Начинает редукцию с отсутствия ограничения на шаг.
    AERO_HOST_DEVICE void init(CflStep& best) const { best = unconstrained(); }
    // Сравнивает шаг одного узла с текущим минимумом редукции.
    AERO_HOST_DEVICE void operator()(size_t node, CflStep& best) const { best = minimum(best, read_node(node)); }
    // Выбирает меньший шаг из двух частичных результатов редукции.
    AERO_HOST_DEVICE void join(volatile CflStep& best, const volatile CflStep& next) const {
        const CflStep chosen = minimum({best.dt, best.node}, {next.dt, next.node});
        best.dt = chosen.dt;
        best.node = chosen.node;
    }
};
#ifndef KOKKOS_ENABLE_CUDA
// Находит минимальный допустимый шаг по узлам на CPU.
template <class ReadNode> CflStep host_min(size_t nodes, const ReadNode& read_node) {
    CflStep result = unconstrained();
    // На малой сетке время запуска команды дороже обхода. Правило выбора шага общее.
    const MinimumStep<ReadNode> reduce{read_node};
    if (nodes < 4096) {
        for (size_t node = 0; node < nodes; ++node)
            reduce(node, result);
    } else {
        host_parallel::ensure_runtime();
        Kokkos::parallel_reduce("Aero: CFL minimum", host_parallel::Range(0, nodes), reduce, result);
    }
    return result;
}
#endif
// Объект хранит на устройстве сетку и рабочую память для поиска минимума между вызовами.
void* upload(Grid grid, int device);
// Освобождает GPU-хранилище геометрии оценки CFL.
void release(void* storage) noexcept;
// Вычисляет минимальный шаг по узлам с помощью выбранного исполнителя.
CflStep reduce(void* storage, WaveSpeeds speeds, double cfl, void* stream);
}
}
#endif
