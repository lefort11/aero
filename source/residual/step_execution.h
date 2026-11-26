#pragma once
#include <Kokkos_Core.hpp>
#include "flux_cfl.h"

namespace aero {
namespace residual_detail {
// Вычисляет локальные псевдошаги и одновременно собирает их диапазон.
// Диапазон Kokkos приходит от вызывающей стороны вместе с её потоком исполнения.
template<class Range>
inline void collect_local_steps(const Range& rows, const LocalSteadySteps& steps, StepRange& result) {
    Kokkos::parallel_reduce("Aero: local steady steps", rows, steps, result);
}
// Применяет верхнюю границу к уже рассчитанным узловым шагам за один проход.
template<class Range>
inline void cap_local_steps(const Range& rows, const LimitLocalSteps& limit) {
    Kokkos::parallel_for("Aero: limit local steady steps", rows, limit);
}
// За один обход ограничивает отношение соседних шагов и собирает признаки изменения.
template<class Range>
inline void sweep_neighbour_steps(const Range& rows, const NeighbourStepSweep& sweep, NeighbourStepRange& result) {
    Kokkos::parallel_reduce("Aero: neighbouring local steps", rows, sweep, result);
}
}
}
