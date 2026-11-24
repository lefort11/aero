#pragma once
#include "../parallel/kernel_math.h"

namespace aero {
// В обычном режиме параметр равен общему dt. При локальном установлении
// он задаёт долю узлового шага CFL (единица до ограничения расписанием).
struct NodeSteps {
    // Массив узловых шагов при локальном режиме.
    double* values;
    // Возвращает шаг выбранного узла либо общий шаг при отсутствии локального массива.
    AERO_HOST_DEVICE double at(size_t node, double dt_or_fraction) const {
        return values ? values[node] * dt_or_fraction : dt_or_fraction;
    }
};

// На CPU хранится описание режима. Память под шаги принадлежит исполнителю.
// Перед запуском ядро получает только адрес массива, действующий на этом шаге.
struct SteadyNodeSteps {
    // Выбран ли локальный псевдошаг.
    bool local = false;
    // Допустимое отношение максимального шага к минимальному.
    double max_ratio = 0.;
    // Узловые значения активного локального шага.
    double* values = nullptr;
    // Возвращает локальные шаги только при включённом локальном режиме.
    static NodeSteps active(const SteadyNodeSteps* steps) {
        return {steps && steps->local ? steps->values : nullptr};
    }
};
}
