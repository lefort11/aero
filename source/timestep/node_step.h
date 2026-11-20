#pragma once
#include "../parallel/kernel_math.h"

namespace aero {
// В обычном режиме параметр равен общему dt. При локальном установлении
// он задаёт долю узлового шага CFL (единица до ограничения расписанием).
struct NodeSteps {
    double* values;
    AERO_HOST_DEVICE double at(size_t node, double dt_or_fraction) const {
        return values ? values[node] * dt_or_fraction : dt_or_fraction;
    }
};

// На CPU хранится описание режима. Память под шаги принадлежит исполнителю.
// Перед запуском ядро получает только адрес массива, действующий на этом шаге.
struct SteadyNodeSteps {
    bool local = false;
    double max_ratio = 4.;
    double* values = nullptr;
    static NodeSteps active(const SteadyNodeSteps* steps) {
        return {steps && steps->local ? steps->values : nullptr};
    }
};
}
