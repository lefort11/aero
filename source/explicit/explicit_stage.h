#ifndef AERO_EXPLICIT_STAGE_H
#define AERO_EXPLICIT_STAGE_H

#include "../parallel/kernel_math.h"

namespace aero {
namespace explicit_detail {

// Коэффициенты одной стадии в форме Шу–Ошера. step_start всегда означает U в начале всего шага.
struct Stage {
    double time_fraction;   // Время вычисления невязки: t_n + time_fraction * dt.
    double original_weight; // Вес U в начале шага.
    double forward_weight;  // Вес результата шага Эйлера из текущего состояния стадии.

    AERO_HOST_DEVICE bool uses_step_start() const { return original_weight != 0; }

    // На поздних стадиях SSPRK3 добавляем долю исходного U_n к результату Эйлера из текущего поля.
    AERO_HOST_DEVICE double combine(double step_start, double forward) const {
        // Для Эйлера и первой стадии берём forward. Совпадающие значения сохраняем точно и при смешивании.
        if (!uses_step_start() || forward == step_start)
            return forward;
        return original_weight * step_start + forward_weight * forward;
    }
};
struct Stages {
    const Stage* data; // Таблица коэффициентов в порядке выполнения.
    unsigned count;    // Одна стадия у Эйлера, три у SSPRK3.
};

// Эйлер и первая стадия SSPRK3 начинают с одной и той же формулы при t_n.
constexpr Stage first_euler_stage() {
    return {0.0, 0.0, 1.0};
}
Stages euler_stages();
Stages ssprk3_stages();

// Обновляет одну компоненту одного узла. Невязка уже содержит площадь всех его граней.
AERO_HOST_DEVICE inline double forward_euler_value(double current, double residual, double volume,
                                                   double dt) {
    // Положительная невязка означает отток. Деление на объём переводит её в скорость изменения U.
    return current - dt * (residual / volume);
}

// Общая формула обычного решателя и MPI на CPU и CUDA. next может совпадать с текущей компонентой.
AERO_HOST_DEVICE inline bool advance_component(Stage stage, double step_start, double current,
                                               double residual, double volume, double dt, double& next) {
    const double forward = forward_euler_value(current, residual, volume, dt);
    next = stage.combine(step_start, forward);
    // Запрещаем принять стадию с NaN или бесконечностью.
    return next >= -max_finite_double && next <= max_finite_double;
}

// Вид массивов для применения одной стадии к плотному полю. Один и тот же обход компонент на CPU и CUDA.
struct FieldStageUpdate {
    Stage formula;            // Коэффициенты выбранной стадии.
    size_t components;        // Сколько последовательных компонент размещено в одном узле.
    const double* volumes;    // По одному неподвижному объёму на узел.
    const double* step_start; // Поле U в начале всего шага.
    const double* current;    // Поле, по которому посчитана невязка стадии.
    const double* residual;   // Сумма выходящих потоков по каждой компоненте.
    double dt;                // Шаг по времени одинаков для всех стадий.
    double* next;             // Буфер результата стадии. Может совпадать с current.

    AERO_HOST_DEVICE bool apply(size_t entry) const {
        // Поля хранятся по схеме [узел][компонента], а объём общий для всех компонент узла.
        return advance_component(formula, step_start[entry], current[entry], residual[entry],
                                 volumes[entry / components], dt, next[entry]);
    }
};

#ifdef AERO_WITH_CUDA
void launch_update(FieldStageUpdate update, size_t entries, int* invalid, void* stream);
#endif
}
}
#endif
