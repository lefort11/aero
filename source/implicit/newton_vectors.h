#ifndef AERO_IMPLICIT_NEWTON_VECTORS_H
#define AERO_IMPLICIT_NEWTON_VECTORS_H
#include "../timestep/node_step.h"
#include "../residual/residual_detail.h"

namespace aero {
namespace implicit {
// Поля текущего шага. Один вызов обрабатывает все компоненты одного узла.
struct NewtonUpdate {
    enum Operation { SaveStart, FormRhs, AddCorrection, PredictCorrection, RestoreStart, Predict, RelaxPrediction, UndoPrediction, SpatialRate };
    residual_detail::Geometry grid;
    const EulerFluxResult* flux; // Потоки в тех же ориентациях, что при явном обновлении.
    const double* volumes;
    double *state, *start, *rhs, *correction;
    const double* previous;  // U_(n-1), хранится в компактной нумерации собственных узлов.
    double dt; // Шаг, на котором решается уравнение баланса.
    double leading, history_weight; // Коэффициенты разностей (U-U_n) и (U_n-U_(n-1)).
    Operation operation;
    const MeshIndex* owned; // Номера узлов в полном поле для компактных строк решателя.

    double relaxation; // Доля пробной поправки при ослаблении вязкого предиктора.

    NodeSteps local_steps;
    AERO_HOST_DEVICE void operator()(MeshIndex row) const {
        const MeshIndex node = owned ? owned[row] : row;
        const unsigned width = grid.components();
        double spatial[5];
        // Суммируем исходящие потоки один раз для всех уравнений узла.
        const bool needs_flux = operation == FormRhs || operation == SpatialRate;
        const bool valid = !needs_flux || residual_detail::assemble_node_residual(grid, node, flux, spatial);
        if (operation == SpatialRate) {
            // Деление на объём даёт скорость изменения поля. Размер контрольного
            // объёма и величина псевдошага не входят в критерий стационарности.
            for (unsigned k = 0; k < width; ++k)
                rhs[row * width + k] = valid ? spatial[k] / volumes[node] : HUGE_VAL;
            return;
        }
        for (unsigned k = 0; k < width; ++k) {
            const MeshIndex i = row * width + k;
            double& current = state[node * width + k];
            switch (operation) {
            case SpatialRate: break; // Все компоненты уже обработаны выше.
            case SaveStart: start[i] = current; break;
            case RestoreStart: current = start[i]; break;
            case Predict:
                // Один шаг последовательных приближений к балансу BDF. Обратная
                // временная диагональ равна dt/(a*V). Сохраняем состояние до пробы.
                correction[i] = current;
                current += (local_steps.at(node, dt)/leading/volumes[node])*rhs[i];
                break;
            case RelaxPrediction:
                current = correction[i] + relaxation * (current - correction[i]);
                break;
            case UndoPrediction: current = correction[i]; break;
            case PredictCorrection:
            case AddCorrection: {
                const double change=correction[i];
                if (operation==PredictCorrection) correction[i]=current;
#ifdef AERO_HAS_VISCOSITY
                // Приближённая матрица Ньютона может связать стенку с соседями.
                // Предписанное значение импульса сохраняется при каждой поправке.
                if (grid.options.wall && k > 0 && k+1 < width && grid.options.wall[node]) break;
#endif
                current += change; break;
            }
            case FormRhs:
                // Правая часть A*delta = -F(U) включает накопление за шаг и чистый отток.
                // Форма через разности точно обнуляет временную часть на постоянном поле.
                // На первом шаге предыдущего слоя нет, используем неявный Эйлер.
                double change = leading * (current - start[i]);
                if (history_weight != 0.) change -= history_weight * (start[i] - previous[i]);
                rhs[i] = valid ? -((volumes[node] / local_steps.at(node, dt)) * change + spatial[k]) : HUGE_VAL;
                break;
            }
        }
    }
};
}
}
#endif
