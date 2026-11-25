#ifndef AERO_IMPLICIT_NEWTON_VECTORS_H
#define AERO_IMPLICIT_NEWTON_VECTORS_H
#include "../timestep/node_step.h"
#include <Kokkos_Core.hpp>
#include "../residual/residual_detail.h"

namespace aero {
namespace implicit {
// Поля текущего шага. Один вызов обрабатывает все компоненты одного узла.
struct NewtonUpdate {
    // Операции над узловыми полями на этапах решения уравнения Ньютона.
    enum Operation { InitialRhs, FormRhs, AddCorrection, PredictCorrection, RestoreStart, Predict, RelaxPrediction, UndoPrediction, SpatialRate };
    // Связность для сборки узловой невязки.
    residual_detail::Geometry grid;
    const EulerFluxResult* flux; // Потоки в тех же ориентациях, что при явном обновлении.
    // Объёмы узлов, масштабирующие временной вклад.
    const double* volumes;
    // Поле, сохранённое начало шага, правая часть и поправка Ньютона.
    double *state, *start, *rhs, *correction;
    const double* previous;  // U_(n-1), хранится в компактной нумерации собственных узлов.
    double dt; // Шаг, на котором решается уравнение баланса.
    double leading, history_weight; // Коэффициенты разностей (U-U_n) и (U_n-U_(n-1)).
    // Выбранное преобразование узловых векторов.
    Operation operation;
    const MeshIndex* owned; // Номера узлов в полном поле для компактных строк решателя.

    double relaxation; // Доля пробной поправки при ослаблении вязкого предиктора.

    // Узловые псевдошаги текущей сборки.
    NodeSteps local_steps;
    // Собираем баланс узла и сразу определяем наибольшую по модулю компоненту.
    // spatial[k] содержит сумму исходящих потоков через все грани контрольного объёма.
    template<unsigned width>
    AERO_HOST_DEVICE double residual_norm(MeshIndex row) const {
        const MeshIndex node = owned ? owned[row] : row;
        double spatial[width] = {};
        const bool valid = residual_detail::assemble_node_residual<width>(grid, node, flux, spatial);
        double largest = 0.;
        for (unsigned k = 0; k < width; ++k) {
            double value;
            if (operation == SpatialRate) {
                // Критерий выхода на стационар использует скорость изменения поля.
                value = valid ? spatial[k] / volumes[node] : HUGE_VAL;
                // Грубая система суммирует подписанные интегральные балансы.
                // Нормировка на объём нужна только критерию останова steady.
                if (rhs) rhs[row*width+k] = valid ? -spatial[k] : HUGE_VAL;
            } else {
                const MeshIndex i = row * width + k;
                const double current = state[node * width + k];
                // На первой итерации копируем принятое поле для баланса BDF и отката.
                // Ровно тот же узел уже понадобился для вычисления правой части.
                if (operation == InitialRhs) start[i] = current;
                double change = leading * (current - start[i]);
                if (history_weight != 0.) change -= history_weight * (start[i] - previous[i]);
                value = valid ? -((volumes[node] / local_steps.at(node, dt)) * change + spatial[k]) : HUGE_VAL;
                rhs[i] = value;
            }
            // Любая неконечная компонента должна прервать принятие шага на всех процессах.
            accumulate_infinity_norm(value, largest);
        }
        // После принятия слоя вектор поправки свободен. Оставляем узловую
        // норму для следующего LTS, не пересчитывая и не копируя потоки.
        if (operation == SpatialRate && correction) correction[row] = largest;
        return largest;
    }

    // Вносит уже найденную поправку в поле либо восстанавливает сохранённое состояние.
    AERO_HOST_DEVICE void operator()(MeshIndex row) const {
        const MeshIndex node = owned ? owned[row] : row;
        const unsigned width = grid.components();
        for (unsigned k = 0; k < width; ++k) {
            const MeshIndex i = row * width + k;
            double& current = state[node * width + k];
            switch (operation) {
            case InitialRhs:
            case FormRhs:
            case SpatialRate: return; // Эти операции обслуживает residual_norm().
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
            }
        }
    }
};

// Отрицательную невязку нужно и сохранить для решателя, и проверить по норме.
// Один исполнитель обрабатывает узел целиком. Норма по процессу получается
// при том же обходе, без повторного чтения всего вектора из памяти карты.
template<unsigned width, class Execution>
double form_residual_and_norm(const Execution& exec, MeshIndex rows, NewtonUpdate update) {
    if (!rows) return 0.;
    using Range = Kokkos::RangePolicy<Execution,Kokkos::IndexType<MeshIndex>>;
    const auto range = Kokkos::Experimental::require(Range(exec,0,rows),
                                  Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    double norm = 0.;
    Kokkos::parallel_reduce("Aero: Newton residual and norm",range,KOKKOS_LAMBDA(MeshIndex row,double& largest) {
        const double node_norm = update.residual_norm<width>(row);
        if (node_norm > largest) largest = node_norm;
    },Kokkos::Max<double>(norm));
    return norm;
}

// На одной сетке все строки имеют одну ширину. Выбираем её до запуска,
// чтобы исполнитель складывал компоненты без проверки размерности на каждой грани.
template<class Execution>
double form_residual_and_norm(const Execution& exec, MeshIndex rows, NewtonUpdate update) {
    switch (update.grid.dim) {
    case 1: return form_residual_and_norm<3>(exec, rows, update);
    case 2: return form_residual_and_norm<4>(exec, rows, update);
    default: return form_residual_and_norm<5>(exec, rows, update);
    }
}

}
}
#endif
