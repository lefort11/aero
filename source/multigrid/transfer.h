#pragma once
#include "../timestep/node_step.h"
#include "../mesh/mesh_index.h"
#include "../eulersolver/euler.h"

namespace aero {
namespace multigrid {
// Эти операции компилируются для обоих исполнителей. Одна задача обрабатывает
// одну компоненту агрегата. Записи разных задач не пересекаются.
struct Transfer {
    enum Operation { RestrictResidual, RestrictProduct, Prolong, SmoothingRhs, SaveField, TrialField,
                     HistoryDirection, HistoryTrial, CheckGas };
    const MeshIndex* offsets; // Диапазон собственных строк агрегата.
    const MeshIndex* members; // Строки в компактной нумерации Ньютона.
    const MeshIndex* owned;   // Перевод строки в узел массива состояния, если работает MPI.
    const unsigned char* wall; // Маска прилипания. У закреплённого импульса поправка равна нулю.
    unsigned width;           // Плотность, компоненты импульса и энергия переносятся единым блоком.
    const double* volumes;
    NodeSteps steps;          // Те же узловые псевдошаги, что при последней сборке матрицы.
    double dt;                // Глобальный шаг или общий множитель локальных шагов.

    Operation operation;
    double* coarse;           // Одна компонента поправки или суммы невязок на агрегат.
    double* fine;             // Компактный вектор исходной сетки.
    double* product;          // Образ поправки под действием матрицы Ньютона.
    double* state;            // Полное состояние, включая узлы обмена.
    double* inverse_mass;     // Диагональное масштабирование грубой правой части для поиска направления.
    double* smoother;         // Исходная невязка, затем поправка сглаживателя на мелкой сетке.
    const double* start;      // Поле до обычного мелкосеточного шага в нумерации собственных строк.
    double fraction;          // Множитель агрегатной поправки или изменения газа за обычный псевдошаг.
    double gamma;             // Отношение теплоёмкостей для проверки допустимости пробы Эйлера.

    // У Ньютона матрица содержит временную диагональ, у стационарной задачи — только J.
    // Для одного узла используем ту же массу и тот же псевдошаг, что при сборке матрицы.
    AERO_HOST_DEVICE double spatial_product(MeshIndex index, MeshIndex node) const {
        return product[index] - (volumes[node]/steps.at(node,dt))*fine[index];
    }

    // Вычисляет одну компоненту ограниченного или продлённого вектора.
    AERO_HOST_DEVICE void operator()(MeshIndex i) const {
        const MeshIndex aggregate = i / width;
        const unsigned k = static_cast<unsigned>(i % width);
        double sum = 0., mass = 0.;
        for (MeshIndex j = offsets[aggregate]; j < offsets[aggregate+1]; ++j) {
            const MeshIndex row = members[j], node = owned ? owned[row] : row;
            const MeshIndex index = row*width+k;
            const bool fixed = wall && wall[node] && k > 0 && k+1 < width;
            switch (operation) {
            case CheckGas:
                if (k == 0) {
                    EulerState gas;
                    if (euler_state(width-2,state+node*width,gamma,&gas) != EulerStatus::Success) sum = 1.;
                }
                break;
            case RestrictResidual:
                // Для постсглаживания нужна невязка после обычного шага.
                smoother[index] = fine[index];
                if (!fixed) {
                    sum += fine[index]; // Интегральная невязка уже содержит площадь каждой грани.
                    mass += volumes[node] / steps.at(node, dt);
                }
                break;
            case RestrictProduct:
                // Из (J + V/dt) P e вычитаем временную часть до суммирования.
                if (!fixed) sum += spatial_product(index,node);
                break;
            case Prolong: fine[index] = fixed ? 0. : coarse[i]; break;
            case HistoryDirection:
                fine[index] = fixed ? 0. : state[node*width+k]-start[index];
                break;
            case SmoothingRhs:
                // После продолжения e остаётся дефект b−JPe. Именно его должен
                // погасить сглаживатель, включая скачки на стыках агрегатов.
                fine[index] = fixed ? 0. : smoother[index] - spatial_product(index,node);
                break;
            case SaveField:
                fine[index] = state[node*width+k];
                break;
            case HistoryTrial:
            case TrialField: {
                // Каждую пробу строим заново от принятого газа. При нулевой доле
                // восстанавливаем исходные биты даже при переполнении поправки.
                double value = fine[index];
                if (!fixed && fraction != 0.) {
                    const double change = operation == HistoryTrial ? value-start[index] : coarse[i]+smoother[index];
                    value += fraction*change;
                }
                state[node*width+k] = value;
                break;
            }
            }
        }
        if (operation == RestrictResidual) inverse_mass[i] = mass > 0. ? 1./mass : 0.;
        if (operation == RestrictResidual || operation == RestrictProduct) coarse[i] = sum;
        if (operation == CheckGas) coarse[i] = sum;
    }
};

// Общие действия над короткими векторами грубого уровня.
struct VectorUpdate {
    enum Operation { Clear, ScaleResidual, Add };
    Operation operation;
    double* target;          // Вектор, который обновляем.
    const double* source;    // Направление поиска или невязка.
    const double* diagonal;  // Обратная масса агрегата для предварительного масштабирования.
    double factor;           // Коэффициент при добавляемом направлении.
    // Обновляет одну компоненту рабочего вектора грубосеточной коррекции.
    AERO_HOST_DEVICE void operator()(MeshIndex i) const {
        switch (operation) {
        case Clear: target[i] = 0.; break;
        case ScaleResidual: target[i] = source[i]*diagonal[i]; break;
        case Add: target[i] += factor*source[i]; break;
        }
    }
};
}
}
