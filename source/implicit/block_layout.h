#pragma once
#include "index_list.h"
#include "../parallel/kernel_math.h"
#include <cfloat>
#include <type_traits>

namespace aero {
namespace implicit {
constexpr bool mixed_linear = AERO_MIXED_LINEAR != 0;
using LinearCoefficient = std::conditional<mixed_linear,float,double>::type;
static_assert(mixed_linear || std::is_same<LinearCoefficient,double>::value,
              "MIXED_LINEAR=0 requires double coefficients");
// Блок 5×5 (или меньший по размерности). В CSR коэффициенты соседствуют,
// в раскладке CSGS между ними лежат коэффициенты других строк того же пакета.
struct BlockView {
    // Адрес первого элемента матричного блока.
    double* first;
    // Расстояние между соседними элементами блока в памяти.
    unsigned step;
    LinearCoefficient* neighbours; // Отдельное хранение коэффициентов соседних блоков.
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE double operator[](unsigned k) const {
        return neighbours ? static_cast<double>(neighbours[k*step]) : first[k*step];
    }
    // Записывает коэффициент блока с учётом шага упакованного хранения.
    AERO_HOST_DEVICE void set(unsigned k,double value) const {
        if (neighbours) neighbours[k*step] = mixed_linear && ::fabs(value) > FLT_MAX
                ? static_cast<LinearCoefficient>(HUGE_VAL) : static_cast<LinearCoefficient>(value);
        else first[k*step] = value;
    }
    // Добавляет вклад к коэффициенту блока с учётом его раскладки.
    AERO_HOST_DEVICE void add(unsigned k,double value) const { set(k, (*this)[k]+value); }
};

// Адреса и шаг хранения коэффициентов разреженных матричных блоков.
struct BlockLayout {
    IndexList first;         // Адрес первого коэффициента каждого блока CSR в общей области.
    unsigned lanes;          // Шаг между коэффициентами одного внедиагонального блока.
    LinearCoefficient* neighbours; // Отдельная область соседей. Диагональ целиком остаётся в double.

    // Возвращает представление выбранного матричного блока.
    AERO_HOST_DEVICE BlockView view(double* values, MeshIndex slot, unsigned width,
                                    bool diagonal) const {
        if (!first) return {values + slot * width * width, 1, nullptr};
        if (!diagonal && neighbours) return {nullptr, lanes, neighbours + first[slot]};
        return {values + first[slot], diagonal ? 1u : lanes, nullptr};
    }
};
}
}
