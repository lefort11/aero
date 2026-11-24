#pragma once
#include "sigma.h"
#include "../mesh/mesh_index.h"
#include <cstdint>

namespace aero {
namespace turbulence {
// Геометрия деформированных гексаэдров для повторных расчётов Sigma и DDES.
// На CPU коэффициенты одного элемента лежат подряд. На GPU соседние исполнители
// читают один и тот же коэффициент своих элементов из соседних адресов.
struct HexFilterData {
    enum { Volume, VolumeLength, GradientSum, OtherDiameter, Basis, Diagonals=Basis+7*3, Coefficients=Diagonals+4*3 };
    const std::uint32_t* nodes;
    const double* coefficients;
    unsigned lanes;

    // Вычисляет начало данных элемента в чередующейся раскладке GPU.
    AERO_HOST_DEVICE MeshIndex first(MeshIndex element, unsigned columns) const {
        return (element/lanes)*columns*lanes + element%lanes;
    }
};

// Один элемент в подготовленной таблице. Общая формула фильтра обращается
// к полной геометрии, когда четыре длиннейшие пары не определяют диаметр.
template<class Shape>
// Представление упакованной геометрии гексаэдра для фильтра LES/DDES.
struct PackedHexFilter {
    const double* column;
    unsigned stride;
    const Shape* shapes;
    const MeshIndex* shape_ids;
    MeshIndex element;
    double vol_length;

    // Читает коэффициент фильтра с учётом шага упакованного хранения.
    AERO_HOST_DEVICE double value(unsigned index) const { return column[index*stride]; }
    // Возвращает компоненту среднего градиента базиса вершины гексаэдра.
    AERO_HOST_DEVICE double basis(unsigned vertex,unsigned direction) const {
        return value(HexFilterData::Basis+3*(vertex-1)+direction);
    }
    // stride равен единице на CPU. На GPU соседние элементы одного блока
    // читают одну компоненту диагонали из соседних адресов.
    AERO_HOST_DEVICE Coordinate long_diagonal(unsigned k) const {
        Coordinate diagonal;
        for (unsigned i=0; i<3; ++i) diagonal[i]=value(HexFilterData::Diagonals+3*k+i);
        return diagonal;
    }
    // Возвращает квадрат геометрического масштаба фильтра.
    AERO_HOST_DEVICE double length_squared(Scale scale, unsigned count, const Coordinate& axis) const {
        if (scale==Scale::Min || scale==Scale::Max)
            return shapes[shape_ids[element]].length_squared(scale,count,axis);
        const double square=dot3(axis,axis);
        if (scale==Scale::Vol || square==0.) return vol_length*vol_length;
        const double remaining=value(HexFilterData::OtherDiameter);
        if (remaining>=0. && (scale==Scale::Sla || scale==Scale::OmegaTilde)) {
            double filter2;
            if (try_transverse_filter(*this,axis,remaining,filter2)) return filter2;
            // Упакованные диагонали уже проверены. Остальные вершины доступны
            // в полной геометрии, где можно сразу вычислить точный диаметр.
            return shapes[shape_ids[element]].full_transverse_filter(count,axis);
        }
        return shapes[shape_ids[element]].length_squared(scale,count,axis);
    }
};
using HexFilter=PackedHexFilter<FilterCell>;
}
}
