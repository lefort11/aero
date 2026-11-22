#pragma once
#include "../viscosity/grad.h"

namespace aero {
namespace turbulence {
// Коэффициент переноса рабочей вязкости между двумя вершинами элемента.
// Периодические копии одного узла имеют одну неизвестную и не дают вклада.
// CPU использует эту формулу при подготовке строк и линий, GPU — на стадии расчёта.
AERO_HOST_DEVICE inline double diffusion_weight(const viscosity::GradientStencil& geometry,
        const viscosity::Cell& cell,unsigned i,unsigned j,double scale) {
    if (cell.node[i]==cell.node[j]) return 0.;
    return -geometry.heat(cell,i,j)*scale;
}
}
}
