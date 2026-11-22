#pragma once
#include "../mesh/mesh_index.h"

namespace aero {
namespace residual_detail {
// По внутренней поверхности поток выходит из begin в end. Физической площадке
// принадлежит только узел begin и номер метки patch.
struct Surface {
    MeshIndex begin, end, patch;
    double area[3]; // Вектор площади в координатах решателя.
    double measure; // Длина этого вектора.
    unsigned char use_nodal_traces; // Маска перехода на первый порядок для сторон ребра.
};
}
}
