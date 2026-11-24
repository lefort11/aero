#pragma once
#include "sa.h"
#include "../residual/residual_detail.h"

namespace aero {
namespace turbulence {
// Перенос SA использует те же рёбра и те же физические границы, что и газ.
// Ориентируем площадь наружу для текущего узла. Половина площади нужна для
// средней скорости на внутренней грани, на внешней границе используется вся площадь.
AERO_HOST_DEVICE inline FaceTerm make_face_term(const residual_detail::Surface &face,
                                                residual_detail::OrientedFace side, MeshIndex node,
                                                MeshIndex edge_count, unsigned dim) {
    const bool interior = side.face() < edge_count;
    const auto neighbour = interior ? (face.begin == node ? face.end : face.begin) : node;
    FaceTerm term{side.encoded, neighbour, {}};
    const double weight = side.outward_flux(interior ? .5 : 1.);
    for (unsigned axis = 0; axis < dim; ++axis)
        term.area[axis] = weight * face.area[axis];
    return term;
}

// На тесной карте читаем геометрию газа на месте. Смещения задают исходный
// порядок граней в строке, поэтому порядок суммирования в уравнении SA сохраняется.
struct GasFaces {
    const residual_detail::Surface *faces;
    const residual_detail::OrientedFace *incidence;
    const residual_detail::Index *offsets;
    // Возвращает диапазон площадок данного узла в CSR-таблице.
    AERO_HOST_DEVICE RowRange row(MeshIndex node) const { return {offsets[node], offsets[node + 1]}; }
    // Формирует ориентированный вклад площадки в конвективный оператор SA.
    AERO_HOST_DEVICE FaceTerm read(MeshIndex node, MeshIndex entry, MeshIndex edges, unsigned dim) const {
        const auto side = incidence[entry];
        return make_face_term(faces[side.face()], side, node, edges, dim);
    }
};
}
}
