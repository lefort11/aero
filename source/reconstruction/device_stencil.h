#pragma once
#include "reconstruction.h"

namespace aero {
namespace detail {
// Копируем одну таблицу разреженных строк. Число номеров и весов задаётся
// последним смещением. При расчёте размера памяти allocate только считает байты.
template<class Storage>
void upload_stencil_rows(Storage& storage, ReconstructionIndex rows, const ReconstructionIndex*& offsets,
                         const ReconstructionIndex*& nodes, const double*& weights) {
    const auto entries = offsets ? offsets[rows] : 0;
    // Если все строки пусты, вместо массива одинаковых нулей достаточно nullptr.
    offsets = storage.allocate(entries ? rows + 1 : 0,offsets);
    nodes = storage.allocate(entries,nodes);
    weights = storage.allocate(entries,weights);
}
}

// Принимает копию описания на CPU и заменяет адреса адресами GPU. Владелец
// памяти определяет способ выделения: отдельные массивы или одна область подобласти.
template<class Storage>
ReconstructionStencilView upload_reconstruction_stencil(Storage& storage, ReconstructionStencilView view,
                                                         ReconstructionIndex exception_rows = 0) {
    const auto rows = 2 * view.edge_count;
    detail::upload_stencil_rows(storage,rows,view.offsets,view.nodes,view.weights);
    view.row_owner = storage.allocate(view.row_owner ? rows : 0,view.row_owner);

    // У WENO и MP на каждую сторону отведено четыре строки наклонов.
    // Линейная схема использует только первую таблицу и обходится без второй.
    const auto nonlinear_rows = view.weno_order ? rows : 0;
    if (!nonlinear_rows) view.slope_offsets = nullptr;
    const auto rows_to_copy = view.pairs ? exception_rows : 4 * nonlinear_rows;
    view.pairs = storage.allocate(view.pair_count(),view.pairs);
    detail::upload_stencil_rows(storage,rows_to_copy,view.slope_offsets,view.slope_nodes,view.slope_weights);
    view.weno_order = storage.allocate(nonlinear_rows,view.weno_order);
    view.weno_epsilon_scale = storage.allocate(view.epsilon_per_edge ? nonlinear_rows/2 : nonlinear_rows,
                                               view.weno_epsilon_scale);
    return view;
}
}
