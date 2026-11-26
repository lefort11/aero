#pragma once
#include "../parallel/row_loop.h"
#include "explicit_stage.h"

namespace aero {
namespace explicit_detail {
// Обновляет все газовые компоненты стадии. Для допустимого поля требуется только чтение
// исходной компоненты и запись результата. Об ошибке сообщает общий флаг исполнителя.
template<class Space>
void update_field(const Space& exec, const FieldStageUpdate& step, MeshIndex count, int* bad) {
    row_loop::for_indices(exec,"Aero: explicit stage",count,KOKKOS_LAMBDA(MeshIndex k) {
        if (!step.apply(k)) Kokkos::atomic_exchange(bad,1);
    });
}
}
}
