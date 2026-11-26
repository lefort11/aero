#pragma once
#include "../parallel/row_loop.h"
#include "partition_kernels.h"

namespace aero {
namespace partition_detail {
using row_loop::for_indices;
// Резервирует место в списке отказавших узлов на активном исполнителе.
KOKKOS_INLINE_FUNCTION void record_failed_node(const partition_detail::Grid& g, Index node) {
    // Место в списке резервирует только отказавший узел. Ёмкость равна числу владельцев.
    const Index slot = Kokkos::atomic_fetch_add(g.failed_count, Index(1));
    g.failed_nodes[slot] = node;
}

// Запускает поузловое обновление поля на выбранном контексте Kokkos.
template <unsigned width, bool fused, class Space>
void update_nodes(const Space& exec, partition_detail::Grid g, explicit_detail::Stage stage,
                  double dt, bool save, const viscosity::Operator* material, const turbulence::View* sa) {
    for_indices(exec, "Aero: owned nodes", g.owned_count, KOKKOS_LAMBDA(Index k) {
        if (!partition_detail::advance_owned_node<width,fused>(g.geometry, g.flux, g.volumes, g.owned, g.state,
                                                              g.original, k, stage, dt, save, material, sa)) {
            Kokkos::atomic_fetch_min(g.bad, Index(g.owned ? g.owned[k] : k));
            return;
        }
        if (g.failed_count) {
            const Index node = g.owned ? g.owned[k] : k;
            // На цельной сетке update_node уже получил давление и пометил недопустимый газ.
            const bool ok = !g.owned && g.geometry.packed.gas ? g.geometry.packed.gas[node].density > 0.
                : partition_detail::admissible_node(g.geometry, g.state, node);
            if (!ok) record_failed_node(g, node);
        }
    });
}

// Выбирает число компонент на весь пакет до входа в цикл по узлам.
// Указатели на SA определяют, можно ли в том же проходе обновить турбулентное поле.
template<bool fused,class Space>
void advance_rows(const Space& exec,const Grid& grid,explicit_detail::Stage step,double dt,bool save,
                  const viscosity::Operator* material,const turbulence::View* sa) {
    switch (grid.geometry.dim) {
    case 1: return update_nodes<3,fused>(exec,grid,step,dt,save,material,sa);
    case 2: return update_nodes<4,fused>(exec,grid,step,dt,save,material,sa);
    default: return update_nodes<5,fused>(exec,grid,step,dt,save,material,sa);
    }
}

// Обменивает последовательность типизированных записей. Один индекс передаёт одну полную запись.
template<class Space>
void transfer_records(const Space& exec, const RecordField& field, const MeshIndex* ids, MeshIndex count,
                      const double* input, double* output) {
    const auto copy = KOKKOS_LAMBDA(MeshIndex k) {
        field.copy_record(ids ? ids[k] : k, k, input, output);
    };
    if (std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value) {
        // На хосте сохраняем исходный порядок записей при совпадении индексов.
        for (MeshIndex k = 0; k < count; ++k) copy(k);
    } else {
        for_indices(exec, "Aero: MPI records", count, copy);
    }
}
}
}
