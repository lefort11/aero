#pragma once
#include "../parallel/row_loop.h"
#include "../residual/packed_weno.h"
#include "transport.h"
#include "convection.h"

namespace aero {
namespace turbulence {
namespace execution {
using row_loop::for_indices;
using residual_detail::Index;
using residual_detail::WenoEdgeTile;
// Размерность известна всей задаче, поэтому выбираем ядро до обхода узлов.
// Внутри остаётся только формула для нужного числа компонент скорости.
template<unsigned dim,bool same_gas,class Space>
void sa_rows(const Space& exec,const turbulence::Transport& transport,bool save_rotation) {
    for_indices(exec,"Aero: SA residual",transport.count,KOKKOS_LAMBDA(Index i) {
        transport.residual_row<dim,same_gas,std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value>(i,save_rotation);
    });
}
// Собирает локальные невязки уравнения SA по строкам раздела.
template<bool same_gas,class Space>
void sa_residual(const Space& exec,const turbulence::Step& step) {
    // На первом обходе неявного газа сохраняем ротор. Во внутренних итерациях
    // меняется только nu_tilde, поэтому берём уже сохранённый ротор.
    const bool save_rotation=!same_gas && step.keep_gas;
    const auto launch = for_dimension(step.transport.dim, sa_rows<1,same_gas,Space>,
                                      sa_rows<2,same_gas,Space>, sa_rows<3,same_gas,Space>);
    launch(exec, step.transport, save_rotation);
}
// Обходит обычные либо упакованные рёбра в соответствии с геометрией раздела.
template<class Space>
void convection(const Space& exec, const SaEdgeConvection& edges, MeshIndex count, const void* tiles, unsigned order) {
    if (!tiles || edges.hybrid>0. || edges.stencil.mp5.weno) {
        for_indices(exec,"Aero: reconstructed SA convection",count,edges);
    } else if (order==3) {
        const auto* geometry=static_cast<const WenoEdgeTile<3>*>(tiles);
        for_indices(exec,"Aero: packed WENO3 SA convection",count,KOKKOS_LAMBDA(Index k){edges.packed<3,false>(k,geometry);});
    } else {
        const auto* geometry=static_cast<const WenoEdgeTile<5>*>(tiles);
        if (edges.stencil.mp5.enabled)
            for_indices(exec,"Aero: packed MP5 SA convection",count,KOKKOS_LAMBDA(Index k){edges.packed<5,true>(k,geometry);});
        else
            for_indices(exec,"Aero: packed WENO5 SA convection",count,KOKKOS_LAMBDA(Index k){edges.packed<5,false>(k,geometry);});
    }
}
// Выполняет операцию шага SA. При наличии кэша вкладов вершин заполняет его до обхода строк.
template<class Space>
void apply_step(const Space& exec, const Step& step, Operation op) {
    step.dispatch(op,[&](Index count) {
        for_indices(exec,"Aero: SA line relaxation",count,KOKKOS_LAMBDA(Index k){step.relax_line(k);});
    }, [&] {
        if (std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value && step.transport.sa.diffusion.vertex_contributions)
            for_indices(exec,"Aero: SA element diffusion",step.transport.cell_count,
                        KOKKOS_LAMBDA(Index e){step.transport.diffuse_element(e);});
        sa_residual<false>(exec,step);
    }, [&](Index count) {
        for_indices(exec,"Aero: SA transport",count,KOKKOS_LAMBDA(Index i){step.apply(op,i);});
    });
}
// Выполняет редукцию максимальной величины SA в выбранном пространстве исполнения.
template<class Space>
double sa_norm(const Space& exec,const turbulence::Step& step,turbulence::Norm mode) {
    double norm=0.;
    if (step.transport.count) {
        using Range=row_loop::Range<Space>;
        Kokkos::parallel_reduce("Aero: SA maximum",Range(exec,0,step.transport.count),
            KOKKOS_LAMBDA(Index i,double& best) {
                step.collect_maximum(mode,i,best);
            },Kokkos::Max<double>(norm));
    }
    return norm;
}
// При обновлении газа последовательно пересчитывает диффузию и строки, затем читает норму.
// Все три операции используют существующие массивы текущего шага.
template<class Space>
double measure_norm(const Space& exec, const Step& step, Norm mode, bool refresh) {
    return step.evaluate_norm(mode,refresh,[&] {
        for_indices(exec,"Aero: SA element diffusivity",step.transport.cell_count,
                    KOKKOS_LAMBDA(Index e){step.transport.refresh_diffusivity<std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value>(e);});
    }, [&] {
        // Сборка строки читает площадки и пары AES. Её запускаем по всем
        // узлам обычным parallel_for, у которого достаточно блоков и на
        // большой сетке. Максимум использует готовые rate и diagonal.
        sa_residual<true>(exec,step);
        return sa_norm(exec,step,turbulence::Norm::Bdf);
    }, [&](turbulence::Norm which) { return sa_norm(exec,step,which); });
}
}
}
}
