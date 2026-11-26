#pragma once
#include "local_gas_prediction.h"
#include "../parallel/device_faces.h"

namespace aero {
namespace implicit {
// Отбирает грани активных узлов и уточняет начальное приближение на той же памяти поправки.
// stored задаётся для GPU. На CPU список и описание пакета доступны в обычной памяти.
template<class Space>
void predict_local_gas(const Space& exec, MeshIndex rows, const NewtonUpdate& gas, EulerFluxResult* flux,
                       const residual_detail::FaceBatch* stored,const double* initial,double threshold) {
    using namespace residual_detail;
    using row_loop::for_indices;
    static_assert(sizeof(MeshIndex)==sizeof(double),"prediction scratch uses one index per double");
    auto* list=reinterpret_cast<MeshIndex*>(gas.correction);
    auto* scratch = gas.correction;
    const auto surfaces=gas.grid.surfaces;
    const auto edges=gas.grid.edges;
    const auto* rhs=gas.rhs;
    MeshIndex selected=0;
    Kokkos::parallel_scan("Aero: select prediction faces",row_loop::Range<Space>(exec,0,gas.grid.faces),
        KOKKOS_LAMBDA(Index face,MeshIndex& count,bool write) {
            const auto& s=surfaces[face];
            if (implicit::LocalGasPrediction::active(rhs,s.begin,threshold) ||
                (face<edges && implicit::LocalGasPrediction::active(rhs,s.end,threshold))) {
                if (write) {
                    // Сохраняем представление индекса в существующую ячейку поправки.
                    const MeshIndex id = face;
                    ::memcpy(scratch + count, &id, sizeof(id));
                }
                ++count;
            }
        },selected);
    if (!selected || selected>gas.grid.faces/8) return;
    // Счётчик ошибки читает ядро потоков через Index, который в Linux
    // может отличаться от MeshIndex при одинаковой ширине в 64 бита.
    static_assert(sizeof(Index)==sizeof(MeshIndex),"face status must fit one scratch slot");
    Index host_error = ~Index(0);
    auto* bad = std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value
        ? &host_error : reinterpret_cast<Index*>(gas.correction + gas.grid.faces);
    const auto* state=gas.state;
    auto* cached=gas.grid.packed.gas;
    const double gamma=gas.grid.options.gamma;
    for_indices(exec,"Aero: prediction gas",rows,KOKKOS_LAMBDA(Index i) {
        if (!i) *bad=~Index(0);
        decode_nodal_gas(2,state,gamma,i,cached);
    });
    const FaceBatch batch{gas.grid,list,selected,state,initial,flux,nullptr,nullptr,true};
    // Полные рёбра берём из короткого списка. Неполные шаблоны и физические
    // границы проходят свои малые штатные ядра с уже выделенной памятью.
    device_work::face_kernels::compute_packed_fluxes(exec,batch,bad,stored ? stored : &batch);
    for_indices(exec,"Aero: local prediction",rows,implicit::LocalGasPrediction{gas,threshold,bad});
}
}
}
