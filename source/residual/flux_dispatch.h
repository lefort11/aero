#pragma once
#include "split_flux.h"

namespace aero {
namespace residual_detail {
// Решение о подготовке газа и физическом режиме принимается один раз на пакет.
// Обходы CPU и специализированные запуски GPU остаются у вызывающего исполнителя.
template<class Gas, class Linear, class Split, class Reconstructed>
inline void dispatch_face_fluxes(const FaceBatch& batch, const Gas& gas, const Linear& linear,
                                 const Split& split, const Reconstructed& reconstructed) {
    if (batch.needs_gas_update()) gas();
    if (batch.geometry.options.background.active()) linear();
    else if (uses_split_flux(batch.geometry)) split();
    else reconstructed();
}

// Порядок известен компилятору. CPU вызывает формулу одного ребра, GPU — выбранное ядро.
template<unsigned order, class Work>
inline auto dispatch_packed_reconstruction(const Geometry& g, const Work& work)
    -> decltype(work.apply_weno()) {
    const auto& mp=g.stencil.mp5;
    if (order==5 && mp.weno) return work.template apply_mp<true>();
    if (order==5 && mp.enabled) return work.template apply_mp<false>();
    return work.apply_weno();
}
}
}
