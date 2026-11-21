#pragma once
#include "device_execution.h"
#include "mpweno5_edges.h"

namespace aero {
namespace device_work {
namespace face_kernels {
using namespace residual_detail;
// Полный шаблон обслуживает обе стороны ребра. Геометрия и формула WENO общие с CPU.
template <unsigned order, bool use_mp5, bool use_mpweno, bool hybrid>
KOKKOS_INLINE_FUNCTION void packed_flux(const FaceBatch& batch, Index face, Index* bad) {
    const auto* tile = complete_tile<order>(batch.geometry, face);
    if (tile && !compute_packed_face<use_mp5, use_mpweno, hybrid>(batch, face, *tile))
        Kokkos::atomic_fetch_min(bad, face);
}

// Для подобласти передаём в ядро один указатель на постоянные адреса. При отдельном
// библиотечном вызове указатели могут измениться, тогда передаём значения текущего пакета.
template <unsigned order, bool use_mp5 = false, bool use_mpweno = false, bool hybrid = false>
void full_edges(const Execution& exec, FaceBatch batch, Index* bad, const FaceBatch* device_faces) {
    if (device_faces) {
        const MeshIndex* indices = batch.indices;
        for_indices<use_mpweno ? 4 : 0>(exec, "Aero: WENO edges", batch.count, KOKKOS_LAMBDA(Index k) {
            packed_flux<order, use_mp5, use_mpweno, hybrid>(*device_faces, indices ? indices[k] : k, bad);
        });
    } else {
        for_indices<use_mpweno ? 4 : 0>(exec, "Aero: WENO edges", batch.count, KOKKOS_LAMBDA(Index k) {
            packed_flux<order, use_mp5, use_mpweno, hybrid>(batch, batch.face(k), bad);
        });
    }
}

// Полные рёбра MPWENO5 и их укороченные стороны проходят три ядра в одном потоке CUDA.
// Первый проход строит наклоны, второй восстанавливает следы, третий записывает один поток на ребро.
void process_mpweno_edges(const Execution& exec, const FaceBatch& batch, Index* bad, const FaceBatch* stored) {
    auto* work = batch.geometry.packed.mp_work;
    const auto* indices = batch.indices;
    // Номер k относится к текущей порции. Номер face учитывает всю подобласть MPI
    // и может отличаться от k даже в первой порции.
    for (Index first = 0; first < batch.count; ) {
        const Index count = std::min(batch.count - first, Index(MpEdgeTile::batch_limit));
        for_indices(exec, "Aero: MPWENO5 projected stencil", count, KOKKOS_LAMBDA(Index k) {
            const Index face = indices ? indices[first + k] : first + k;
            mpweno_edges::project(*stored, face, work[edge_tile::index(k)], edge_tile::lane(k), bad);
        });
        mpweno_edges::reconstruct(exec, work, count, batch.geometry.stencil.mp5);
        for_indices(exec, "Aero: MPWENO5 complete edge flux", count, KOKKOS_LAMBDA(Index k) {
            const Index face = indices ? indices[first + k] : first + k;
            mpweno_edges::write_flux(*stored, face, work[edge_tile::index(k)], edge_tile::lane(k), bad);
        });
        first += count;
    }
}

// Гибрид требует сохранить интервалы всех волн до второго прохода. Выбираем ядро один раз
// на пакет, чтобы при нулевом весе эти интервалы не занимали регистры и локальную память GPU.
template <bool weno>
void mp_edges(const Execution& exec, FaceBatch batch, Index* bad, const FaceBatch* device_faces) {
    if (batch.geometry.options.mp6_hybrid > 0)
        full_edges<5, !weno, weno, true>(exec, batch, bad, device_faces);
    else
        full_edges<5, !weno, weno, false>(exec, batch, bad, device_faces);
}

// При подготовке упаковки рёбра с неполной реконструкцией, узловые рёбра и границы
// уже собраны по типам. Все потоки одного запуска выполняют одну физическую операцию.
enum class RemainingFace { Sparse, Nodal, Boundary };
template <RemainingFace kind>
void remaining_group(const Execution& exec, FaceBatch batch, const FaceBatch* stored, Index begin,
                     Index count, Index* bad) {
    const auto apply = KOKKOS_LAMBDA(const FaceBatch& data, Index k) {
        const Index face = data.geometry.packed.remaining[begin + k];
        const bool ok = kind == RemainingFace::Boundary ? data.compute_boundary(face)
                          : data.compute_edge<true, kind == RemainingFace::Nodal, 0, kind == RemainingFace::Sparse>(face);
        if (!ok)
            Kokkos::atomic_fetch_min(bad, face);
    };
    if (stored)
        for_indices(exec, "Aero: remaining WENO faces", count, KOKKOS_LAMBDA(Index k) { apply(*stored, k); });
    else
        for_indices(exec, "Aero: remaining WENO faces", count, KOKKOS_LAMBDA(Index k) { apply(batch, k); });
}

// Выбираем обход один раз на весь пакет. Внутри площадки остаются только формулы выбранного решателя.
template <bool roe_2d, bool nodal_only = false, unsigned linear_width = 0, unsigned lf_dim = 0, bool ebr_flux = false>
void other_faces(const Execution& exec, FaceBatch batch, Index* bad) {
    // У первого порядка, EBR и двумерного Роу есть специализированные ядра.
    // Для произвольных нелинейных шаблонов идём по общему списку площадок.
    // Один вызов compute_face выбирает поток по ребру или граничное условие.
    if ((nodal_only || linear_width || roe_2d) && !batch.indices && batch.count == batch.geometry.faces) {
        // В однопроцессном расчёте рёбра и внешние площадки уже лежат двумя непрерывными группами.
        // У каждого обхода свои регистры. Граничная характеристика не занимает регистры потока на ребре.
        const Index edges = batch.geometry.edges;
        for_indices(exec, "Aero: interior edges", edges, KOKKOS_LAMBDA(Index face) {
            if (!batch.compute_edge<roe_2d, nodal_only, linear_width, false, ebr_flux>(face,lf_dim))
                Kokkos::atomic_fetch_min(bad, face);
        });
        for_indices(exec, "Aero: boundary faces", batch.count - edges, KOKKOS_LAMBDA(Index i) {
            const Index face = edges + i;
            if (!batch.compute_boundary(face))
                Kokkos::atomic_fetch_min(bad, face);
        });
        return;
    }
    // После разделения сетки MPI и у неполных шаблонов нужные номера берём из списка.
    for_indices(exec, "Aero: remaining fluxes", batch.count, KOKKOS_LAMBDA(Index k) {
        const Index face = batch.face(k);
        if (!batch.compute_face<roe_2d, nodal_only, linear_width, ebr_flux>(face,lf_dim))
            Kokkos::atomic_fetch_min(bad, face);
    });
}

// Условия отбора рёбер и геометрия пакетов общие с характеристической реконструкцией.
// На границах и при нехватке точек остаётся обычная формула именно выбранной схемы.
template<bool physical, unsigned order, bool mp = false, bool mpweno = false, bool hybrid = false>
void cartesian_edges(const Execution& exec, FaceBatch batch, Index* bad, const FaceBatch* stored) {
    const Index* ids = batch.indices;
    // У MPWENO5 больше временных весов. Ограничиваем число регистров на
    // поток, чтобы несколько блоков одновременно загружали соседние шаблоны.
    for_indices<mpweno ? 4 : 0>(exec,"Aero: Cartesian reconstructed edges",batch.count,KOKKOS_LAMBDA(Index k) {
        const Index edge = ids ? ids[k] : k;
        const auto* tile = complete_tile<order>(stored->geometry, edge);
        if (tile && !cartesian_packed_flux<physical,order,mp,mpweno,hybrid>(*stored,edge,*tile))
            Kokkos::atomic_fetch_min(bad,edge);
    });
    batch.indices = batch.geometry.packed.remaining;
    batch.count = batch.geometry.packed.count;
    if (batch.count) other_faces<true>(exec,batch,bad);
}

template<bool physical>
void cartesian_fluxes(const Execution& exec, const FaceBatch& batch, Index* bad, const FaceBatch* stored) {
    const auto& mp = batch.geometry.stencil.mp5;
    if (mp.weno && batch.geometry.options.mp6_hybrid > 0.)
        return cartesian_edges<physical,5,false,true,true>(exec,batch,bad,stored);
    if (mp.enabled && batch.geometry.options.mp6_hybrid > 0.)
        return cartesian_edges<physical,5,true,false,true>(exec,batch,bad,stored);
    if (batch.geometry.packed.order == 3) cartesian_edges<physical,3>(exec,batch,bad,stored);
    else if (batch.geometry.stencil.mp5.weno) cartesian_edges<physical,5,false,true>(exec,batch,bad,stored);
    else if (batch.geometry.stencil.mp5.enabled) cartesian_edges<physical,5,true>(exec,batch,bad,stored);
    else cartesian_edges<physical,5>(exec,batch,bad,stored);
}

// Полные шаблоны WENO и MP уложены по 32 ребра. Выбираем ядро один раз на пакет.
void compute_packed_fluxes(const Execution& exec, FaceBatch batch, Index* bad, const FaceBatch* device_faces) {
    const auto& packed = batch.geometry.packed;
    // После включения гибрида используем ядро с ограничением MP6 даже при наличии рабочего буфера MPWENO5.
    const bool split_mpweno = device_faces && packed.mp_work && batch.geometry.options.mp6_hybrid == 0.;
    if (split_mpweno)
        process_mpweno_edges(exec, batch, bad, device_faces);
    else if (packed.order == 5 && batch.geometry.stencil.mp5.weno)
        mp_edges<true>(exec, batch, bad, device_faces);
    else if (packed.order == 5 && batch.geometry.stencil.mp5.enabled)
        mp_edges<false>(exec, batch, bad, device_faces);
    else if (packed.order == 5)
        full_edges<5>(exec, batch, bad, device_faces);
    else
        full_edges<3>(exec, batch, bad, device_faces);

    // После полных шаблонов обходим три непересекающихся диапазона списка remaining.
    const Index boundary = packed.sparse_edges + packed.nodal_edges;
    // Укороченные стороны уже получили поток в третьем проходе MPWENO5.
    const Index handled_mixed = split_mpweno ? packed.mixed_edges : 0;
    remaining_group<RemainingFace::Sparse>(exec, batch, device_faces, 0, packed.sparse_edges - handled_mixed, bad);
    remaining_group<RemainingFace::Nodal>(exec, batch, device_faces, packed.sparse_edges, packed.nodal_edges, bad);
    remaining_group<RemainingFace::Boundary>(exec, batch, device_faces, boundary, packed.count - boundary, bad);
}

// В консервативных и характеристических переменных EBR суммирует только U.
// В потоковых переменных те же веса применяются также к F(U). Задаём состав
// строки один раз на пакет рёбер, чтобы малое ядро не резервировало память большого.
template<bool roe_2d, unsigned width>
void ebr_faces(const Execution& exec, FaceBatch batch, Index* bad) {
    if (batch.geometry.options.variables == ResidualVariables::Flux)
        other_faces<roe_2d, false, width, 0, true>(exec, batch, bad);
    else
        other_faces<roe_2d, false, width>(exec, batch, bad);
}

// Для первого порядка, EBR и сеток без упаковки используем обычный список площадок.
void compute_unpacked_fluxes(const Execution& exec, FaceBatch batch, Index* bad) {
    // Первый порядок читает два узла без обращения к строкам шаблона.
    if (batch.geometry.options.background.active()) {
        for_indices(exec, "Aero: linear Euler fluxes", batch.count, KOKKOS_LAMBDA(Index k) {
            const auto face = batch.face(k);
            if (!batch.compute_linear(face)) Kokkos::atomic_fetch_min(bad,face);
        });
        return;
    }
    // Как и на CPU, LF первого порядка получает состояния прямо из узлов.
    // Постоянная размерность исключает реконструкцию из кода ядра этого запуска.
    if (batch.nodal_lf()) {
        const auto launch = batch.select_by_dimension(other_faces<false,true,0,1>,
                                                       other_faces<false,true,0,2>,
                                                       other_faces<false,true,0,3>);
        launch(exec,batch,bad);
    } else if (batch.nodal_roe_2d())
        other_faces<true, true>(exec, batch, bad);
    else if (!batch.geometry.stencil.row_owner)
        // В 1D и 3D поток Роу первого порядка также использует только концы ребра.
        other_faces<false, true>(exec, batch, bad);
    // У EBR3 и EBR5 различны только заранее рассчитанные веса. В 2D в узле четыре компоненты.
    else if (batch.roe_2d() && !batch.geometry.stencil.weno_order)
        ebr_faces<true, 4>(exec, batch, bad);
    else if (batch.roe_2d())
        other_faces<true>(exec, batch, bad);
    // В 3D фиксированная ширина линейной строки равна пяти.
    else if (batch.geometry.dim == 3 && batch.geometry.stencil.row_owner &&
             !batch.geometry.stencil.weno_order)
        ebr_faces<false, 5>(exec, batch, bad);
    else
        other_faces<false>(exec, batch, bad);
}

} // namespace face_kernels
} // namespace device_work
} // namespace aero
