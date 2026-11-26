#pragma once
#include "../parallel/device_faces.h"
#include "split_flux.h"

namespace aero {
namespace residual_detail {
using row_loop::for_indices;
namespace face_kernels = device_work::face_kernels;
// Все исполнители GPU записывают номер первого отказа в один счётчик.
struct SplitFluxFailure {
    Index* bad; // Первый отказ в пакете площадок.
    // Вычисляет свою площадку либо регистрирует отказ.
    KOKKOS_INLINE_FUNCTION void operator()(Index face) const { Kokkos::atomic_fetch_min(bad,face); }
};
template<unsigned W> using CheckedSplitFaces = residual_detail::SplitFluxFaces<W,SplitFluxFailure>;

// Размерность и вид шаблона выбираются до запуска. Укороченные стороны и
// маски восстановления читаются по конкретной грани из той же геометрии.
template<unsigned W,bool linear>
struct StencilSplitFaces {
    const FaceBatch* faces; // Адрес геометрии и полей пакета.
    const MeshIndex* indices; // Номера площадок после разбиения MPI.
    Index* bad; // Первый отказ в пакете площадок.
    // Вычисляет свою площадку либо регистрирует отказ.
    KOKKOS_INLINE_FUNCTION void operator()(Index k) const {
        const FaceBatch& batch=*faces;
        const Index f=FaceBatch::face(indices,k);
        if (batch.is_boundary(f)) {
            if (!linear && !batch.compute_boundary(f)) Kokkos::atomic_fetch_min(bad,f);
            return;
        }
        const auto& g=batch.geometry;
        const auto& st=g.stencil;
        const bool full=!linear && st.weno_order[2*f]==5 && st.weno_order[2*f+1]==5 &&
                        !g.surfaces[f].use_nodal_traces;
        const bool ok=full ? residual_detail::characteristic_flux<W,false,true>(g,f,batch.state,batch.flux[f]) :
                             residual_detail::characteristic_flux<W,linear>(g,f,batch.state,batch.flux[f]);
        if (ok) batch.save_speed(f);
        else Kokkos::atomic_fetch_min(bad,f);
    }
};
// В отдельном ядре граничные условия не увеличивают расход регистров внутренних рёбер.
struct SplitBoundaryFaces {
    const FaceBatch* faces; // Адрес геометрии и полей пакета.
    const MeshIndex* indices; // Номера площадок после разбиения MPI.
    MeshIndex first; // Начало граничной части диапазона.
    Index* bad; // Первый отказ в пакете площадок.
    // Вычисляет свою площадку либо регистрирует отказ.
    KOKKOS_INLINE_FUNCTION void operator()(MeshIndex k) const {
        const Index f=FaceBatch::face(indices,first+k);
        if (faces->is_boundary(f) && !faces->compute_boundary(f)) Kokkos::atomic_fetch_min(bad,f);
    }
};
// Выбирает ядро один раз на пакет. stored указывает на описание в доступной ядру памяти.
// Номер отказа записывается по адресу исполнителя. MPI-диагностику формирует вызывающая сторона.
template<class Space>
void compute_flux_batch(const Space& exec, const FaceBatch& input, Index* bad, const FaceBatch* stored) {
    FaceBatch batch = input;
    const auto& packed = batch.geometry.packed;
    dispatch_face_fluxes(batch, [&] {
        // Ядру нужны только поля узлов. Передаём адреса и два параметра газа,
        // чтобы nvcc не включал в параметры ядра всю геометрию рёбер.
        const unsigned dimension = batch.geometry.dim;
        const double gamma = batch.geometry.options.gamma;
        const double* state = batch.state;
        EulerState* gas = packed.gas;
        for_indices(exec, "Aero: nodal gas", packed.state_nodes, KOKKOS_LAMBDA(Index i) {
            decode_nodal_gas(dimension, state, gamma, i, gas);
        });
    }, [&] {
        face_kernels::compute_linear_fluxes(exec,batch,bad);
    }, [&] {
        if (!stored) {
            for_indices(exec,"Aero: split flux",batch.count,KOKKOS_LAMBDA(Index k) {
                const Index face=batch.face(k);
                CheckedSplitFaces<0>{&batch,nullptr,{bad}}(face);
            });
        } else if (!batch.geometry.stencil.weno_order) {
            // На цельной сетке диапазоны не пересекаются. На разделе MPI
            // фильтруем выданный список по типу площадки внутри каждого ядра.
            const Index edges=batch.geometry.edges;
            const MeshIndex* indices=batch.indices;
            const Index inner=indices ? batch.count : std::min(batch.count,edges);
            const Index first=indices ? 0 : inner;
            for_indices(exec,"Aero: EBR5 boundary flux",batch.count-first,
                        SplitBoundaryFaces{stored,indices,first,bad});
            if (batch.geometry.dim==2) for_indices(exec,"Aero: EBR5 split flux",inner,StencilSplitFaces<4,true>{stored,batch.indices,bad});
            else if (batch.geometry.dim==3) for_indices(exec,"Aero: EBR5 split flux",inner,StencilSplitFaces<5,true>{stored,batch.indices,bad});
            else for_indices(exec,"Aero: EBR5 split flux",inner,StencilSplitFaces<3,true>{stored,batch.indices,bad});
        } else if (batch.geometry.stencil.shared_edge_slopes && batch.geometry.stencil.central_from_edges &&
                   !batch.geometry.options.mp6_hybrid &&
                   (batch.geometry.stencil.mp5.weno || batch.geometry.options.solver!=ResidualSolver::Roe)) {
            if (batch.geometry.dim==2) for_indices(exec,"Aero: full MP split flux",batch.count,StencilSplitFaces<4,false>{stored,batch.indices,bad});
            else if (batch.geometry.dim==3) for_indices(exec,"Aero: full MP split flux",batch.count,StencilSplitFaces<5,false>{stored,batch.indices,bad});
            else for_indices(exec,"Aero: full MP split flux",batch.count,StencilSplitFaces<3,false>{stored,batch.indices,bad});
        } else if (batch.geometry.dim==2)
            for_indices(exec,"Aero: 2D split flux",batch.count,CheckedSplitFaces<4>{stored,batch.indices,{bad}});
        else if (batch.geometry.dim==3)
            for_indices(exec,"Aero: 3D split flux",batch.count,CheckedSplitFaces<5>{stored,batch.indices,{bad}});
        else
            for_indices(exec,"Aero: 1D split flux",batch.count,CheckedSplitFaces<3>{stored,batch.indices,{bad}});
    }, [&] {
        const auto variables = batch.geometry.options.variables;
        if (packed.tiles && variables == ResidualVariables::Characteristic)
            face_kernels::compute_packed_fluxes(exec, batch, bad, stored);
        else if (packed.tiles && stored) {
            if (variables == ResidualVariables::Flux) face_kernels::cartesian_fluxes<true>(exec,batch,bad,stored);
            else face_kernels::cartesian_fluxes<false>(exec,batch,bad,stored);
        } else
            face_kernels::compute_unpacked_fluxes(exec, batch, bad);
    });
}

// За один обход собирает все компоненты узловой невязки из уже вычисленных потоков.
template<class Space>
void gather_flux_batch(const Space& exec, const Geometry& g, const EulerFluxResult* flux, double* residual, Index* bad) {
    for_indices(exec, "Aero: node residuals", g.nodes, KOKKOS_LAMBDA(Index i) {
        if (!assemble_node_residual(g, i, flux, g.node_row(residual, i)))
            Kokkos::atomic_fetch_min(bad, g.faces+i);
    });
}
}
}
