#pragma once
#include "../parallel/row_loop.h"
#include "csgs_sweep.h"
#include "spatial_jacobian_detail.h"

namespace aero {
namespace implicit {
namespace matrix_rows {
using row_loop::for_indices;
using Index = MeshIndex;
// Разлагает диагональные блоки и возвращает наименьший номер вырожденного блока.
template<class Space>
void factor(const Space& exec, const JacobiBlocks& blocks, const double* diagonal, const MeshIndex* row_starts, MeshIndex* first_bad) {
    // Одна строка редукции разлагает один узловой блок. На CPU нужен лишь
    // минимальный номер сбоя, чтобы не начинать итерации с вырожденной диагональю.
    Kokkos::parallel_reduce("Aero: factor block Jacobi", row_loop::Range<Space>(exec, 0, blocks.nodes),
        KOKKOS_LAMBDA(Index node, MeshIndex& bad) {
            if (!blocks.factor_node(node, diagonal, row_starts) && node < bad)
                bad = node;
        }, Kokkos::Min<MeshIndex>(*first_bad));
}
// Применяет независимые подстановки к строкам предобусловливателя.
template<class Space>
void solve(const Space& exec, const JacobiBlocks& blocks, const double* residual, double* correction) {
    for_indices(exec, "Aero: apply block Jacobi", blocks.nodes, KOKKOS_LAMBDA(Index node) {
        blocks.solve_node(node, residual, correction);
    });
}
// Переносит коэффициенты в заранее выбранную раскладку CSGS.
template<class Space>
void pack(const Space& exec, const CsgsPacking& plan, const double* matrix) {
    for_indices(exec,"Aero: prepare CSGS coefficients",plan.count,KOKKOS_LAMBDA(Index p){plan.copy(p,matrix);});
}
// Вычисляет образ вектора в фактической раскладке проходов CSGS.
template<class Space>
void image(const Space& exec, const CsgsSweep& sweep, MeshIndex count, double* image) {
    if (sweep.lanes==32) for_indices(exec,"Aero: CSGS matrix product",count,KOKKOS_LAMBDA(Index k){sweep.image<32>(k,image);});
    else for_indices(exec,"Aero: CSGS matrix product",count,KOKKOS_LAMBDA(Index k){sweep.image<1>(k,image);});
}
// Дифференцирует потоки и сохраняет номер первой недопустимой площадки.
template<class Space>
void differentiate(const Space& exec, const SpatialBlocks& blocks, const double* u, const double* initial, MeshIndex* bad) {
    const auto light = Kokkos::Experimental::WorkItemProperty::HintLightWeight;
    Kokkos::parallel_reduce("Aero: differentiate face fluxes",
        Kokkos::Experimental::require(row_loop::Range<Space>(exec, 0, blocks.face_count), light),
        KOKKOS_LAMBDA(Index k, MeshIndex& first) {
            const auto f = blocks.face(k);
            if (!blocks.differentiate(f, u, initial) && f < first) first = f;
        }, Kokkos::Min<MeshIndex>(*bad));
}
// Собирает все компоненты строки на CPU или соседние компоненты параллельно на GPU.
template<unsigned Width,bool direct=false,class Space>
void assemble_matrix_rows(const Space& exec, implicit::SpatialBlocks blocks,
                          const double* volumes, double dt, MeshIndex* bad) {
    // При вычислении производной на месте один исполнитель сразу записывает
    // весь блок: пять компонент не повторяют вычисление потока.
    constexpr unsigned lanes = direct || std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value ? 1u : Width;
    const auto range=Kokkos::Experimental::require(row_loop::Range<Space>(exec,0,blocks.row_count*lanes),
                           Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_reduce("Aero: assemble Jacobian rows",range,KOKKOS_LAMBDA(Index i, MeshIndex& first) {
        const auto row=i/lanes;
        if (!blocks.assemble_equations<Width,Width/lanes,direct>(row,static_cast<unsigned>(i%lanes),volumes,dt)) {
            const auto node=blocks.node(row);
            if (node<first) first=node;
        }
    }, Kokkos::Min<MeshIndex>(*bad));
}
// Выбирает размер блока и собирает строки пространственного якобиана.
template<class Space>
void assemble(const Space& exec, const SpatialBlocks& matrix, const double* volumes, double dt, MeshIndex* bad) {
    if (matrix.direct_faces) return assemble_matrix_rows<5,true>(exec,matrix,volumes,dt,bad);
    switch (matrix.grid.dim) {
    case 1: return assemble_matrix_rows<3>(exec,matrix,volumes,dt,bad);
    case 2: return assemble_matrix_rows<4>(exec,matrix,volumes,dt,bad);
    default: return assemble_matrix_rows<5>(exec,matrix,volumes,dt,bad);
    }
}
// Обновляет временной вклад диагонали и регистрирует отказ узла.
template<class Space>
void time_diagonal(const Space& exec, const SpatialBlocks& blocks, const double* volumes, double dt, MeshIndex* bad) {
    // Передаём описание матрицы в аргументах ядра выбранной карты.
    // В одном процессе каждая карта использует свой поток исполнения.
    const auto range = Kokkos::Experimental::require(row_loop::Range<Space>(exec, 0, blocks.row_count),
                              Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_reduce("Aero: update BDF time diagonal", range,
        KOKKOS_LAMBDA(Index row, MeshIndex& first) {
            if (!blocks.update_time_diagonal(row, volumes, dt) && blocks.node(row) < first)
                first = blocks.node(row);
        }, Kokkos::Min<MeshIndex>(*bad));
}
// Распределяет компоненты строки между дорожками, сохраняя порядок суммы для каждого уравнения.
template<unsigned Width,class Space>
void multiply_matrix_rows(const Space& exec, implicit::SpatialBlocks blocks, const double* x, double* y) {
    constexpr unsigned lanes = std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value ? 1u : Width;
    for_indices(exec,"Aero: spatial Jacobian times vector",blocks.row_count*lanes,KOKKOS_LAMBDA(Index i) {
        blocks.multiply_equations<Width,Width/lanes>(i/lanes,static_cast<unsigned>(i%lanes),x,y);
    });
}
// Умножает распределённую матрицу на расширенный вектор.
template<class Space>
void multiply(const Space& exec, const SpatialBlocks& matrix, const double* x, double* y) {
    switch (matrix.grid.dim) {
    case 1: return multiply_matrix_rows<3>(exec,matrix,x,y);
    case 2: return multiply_matrix_rows<4>(exec,matrix,x,y);
    default: return multiply_matrix_rows<5>(exec,matrix,x,y);
    }
}
}
}
}
