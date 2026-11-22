#pragma once
#include "device_execution.h"
#include "../viscosity/jacobian.h"

namespace aero {
namespace device_work {
namespace coordinate_aes {
using Team=Kokkos::TeamPolicy<Execution>;

// Одна группа из 32 потоков обрабатывает объёмный элемент. Скорости и координаты его
// вершин читаем один раз. До 24 потоков считают напряжения в тетраэдрах, до восьми
// остальных используют исходные коэффициенты теплопроводности AES.
struct Work {
    Coordinate point[8];
    FixedMatrix<double,8,3> u;
    double temperature[8];
    // Соседние потоки одновременно записывают одну и ту же компоненту
    // в разных тетраэдрах. Их адреса в быстрой памяти группы идут подряд.
    FixedTensor<double,4,3,24> stress;
};
template<bool all_hex>
struct Apply {
    viscosity::Operator grid;
    viscosity::Gradient* output;
    MeshIndex first_cell, first_vertex;

    // Вычисляем площадь противоположной грани для каждой вершины. Один
    // множитель учитывает объём тетраэдра и вес разбиения исходного элемента.
    KOKKOS_INLINE_FUNCTION static void tetra_stress(Work& w,const unsigned char* v,unsigned t,
                                                   const double* factors, double divisor) {
        const viscosity::TetrahedronGeometry geometry(w.point,v);
        const Coordinate* normal=geometry.normal;
        const double factor=factors ? factors[t] : geometry.factor(divisor);
        FixedMatrix<double,3,3> gradient{};
        for (unsigned m=0;m<3;++m)
            for (unsigned j=1;j<4;++j) {
                const double delta=w.u(v[j],m)-w.u(v[0],m);
                for (unsigned d=0;d<3;++d) gradient(m, d)+=delta*normal[j][d];
            }
        const double volume_part=(2./3.)*(gradient(0, 0)+gradient(1, 1)+gradient(2, 2));
        // Симметричный тензор имеет шесть компонент. Геометрический множитель
        // включаем один раз, чтобы повторно использовать напряжение у трёх вершин.
        const double xy=factor*(gradient(0, 1)+gradient(1, 0));
        const double xz=factor*(gradient(0, 2)+gradient(2, 0));
        const double yz=factor*(gradient(1, 2)+gradient(2, 1));
        const Coordinate stress[]={
            {{factor*(2.*gradient(0, 0)-volume_part),xy,xz}},
            {{xy,factor*(2.*gradient(1, 1)-volume_part),yz}},
            {{xz,yz,factor*(2.*gradient(2, 2)-volume_part)}}};
        for (unsigned m=0;m<3;++m) {
            double sum=0.;
            for (unsigned j=1;j<4;++j) {
                const double flux=dot3(stress[m],normal[j]);
                w.stress(j,m,t)=flux;
                sum+=flux;
            }
            // Сумма четырёх площадок тетраэдра равна нулю. Четвёртый вклад
            // получаем из сохранения импульса, отдельно проектировать его не нужно.
            w.stress(0,m,t)=-sum;
        }
    }
    KOKKOS_INLINE_FUNCTION void operator()(const Team::member_type& team) const {
        const auto e=first_cell+static_cast<unsigned>(team.league_rank());
        const unsigned tid=team.team_rank(), i=tid%8;
        auto& w=*static_cast<Work*>(team.team_shmem().get_shmem(sizeof(Work)));
        const auto& geo=grid.tetra.cells[e];
        const auto& pattern=grid.tetra.patterns[geo.pattern];
        const auto& cell=grid.cells[e];
        const unsigned vertices=all_hex ? 8 : cell.count;
        if (tid<vertices) {
            w.point[i]=grid.tetra.position[geo.vertex[i]];
            const auto& gas=grid.gas[cell.node[i]];
            for (unsigned d=0;d<3;++d) w.u(i, d)=gas.velocity[d];
            w.temperature[i]=gas.pressure_over_density;
        }
        team.team_barrier();
        const MeshIndex base=cell.first_vertex-first_vertex;
        const double* factors=e<grid.tetra.cached_cells ? grid.tetra.volume_factors+24*e : nullptr;
        if (tid<(all_hex ? 24u : pattern.tetrahedra))
            tetra_stress(w,pattern.nodes.row(tid),tid,factors,all_hex ? 24. : pattern.volume_divisor);
        else if (tid>=24 && i<vertices) {
            double sum=0.;
            for (unsigned j=0;j<vertices;++j)
                if (i!=j) sum+=grid.stencil.heat(cell,i,j)*(w.temperature[j]-w.temperature[i]);
            output[base+i].heat=sum;
        }
        team.team_barrier();
        if (tid<24 && i<vertices) {
            const unsigned m=tid/8;
            double sum=0.;
            for (unsigned j=0;j<(all_hex ? 12u : pattern.degree[i]);++j) {
                const unsigned code=pattern.incidence(i,j);
                sum+=w.stress(code%4,m,code/4);
            }
            output[base+i].stress[m]=sum;
        }
    }
};
// Объём тетраэдра не зависит от решения или шага времени. Заполняем таблицу
// один раз на карте. Формула та же, что и при вычислении внутри стадии SSPRK3.
inline void prepare_volumes(const Execution& exec,const viscosity::VolumeCoordinates& geometry) {
    const auto g=geometry;
    for_indices(exec,"Aero: prepare AES tetrahedron volumes",24*g.cached_cells,KOKKOS_LAMBDA(MeshIndex k) {
        const auto& cell=g.cells[k/24];
        const auto& pattern=g.patterns[cell.pattern];
        if (k%24>=pattern.tetrahedra) return;
        const auto* v=pattern.nodes.row(static_cast<unsigned>(k%24));
        const viscosity::TetrahedronGeometry tetra(viscosity::VolumePoints{g.position,&cell},v);
        g.volume_factors[k]=tetra.factor(pattern.volume_divisor);
    });
}
// В пределах группы сначала вычисляем нормали каждого тетраэдра один раз.
// Затем до 28 потоков одновременно интегрируют производные для разных пар ячейки.
struct PairWork {
    Coordinate point[8];
    FixedTensor<double,4,3,24> normal;
    double factor[24];
    KOKKOS_INLINE_FUNCTION double operator()(unsigned t,unsigned i,unsigned j,Coordinate& left,Coordinate& right) const {
        for (unsigned m=0;m<3;++m) { left[m]=normal(i,m,t); right[m]=normal(j,m,t); }
        return factor[t];
    }
};
struct PreparePairs {
    viscosity::Operator grid;
    viscosity::CoordinatePairTile tile;
    viscosity::ColumnGraph graph;
    KOKKOS_INLINE_FUNCTION void operator()(const Team::member_type& team) const {
        const unsigned index=static_cast<unsigned>(team.league_rank()), tid=team.team_rank();
        const MeshIndex e=tile.first_cell+index;
        const auto& cell=grid.tetra.cells[e];
        const auto& pattern=grid.tetra.patterns[cell.pattern];
        auto& w=*static_cast<PairWork*>(team.team_shmem().get_shmem(sizeof(PairWork)));
        if (tid<grid.cells[e].count) w.point[tid]=grid.tetra.position[cell.vertex[tid]];
        team.team_barrier();
        if (tid<pattern.tetrahedra) {
            const viscosity::TetrahedronGeometry tetra(w.point,pattern.nodes.row(tid));
            const auto& g=grid.tetra;
            w.factor[tid]=e<g.cached_cells ? g.volume_factors[24*e+tid] : tetra.factor(pattern.volume_divisor);
            for (unsigned i=0;i<4;++i) for (unsigned m=0;m<3;++m) w.normal(i,m,tid)=tetra.normal[i][m];
        }
        team.team_barrier();
        if (tid>=28) return;
        const unsigned first=pattern.pair_offsets[tid];
        if (first!=pattern.pair_offsets[tid+1]) {
            // Любой общий тетраэдр указывает, какие две вершины образуют эту пару.
            // Смотрим номера неизвестных в действительном CSR: диагональ ячейки
            // может оказаться ребром соседнего элемента и попасть в матрицу.
            const unsigned code=pattern.pair_terms[first];
            const auto* vertices=pattern.nodes.row(code/16);
            const auto* nodes=grid.cells[e].node;
            if (!graph.contains_pair(nodes[vertices[(code/4)%4]],nodes[vertices[code%4]])) return;
        }
        // У пары без общего тетраэдра интеграл равен нулю и перезаписывает
        // значение предыдущей группы, даже когда у этих узлов есть связь в CSR.
        tile.pairs[index*28+tid]=viscosity::integrate_pair(pattern,tid,w);
    }
};
inline viscosity::CoordinatePairTile prepare_pairs(const Execution& exec,const viscosity::Jacobian& matrix,
                                                   MeshIndex first,MeshIndex count) {
    const auto& grid=matrix.aes;
    viscosity::CoordinatePairTile tile{nullptr,first};
    // Во время сборки производных рабочие строки пространственной невязки уже свободны.
    // Если вся группа помещается, все пять столбцов читают одни и те же готовые интегралы.
    if (!grid.tetra.cells || !grid.rows || count>tile.capacity(grid.row_count*sizeof(viscosity::Row))) return tile;
    tile.pairs=reinterpret_cast<viscosity::PairStress*>(grid.rows);
    const auto team=Team(exec,static_cast<int>(count),32).set_scratch_size(0,Kokkos::PerTeam(sizeof(PairWork)));
    Kokkos::parallel_for("Aero: integrate AES Jacobian pairs",team,PreparePairs{grid,tile,matrix.column_graph()});
    return tile;
}
inline void gradients(const Execution& exec,const viscosity::Operator& grid,viscosity::Gradient* out,
                      MeshIndex first,MeshIndex count,MeshIndex first_vertex) {
    const auto team=Team(exec,static_cast<int>(count),32).set_scratch_size(0,Kokkos::PerTeam(sizeof(Work)));
    if (grid.tetra.all_hex)
        Kokkos::parallel_for("Aero: AES hex coordinates",team,Apply<true>{grid,out,first,first_vertex});
    else
        Kokkos::parallel_for("Aero: AES volume coordinates",team,Apply<false>{grid,out,first,first_vertex});
}
}
}
}
