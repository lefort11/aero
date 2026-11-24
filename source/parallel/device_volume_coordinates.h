#pragma once
#include "device_execution.h"
#include "../viscosity/jacobian.h"

namespace aero {
namespace device_work {
namespace coordinate_aes {
using Team=Kokkos::TeamPolicy<Execution>;

// Каждые 32 потока обрабатывают свой объёмный элемент. Скорости и координаты его
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
// При обычном расчёте невязки храним только градиенты. При сборке матрицы
// из тех же нормалей и объёмов дополнительно интегрируем производные по парам.
// Обе операции пользуются временной памятью одной группы потоков.
struct GradientsOnly {
    using Scratch=Work;
    enum : unsigned { cells_per_team=4 };
    // Возвращает имя запуска Kokkos для профилирования выбранного типа элементов.
    static const char* label(bool all_hex) {
        return all_hex ? "Aero: AES hex coordinates" : "Aero: AES volume coordinates";
    }
    // Сохраняет геометрию симплекса, если выбран режим вычисления парных производных.
    KOKKOS_INLINE_FUNCTION void remember(Work&,unsigned,const viscosity::TetrahedronGeometry&,double) const {}
    // Записывает парные коэффициенты при включённом расчёте якобиана; в режиме градиентов ничего не делает.
    KOKKOS_INLINE_FUNCTION void write_pair(const Work&,MeshIndex,unsigned,
                                           const viscosity::VolumePattern&,const viscosity::Cell&) const {}
};
// Нормали и объёмные множители симплексов в общей памяти команды GPU.
struct PairBasis : Work {
    FixedTensor<double,4,3,24> normal;
    double factor[24];
    // Возвращает сохранённые нормали пары вершин и объёмный множитель симплекса.
    KOKKOS_INLINE_FUNCTION double operator()(unsigned t,unsigned i,unsigned j,Coordinate& left,Coordinate& right) const {
        for (unsigned m=0;m<3;++m) { left[m]=normal(i,m,t); right[m]=normal(j,m,t); }
        return factor[t];
    }
};
// Политика сохранения парных коэффициентов для вязкого якобиана.
struct JacobianPairs {
    using Scratch=PairBasis;
    // Нормали занимают ещё 2496 байт на элемент. Два элемента на группу
    // укладываются примерно в столько же памяти, сколько четыре обычных.
    enum : unsigned { cells_per_team=2 };
    viscosity::CoordinatePairTile tile;
    viscosity::ColumnGraph graph;
    // Возвращает имя запуска Kokkos для профилирования выбранного типа элементов.
    static const char* label(bool all_hex) {
        return all_hex ? "Aero: AES hex gradients and Jacobian pairs" : "Aero: AES volume gradients and Jacobian pairs";
    }

    // Сохраняет геометрию симплекса, если выбран режим вычисления парных производных.
    KOKKOS_INLINE_FUNCTION void remember(PairBasis& w,unsigned tid,
                                         const viscosity::TetrahedronGeometry& tetra,double factor) const {
        w.factor[tid]=factor;
        for (unsigned i=0;i<4;++i)
            for (unsigned m=0;m<3;++m) w.normal(i,m,tid)=tetra.normal[i][m];
    }

    // Записывает парные коэффициенты при включённом расчёте якобиана; в режиме градиентов ничего не делает.
    KOKKOS_INLINE_FUNCTION void write_pair(const PairBasis& w,MeshIndex index,unsigned tid,
                                           const viscosity::VolumePattern& pattern,const viscosity::Cell& cell) const {
        if (tid>=28) return;
        const unsigned first=pattern.pair_offsets[tid];
        if (first!=pattern.pair_offsets[tid+1]) {
            // Любой общий тетраэдр указывает, какие две вершины образуют эту пару.
            // Смотрим номера неизвестных в действительном CSR: диагональ ячейки
            // может оказаться ребром соседнего элемента и попасть в матрицу.
            const unsigned code=pattern.pair_terms[first];
            const auto* vertices=pattern.nodes.row(code/16);
            const auto* nodes=cell.node;
            if (!graph.contains_pair(nodes[vertices[(code/4)%4]],nodes[vertices[code%4]])) return;
        }
        // У пары без общего тетраэдра интеграл равен нулю и перезаписывает
        // значение предыдущей группы, даже когда у этих узлов есть связь в CSR.
        tile.pairs[index*28+tid]=viscosity::integrate_pair(pattern,tid,w);
    }
};
template<bool all_hex,class PairOutput>
// Командное ядро градиентов AES и необязательных парных производных по координатной геометрии.
struct Apply {
    enum : unsigned { threads_per_cell=32, cells_per_team=PairOutput::cells_per_team };
    viscosity::Operator grid;
    viscosity::Gradient* output;
    MeshIndex first_cell, first_vertex, cell_count;
    PairOutput pair_output;

    // Вычисляем площадь противоположной грани для каждой вершины. Один
    // множитель учитывает объём тетраэдра и вес разбиения исходного элемента.
    KOKKOS_INLINE_FUNCTION static void tetra_stress(Work& w,const unsigned char* v,unsigned t,
                                                   const viscosity::TetrahedronGeometry& geometry,double factor) {
        const Coordinate* normal=geometry.normal;
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
    // Вычисляет градиенты и выбранные парные данные группы элементов командой GPU.
    KOKKOS_INLINE_FUNCTION void operator()(const Team::member_type& team) const {
        const unsigned thread=static_cast<unsigned>(team.team_rank());
        const unsigned slot=thread/threads_per_cell, tid=thread%threads_per_cell, i=tid%8;
        const MeshIndex index=static_cast<MeshIndex>(team.league_rank())*cells_per_team+slot;
        const MeshIndex e=first_cell+index;
        // Области соседних элементов не пересекаются. Барьеры относятся ко всей
        // группе, поэтому потоки последней неполной группы тоже доходят до них.
        using Scratch=typename PairOutput::Scratch;
        auto* group=static_cast<Scratch*>(team.team_shmem().get_shmem(cells_per_team*sizeof(Scratch)));
        Scratch& w=group[slot];
        const bool active=index<cell_count;
        const auto geo=active ? grid.tetra.cells[e] : viscosity::VolumeCellView{};
        const auto* pattern=active ? grid.tetra.patterns+geo.pattern : nullptr;
        const auto* cell=active ? grid.cells+e : nullptr;
        const unsigned vertices=active ? (all_hex ? 8u : cell->count) : 0u;
        if (tid<vertices) {
            w.point[i]=grid.tetra.position[geo.vertex[i]];
            const auto& gas=grid.gas[cell->node[i]];
            for (unsigned d=0;d<3;++d) w.u(i, d)=gas.velocity[d];
            w.temperature[i]=gas.pressure_over_density;
        }
        team.team_barrier();
        const MeshIndex base=active ? cell->first_vertex-first_vertex : 0;
        const double* factors=active && e<grid.tetra.cached_cells ? grid.tetra.volume_factors+24*e : nullptr;
        if (active && tid<(all_hex ? 24u : pattern->tetrahedra)) {
            const auto* vertices_of_tetrahedron=pattern->nodes.row(tid);
            const viscosity::TetrahedronGeometry tetra(w.point,vertices_of_tetrahedron);
            const double factor=factors ? factors[tid] : tetra.factor(all_hex ? 24. : pattern->volume_divisor);
            tetra_stress(w,vertices_of_tetrahedron,tid,tetra,factor);
            pair_output.remember(w,tid,tetra,factor);
        } else if (tid>=24 && i<vertices) {
            double sum=0.;
            for (unsigned j=0;j<vertices;++j)
                if (i!=j) sum+=grid.stencil.heat(*cell,i,j)*(w.temperature[j]-w.temperature[i]);
            output[base+i].heat=sum;
        }
        team.team_barrier();
        if (tid<24 && i<vertices) {
            const unsigned m=tid/8;
            double sum=0.;
            for (unsigned j=0;j<(all_hex ? 12u : pattern->degree[i]);++j) {
                const unsigned code=pattern->incidence(i,j);
                sum+=w.stress(code%4,m,code/4);
            }
            output[base+i].stress[m]=sum;
        }
        if (active) pair_output.write_pair(w,index,tid,*pattern,*cell);
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
// Запускает командное вычисление градиентов и выбранного вида парных данных.
template<class PairOutput>
inline void apply(const Execution& exec,const viscosity::Operator& grid,viscosity::Gradient* out,
                      MeshIndex first,MeshIndex count,MeshIndex first_vertex,PairOutput pairs) {
    if (!count) return;
    using Layout=Apply<false,PairOutput>;
    using Scratch=typename PairOutput::Scratch;
    const auto groups=(count-1)/Layout::cells_per_team+1;
    const unsigned threads=Layout::cells_per_team*Layout::threads_per_cell;
    const auto team=Team(exec,static_cast<int>(groups),threads)
        .set_scratch_size(0,Kokkos::PerTeam(Layout::cells_per_team*sizeof(Scratch)));
    if (grid.tetra.all_hex)
        Kokkos::parallel_for(pairs.label(true),team,Apply<true,PairOutput>{grid,out,first,first_vertex,count,pairs});
    else
        Kokkos::parallel_for(pairs.label(false),team,Apply<false,PairOutput>{grid,out,first,first_vertex,count,pairs});
}
// Запускает координатный оператор только для вычисления градиентов.
inline void gradients(const Execution& exec,const viscosity::Operator& grid,viscosity::Gradient* out,
                      MeshIndex first,MeshIndex count,MeshIndex first_vertex) {
    apply(exec,grid,out,first,count,first_vertex,GradientsOnly{});
}
// Строки вязкой невязки в этот момент свободны. В них помещаем тензоры пар
// ровно одной плитки, а градиенты пишем в предназначенный для них отдельный буфер.
inline viscosity::CoordinatePairTile jacobian_pair_storage(const viscosity::Operator& grid,
                                                           const viscosity::GradientTile& part) {
    viscosity::CoordinatePairTile tile{nullptr,part.first_cell};
    if (grid.tetra.cells && grid.rows && part.cells<=tile.capacity(grid.row_count*sizeof(viscosity::Row)))
        tile.pairs=reinterpret_cast<viscosity::PairStress*>(grid.rows);
    return tile;
}
}
}
}
