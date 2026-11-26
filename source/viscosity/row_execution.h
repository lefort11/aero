#pragma once
#include "../parallel/device_affine_aes.h"
#include "transport.h"

namespace aero {
namespace viscosity {
namespace execution {
using row_loop::for_indices;
using Index = MeshIndex;
namespace affine_aes = device_work::affine_aes;
namespace coordinate_aes = device_work::coordinate_aes;
// Собирает вязкие производные в блоки пространственной матрицы в памяти выбранного исполнителя.
template<class Space>
struct MatrixAssembly {
    // Хост читает по ссылке, GPU получает описание матрицы по значению.
    template<class T> using Capture = typename std::conditional<std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value,const T&,T>::type;
    const Space& exec; // Пространство и поток исполнения сборки.
    // Запускает независимые записи матрицы.
    template<class Work>
    void run(const char* name, Index count, const Work& work) const { for_indices(exec,name,count,work); }

    // Каждая пара «узел, переменная» отвечает за собственный столбец.
    void columns(const viscosity::Jacobian& v) const {
        const unsigned width = v.aes.dim + 2;
        for_indices(exec,"Aero: viscous Jacobian columns",v.aes.node_count*width,KOKKOS_LAMBDA(Index k) {
            v.assemble_column(k/width,static_cast<unsigned>(k%width));
        });
    }
    // Использует компактные плитки, когда такое представление есть у сетки.
    bool assemble_compact(const viscosity::Jacobian& v) const {
        const auto& aes = v.aes;
        if (!aes.affine && !aes.tetra.cells) return false;
        const unsigned width = aes.dim + 2;
        for_indices(exec,"Aero: Jacobian cell material",aes.cell_count,KOKKOS_LAMBDA(Index e) { aes.average_material(e); });
        if (v.column_nodes && aes.gradient_tile_count) {
            // Разные потоки GPU заполняют разные компоненты столбца. Внутри
            // столбца обходим группы по номеру элемента, как при полной сборке.
            affine_aes::for_tiles(aes,[&](viscosity::GradientTile tile) {
                const auto pairs=coordinate_aes::jacobian_pair_storage(aes,tile);
                if (pairs.pairs) {
                    // В этой же группе площадки тетраэдров идут на напряжения и на производные.
                    coordinate_aes::apply(exec,aes,aes.gradient_cache,tile.first_cell,tile.cells,tile.first_vertex,
                                          coordinate_aes::JacobianPairs{pairs,v.column_graph()});
                } else {
                    // При иной геометрии или недостатке свободных строк add_column
                    // по-прежнему получает коэффициенты из представления самой сетки.
                    affine_aes::gradients(exec,aes,aes.gradient_cache,tile.first_cell,tile.cells,tile.first_vertex);
                }
                for_indices(exec,"Aero: viscous Jacobian column slices",tile.slices*width,KOKKOS_LAMBDA(Index k) {
                    const auto slice=aes.gradient_slices[tile.first_slice+k/width];
                    v.add_column<3>(slice,static_cast<unsigned>(k%width),tile.first_vertex,pairs);
                });
            });
            return true;
        }
        affine_aes::gradients(exec,aes,v.gradients,0,aes.cell_count,0);
        columns(v);
        return true;
    }
};

#ifdef AERO_HAS_TURBULENCE
// На сетке из однотипных элементов число вершин и наличие уравнения SA
// одинаковы у всей группы. Выбираем их до запуска, чтобы в цикле по вершинам
// сразу собирать нужные коэффициенты переноса.
template<unsigned vertices,bool with_sa,class Space>
void cell_material(const Space& exec, const viscosity::Operator& g) {
    for_indices(exec,"Aero: turbulent material",g.cell_count,KOKKOS_LAMBDA(Index e){
        g.average_cached_material<vertices,with_sa>(e);
    });
}
// На сетке из однотипных элементов число вершин и наличие уравнения SA
// одинаковы у всей группы. Выбираем их до запуска, чтобы в цикле по вершинам
// сразу собирать нужные коэффициенты переноса.
template<bool with_sa,class Space>
void cell_material(const Space& exec, const viscosity::Operator& g) {
    switch (g.vertices_per_cell) {
    case 3: return cell_material<3,with_sa>(exec,g);
    case 4: return cell_material<4,with_sa>(exec,g);
    case 6: return cell_material<6,with_sa>(exec,g);
    default: return cell_material<8,with_sa>(exec,g);
    }
}

// Однородные сетки без плитки material_nodes читают номера вершин из Cell.
template<unsigned dimension,unsigned vertices,class Space>
void average_cells(const Space& exec,const Operator& g) {
    if (g.sa.active)
        for_indices(exec,"Aero: SA material",g.cell_count,KOKKOS_LAMBDA(Index e){g.average_material<dimension,vertices,Operator::Mixing::Sa>(e);});
    else
        for_indices(exec,"Aero: LES material",g.cell_count,KOKKOS_LAMBDA(Index e){g.average_material<dimension,vertices,Operator::Mixing::Sigma>(e);});
}

// Тип замыкания выбираем один раз на стадии, до запуска ядер Kokkos.
template<bool ddes,unsigned dimension,class Space>
void subgrid(const Space& exec,const viscosity::Operator& g) {
    if (dimension==3 && g.coordinate_filters && !g.tetra.all_hex)
        for_indices(exec,"Aero: mixed AES filter",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_compact_filter<ddes,0>(e);});
    else if (dimension==3 && ((g.affine && g.filter_extents) || g.coordinate_filters))
        for_indices(exec,"Aero: shared AES filter basis",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_compact_filter<ddes,8>(e);});
    else if (dimension==3 && g.hex_filter.coefficients && std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value)
        for_indices(exec,"Aero: hexahedral filter",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_hex<ddes>(e,g.filters);});
    else if (dimension==3 && g.hex_filter.coefficients)
        for_indices(exec,"Aero: hexahedral filter",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_hex<ddes>(e,g.filter_extents);});
    else if (dimension==3 && g.box_nodes)
        for_indices(exec,"Aero: rectangular filter",g.cell_count,KOKKOS_LAMBDA(Index i){g.sample_les<ddes,3,true>(i);});
    else
        for_indices(exec,"Aero: element filter",g.cell_count,KOKKOS_LAMBDA(Index i){g.sample_les<ddes,dimension>(i);});
    // Отдельный короткий обход оставляет регистры большого ядра SA свободными.
    if (!ddes || !std::is_same<Space,Kokkos::DefaultHostExecutionSpace>::value)
        for_indices(exec,"Aero: nodal filter",g.row_count,KOKKOS_LAMBDA(Index i){g.gather_filter<ddes>(i);});
}
#endif
// Одно ядро собирает невязку узла. Выбираем размерность до запуска, чтобы
// на плоской сетке временные массивы и циклы занимали только две компоненты.
template<unsigned dimension, viscosity::Operator::Gradients mode,class Space>
void viscous_rows(const Space& exec, const viscosity::Operator& g) {
    for_indices(exec, "Aero: viscous residual", g.row_count, KOKKOS_LAMBDA(Index node) {
        g.assemble_row<dimension,mode>(node);
    });
}

// Запускает сборку узловых вязких строк после подготовки элементных данных.
template<viscosity::Operator::Gradients mode,class Space>
void launch_viscous_rows(const Space& exec, const viscosity::Operator& g) {
    if (g.dim == 2) viscous_rows<2,mode>(exec,g);
    else if (g.dim == 3) viscous_rows<3,mode>(exec,g);
    else viscous_rows<1,mode>(exec,g);
}

// Выполняет фазы AES на имеющихся массивах. Подготовку газа можно пропустить после явной стадии.
template<class Space>
void evaluate(const Space& exec,const Operator& g,const double* state,const EulerState* cached,
              Evaluation phase,bool update_gas) {
    const viscosity::EvaluationPlan work(phase);
    work.execute(g, update_gas, [&] {
        for_indices(exec,"Aero: viscous gas",g.node_count,KOKKOS_LAMBDA(Index i) { g.prepare_gas(i,state,cached); });
    }, [&] {
#ifdef AERO_HAS_TURBULENCE
        work.select_filter(g.c_des, g.dim,
            [&] { subgrid<false,3>(exec,g); },
            [&] { subgrid<true,3>(exec,g); },
            [&] { subgrid<true,2>(exec,g); },
            [&] { subgrid<true,1>(exec,g); });
#endif
    }, [&] {
#ifdef AERO_HAS_TURBULENCE
        for_indices(exec,"Aero: SA eddy viscosity",g.node_count,KOKKOS_LAMBDA(Index i){g.prepare_sa_material(i);});
#endif
    }, [&] {
#ifdef AERO_HAS_TURBULENCE
        if (g.material_nodes) {
            if (g.sa.active) cell_material<true>(exec,g);
            else cell_material<false>(exec,g);
        }
        else if (g.vertices_per_cell==8 && g.dim==3 && (g.sa.active || g.eddy_nu)) average_cells<3,8>(exec,g);
        else if (g.vertices_per_cell==4 && g.dim==2 && (g.sa.active || g.eddy_nu)) average_cells<2,4>(exec,g);
        else
#endif
            for_indices(exec,"Aero: viscous material",g.cell_count,KOKKOS_LAMBDA(Index e){g.average_material(e);});
    }, [&](bool reuse_gradients) {
        using Mode=viscosity::Operator::Gradients;
        if (g.gradient_tile_count) {
            affine_aes::for_gradient_tiles(exec,g,[&](viscosity::GradientTile tile) {
                for_indices(exec,"Aero: accumulate AES tile",tile.slices,KOKKOS_LAMBDA(Index k) {
                    g.assemble_slice(g.gradient_slices[tile.first_slice+k],tile.first_vertex);
                });
            });
        } else if (g.element_order) {
            if (!reuse_gradients) {
                if (g.affine || g.tetra.cells) affine_aes::gradients(exec,g,g.gradient_cache,0,g.cell_count,0);
                else for_indices(exec,"Aero: AES gradients by element",g.vertex_count,KOKKOS_LAMBDA(Index k) { g.prepare_hex_gradient(k); });
            }
            if (g.tetra.cells && !g.tetra.all_hex)
                launch_viscous_rows<Mode::MixedElement>(exec,g);
            else
                launch_viscous_rows<Mode::Element>(exec,g);
        } else if (!g.gradient_cache)
            launch_viscous_rows<Mode::Direct>(exec,g);
        else if (reuse_gradients)
            launch_viscous_rows<Mode::Reuse>(exec,g);
        else
            launch_viscous_rows<Mode::Save>(exec,g);
    });
}
}
}
}
