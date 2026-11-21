// На NVIDIA Kokkos выполняет эти обходы в выбранном потоке устройства.
// Заголовки солвера содержат те же формулы, которые вызываются при обходе на CPU.
#define AERO_BUILD_DEVICE_WORK
#include <Kokkos_Core.hpp>
#include "device_work.h"
#include "../implicit/block_jacobi_detail.h"
#include "../implicit/csgs_sweep.h"
#include "../implicit/spatial_jacobian_detail.h"
#include "../implicit/bicgstab_vectors.h"
#include "../implicit/newton_vectors.h"
#include "device_execution.h"
#include "mpweno5_edges.h"
#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/jacobian.h"
#endif
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#if defined(_WIN32) && defined(AERO_STATIC_DEVICE_WORK) && !defined(__CUDA_ARCH__)
// cudart_static собрана MSVC с проверкой стека. При запуске через MinGW выполняем
// инициализацию её cookie до конструкторов CUDA, как это делает штатный CRT MSVC.
extern "C" void __cdecl __security_init_cookie();
namespace {
int __cdecl initialize_cuda_crt() { __security_init_cookie(); return 0; }
__attribute__((used, section(".CRT$XIB"))) int (__cdecl* const cuda_crt_initializer)() = initialize_cuda_crt;
}
#endif

namespace aero {
namespace device_work {
namespace {
using namespace residual_detail;
using Range = Kokkos::RangePolicy<Execution, Kokkos::IndexType<Index>, Kokkos::LaunchBounds<128>>;

// Запускаем Kokkos после выбора карты для данного MPI-процесса.
// Стандартный поток используется на каждой стадии, поэтому сохраняем исполнителя между вызовами.
class DeviceRuntime {
    int primary_; // Карта первого обращения. Ей принадлежит стандартный исполнитель Kokkos.
    std::map<int, Execution> defaults_; // Остальные карты для библиотечных вызовов из одного процесса.
    std::mutex defaults_mutex_; // Защищает создание исполнителя при обращениях из разных потоков CPU.
public:
    explicit DeviceRuntime(int device) : primary_(device) {
        Kokkos::InitArguments args;
        args.device_id = device;
        args.disable_warnings = true;
        Kokkos::initialize(args);
    }
    ~DeviceRuntime() {
        int original = primary_;
        cudaGetDevice(&original);
        while (!defaults_.empty()) {
            auto pos = defaults_.begin();
            cudaSetDevice(pos->first);
            defaults_.erase(pos);
        }
        cudaSetDevice(primary_);
        Kokkos::finalize();
        cudaSetDevice(original);
    }
    Execution execution(int device, cudaStream_t stream) {
        // Чужой поток может быть уничтожен сразу после возврата из библиотечного вызова.
        if (stream)
            return Execution(stream);
        if (device == primary_)
            return Execution();
        std::lock_guard<std::mutex> guard(defaults_mutex_);
        auto pos = defaults_.find(device);
        if (pos == defaults_.end())
            pos = defaults_.emplace(device, Execution(cudaStreamLegacy)).first;
        return pos->second;
    }
};

std::unique_ptr<DeviceRuntime>& runtime_holder() {
    static std::unique_ptr<DeviceRuntime> holder;
    return holder;
}
Execution select_execution(int device, cudaStream_t stream) {
    auto& runtime = runtime_holder();
    static std::once_flag initialized;
    std::call_once(initialized, [&] {
        runtime.reset(new DeviceRuntime(device));
        // Копирование полей может быть единственной операцией на GPU. В этом
        // случае host-обёртки вычислительных ядер вообще не вызываются. Регистрируем
        // завершение после инициализации внутренних объектов Kokkos, пока они ещё живы.
        if (std::atexit(aero_device_shutdown)) {
            runtime.reset();
            throw std::runtime_error("Unable to register Kokkos finalizer");
        }
    });
    return runtime->execution(device, stream);
}

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
template <bool roe_2d, bool nodal_only = false, unsigned linear_width = 0, unsigned lf_dim = 0>
void other_faces(const Execution& exec, FaceBatch batch, Index* bad) {
    if (!batch.indices && batch.count == batch.geometry.faces) {
        // В однопроцессном расчёте рёбра и внешние площадки уже лежат двумя непрерывными группами.
        // У каждого обхода свои регистры. Граничная характеристика не занимает регистры потока на ребре.
        const Index edges = batch.geometry.edges;
        for_indices(exec, "Aero: interior edges", edges, KOKKOS_LAMBDA(Index face) {
            if (!batch.compute_edge<roe_2d, nodal_only, linear_width>(face,lf_dim))
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
        if (!batch.compute_face<roe_2d,nodal_only,linear_width>(face,lf_dim))
            Kokkos::atomic_fetch_min(bad, face);
    });
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
    // У EBR3 и EBR5 различны только заранее рассчитанные веса. В 2D в узле четыре компоненты.
    else if (batch.linear_roe_2d())
        other_faces<true, false, 4>(exec, batch, bad);
    else if (batch.roe_2d())
        other_faces<true>(exec, batch, bad);
    // В 3D фиксированная ширина линейной строки равна пяти.
    else if (batch.geometry.dim == 3 && batch.geometry.stencil.row_owner &&
             !batch.geometry.stencil.weno_order && batch.geometry.options.variables != ResidualVariables::Flux)
        other_faces<false, false, 5>(exec, batch, bad);
    else
        other_faces<false>(exec, batch, bad);
}

// На CPU и GPU одна редукция сравнивает пары (шаг, узел), включая случай равных шагов.
template <class Read>
void minimum_step(const Execution& exec, Index count, Read read, CflStep& answer) {
    Kokkos::parallel_reduce("Aero: CFL", Range(exec, 0, count), cfl_detail::MinimumStep<Read>{read}, answer);
}

// Размер строки одинаков на всей сетке. При фиксированной ширине компилятор хранит
// компоненты невязки в регистрах во время обхода смежных рёбер узла.
KOKKOS_INLINE_FUNCTION void record_failed_node(const partition_detail::Grid& g, Index node) {
    // Место в списке резервирует только отказавший узел. Ёмкость равна числу владельцев.
    const Index slot = Kokkos::atomic_fetch_add(g.failed_count, Index(1));
    g.failed_nodes[slot] = node;
}

template <unsigned width>
void update_nodes(const Execution& exec, partition_detail::Grid g, explicit_detail::Stage stage,
                  double dt, bool save) {
    for_indices(exec, "Aero: owned nodes", g.owned_count, KOKKOS_LAMBDA(Index k) {
#ifdef AERO_HAS_TURBULENCE
        if (g.filtered_gas && g.filtered_gas->sa.active)
            g.filtered_gas->sa.advance_explicit(k,stage,dt,save);
#endif
        if (!partition_detail::update_node<width>(g.geometry, g.flux, g.volumes, g.owned, g.state,
                                                  g.original, k, stage, dt, save)) {
            Kokkos::atomic_fetch_min(g.bad, Index(g.owned ? g.owned[k] : k));
            return;
        }
#ifdef AERO_HAS_TURBULENCE
        // update_node уже записал примитивные переменные нового слоя. Из них
        // сразу готовим вязкость, пока давление и скорости находятся в кэше карты.
        if (g.filtered_gas) g.filtered_gas->prepare_gas(k,g.state,g.geometry.packed.gas);
#endif
        if (g.failed_count) {
            const Index node = g.owned ? g.owned[k] : k;
            // На цельной сетке update_node уже получил давление и пометил недопустимый газ.
            const bool ok = !g.owned && g.geometry.packed.gas ? g.geometry.packed.gas[node].density > 0.
                : partition_detail::admissible_node(g.geometry, g.state, node);
            if (!ok) record_failed_node(g, node);
        }
    });
}
}

Execution current_execution(void* stream) {
    int device;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    return select_execution(device, static_cast<cudaStream_t>(stream));
}
void aero_device_shutdown() {
    runtime_holder().reset();
}
const char* aero_device_jacobi_allocate(int device, implicit::JacobiBlocks& blocks) {
    return on_device(nullptr, [&](const Execution&) {
        // Пустому MPI-разделу не нужны буферы. Проверка номера карты уже выполнена.
        if (!blocks.nodes)
            return;
        Execution::memory_space memory;
        const auto rows = blocks.nodes * blocks.width;
        blocks.factors = static_cast<double*>(memory.allocate(rows * blocks.width * sizeof(double)));
        blocks.pivots = static_cast<unsigned char*>(memory.allocate(rows * sizeof(unsigned char)));
    }, device);
}
const char* aero_device_jacobi_release(int device, const implicit::JacobiBlocks& blocks) {
    return on_device(nullptr, [&](const Execution&) {
        Execution::memory_space memory;
        const auto rows = blocks.nodes * blocks.width;
        if (blocks.factors)
            memory.deallocate(blocks.factors, rows * blocks.width * sizeof(double));
        if (blocks.pivots)
            memory.deallocate(blocks.pivots, rows * sizeof(unsigned char));
    }, device);
}
const char* aero_device_jacobi_factor(int device, const implicit::JacobiBlocks& input, const double* diagonal,
                                       const MeshIndex* row_starts, MeshIndex* first_bad) {
    const auto blocks = input;
    return on_device(nullptr, [&](const Execution& exec) {
        // Одна строка редукции разлагает один узловой блок. На CPU нужен лишь
        // минимальный номер сбоя, чтобы не начинать итерации с вырожденной диагональю.
        Kokkos::parallel_reduce("Aero: factor block Jacobi", Range(exec, 0, blocks.nodes),
            KOKKOS_LAMBDA(Index node, MeshIndex& bad) {
                if (!blocks.factor_node(node, diagonal, row_starts) && node < bad)
                    bad = node;
            }, Kokkos::Min<MeshIndex>(*first_bad));
    }, device);
}
const char* aero_device_jacobi_apply(int device, const implicit::JacobiBlocks& input, const double* residual,
                                     double* correction) {
    const auto blocks = input;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: apply block Jacobi", blocks.nodes, KOKKOS_LAMBDA(Index node) {
            blocks.solve_node(node, residual, correction);
        });
    }, device);
}
const char* aero_device_pack_csgs(int device, const implicit::CsgsPacking& packing, const double* matrix) {
    const auto plan=packing;
    return on_device(nullptr,[&](const Execution& exec) {
        for_indices(exec,"Aero: prepare CSGS coefficients",plan.count,KOKKOS_LAMBDA(Index p){plan.copy(p,matrix);});
    },device);
}
const char* aero_device_csgs_image(int device,const implicit::CsgsSweep& input,MeshIndex count,double* image) {
    const auto sweep=input;
    return on_device(nullptr,[&](const Execution& exec) {
        if (sweep.lanes==32) for_indices(exec,"Aero: CSGS matrix product",count,KOKKOS_LAMBDA(Index k){sweep.image<32>(k,image);});
        else for_indices(exec,"Aero: CSGS matrix product",count,KOKKOS_LAMBDA(Index k){sweep.image<1>(k,image);});
    },device);
}
const char* aero_device_csgs_color(int device, const implicit::CsgsSweep& sweep,
                                   MeshIndex first, MeshIndex count) {
    const auto operation = sweep;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: CSGS color", count,
                    KOKKOS_LAMBDA(Index i) { operation(first+i); });
    }, device);
}
#ifdef AERO_HAS_VISCOSITY
const char* aero_device_viscous_matrix(int device, const viscosity::Jacobian& matrix, const double* state) {
    const auto v = matrix;
    const auto aes = v.aes;
    const unsigned width = aes.dim+2;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec,"Aero: clear viscous matrix",v.value_count,KOKKOS_LAMBDA(Index k) { v.values[k] = 0.; });
        for_indices(exec,"Aero: viscous Jacobian states",aes.node_count,KOKKOS_LAMBDA(Index n) { aes.prepare_gas(n,state,nullptr); });
        for_indices(exec,"Aero: viscous Jacobian gradients",aes.cell_count,KOKKOS_LAMBDA(Index e) { v.prepare_cell(e); });
        for_indices(exec,"Aero: viscous Jacobian columns",aes.node_count*width,KOKKOS_LAMBDA(Index k) {
            v.assemble_column(k/width,static_cast<unsigned>(k%width));
        });
    }, device);
}
#endif
const char* aero_device_spatial_faces(int device, const implicit::SpatialBlocks& matrix, const double* u,
                                      const double* initial, MeshIndex* bad) {
    const auto blocks = matrix;
    return on_device(nullptr, [&](const Execution& exec) {
        const auto light = Kokkos::Experimental::WorkItemProperty::HintLightWeight;
        Kokkos::parallel_reduce("Aero: differentiate face fluxes",
            Kokkos::Experimental::require(Range(exec, 0, blocks.face_count), light),
            KOKKOS_LAMBDA(Index k, MeshIndex& first) {
                const auto f = blocks.face(k);
                if (!blocks.differentiate(f, u, initial) && f < first) first = f;
            }, Kokkos::Min<MeshIndex>(*bad));
    }, device);
}
// На GPU соседние потоки считают соседние уравнения одного узла. Для каждого
// ребра они читают соседние строки блока из памяти. Индексы CSR общие для всех.
template<unsigned Width>
void assemble_matrix_rows(const Execution& exec, implicit::SpatialBlocks blocks,
                          const double* volumes, double dt, MeshIndex* bad) {
    const auto range=Kokkos::Experimental::require(Range(exec,0,blocks.row_count*Width),
                           Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_reduce("Aero: assemble Jacobian rows",range,KOKKOS_LAMBDA(Index i, MeshIndex& first) {
        const auto row=i/Width;
        if (!blocks.assemble_equations<Width,1>(row,static_cast<unsigned>(i%Width),volumes,dt)) {
            const auto node=blocks.node(row);
            if (node<first) first=node;
        }
    }, Kokkos::Min<MeshIndex>(*bad));
}
const char* aero_device_spatial_rows(int device, const implicit::SpatialBlocks& matrix,
                                     const double* volumes, double dt, MeshIndex* bad) {
    return on_device(nullptr, [&](const Execution& exec) {
        switch (matrix.grid.dim) {
        case 1: return assemble_matrix_rows<3>(exec,matrix,volumes,dt,bad);
        case 2: return assemble_matrix_rows<4>(exec,matrix,volumes,dt,bad);
        default: return assemble_matrix_rows<5>(exec,matrix,volumes,dt,bad);
        }
    }, device);
}
const char* aero_device_time_diagonal(int device, const implicit::SpatialBlocks& matrix,
                                      const double* volumes, double dt, MeshIndex* bad) {
    const auto blocks = matrix;
    return on_device(nullptr, [&](const Execution& exec) {
        // Передаём описание матрицы в аргументах ядра выбранной карты.
        // В одном процессе каждая карта использует свой поток исполнения.
        const auto range = Kokkos::Experimental::require(Range(exec, 0, blocks.row_count),
                                  Kokkos::Experimental::WorkItemProperty::HintLightWeight);
        Kokkos::parallel_reduce("Aero: update BDF time diagonal", range,
            KOKKOS_LAMBDA(Index row, MeshIndex& first) {
                if (!blocks.update_time_diagonal(row, volumes, dt) && blocks.node(row) < first)
                    first = blocks.node(row);
            }, Kokkos::Min<MeshIndex>(*bad));
    }, device);
}
template<unsigned Width>
void multiply_matrix_rows(const Execution& exec, implicit::SpatialBlocks blocks, const double* x, double* y) {
    for_indices(exec,"Aero: spatial Jacobian times vector",blocks.row_count*Width,KOKKOS_LAMBDA(Index i) {
        blocks.multiply_equations<Width,1>(i/Width,static_cast<unsigned>(i%Width),x,y);
    });
}
const char* aero_device_spatial_apply(int device, const implicit::SpatialBlocks& matrix,
                                      const double* x, double* y) {
    return on_device(nullptr, [&](const Execution& exec) {
        switch (matrix.grid.dim) {
        case 1: return multiply_matrix_rows<3>(exec,matrix,x,y);
        case 2: return multiply_matrix_rows<4>(exec,matrix,x,y);
        default: return multiply_matrix_rows<5>(exec,matrix,x,y);
        }
    }, device);
}
const char* aero_device_newton_update(int device, MeshIndex nodes, const implicit::NewtonUpdate& update) {
    const auto operation = update;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: BDF Newton", nodes, operation);
    }, device);
}
const char* aero_device_krylov_update(int device, MeshIndex count, const implicit::KrylovUpdate& update) {
    const auto operation = update;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: BiCGSTAB update", count, operation);
    }, device);
}
const char* aero_device_krylov_project(int device,MeshIndex count,const double* t,const double* r,double* pair) {
    return on_device(nullptr,[&](const Execution& exec){ implicit::krylov_project(exec,count,t,r,pair); },device);
}
const char* aero_device_krylov_update_norm(int device,MeshIndex count,const implicit::KrylovUpdate& step,double* norm) {
    return on_device(nullptr,[&](const Execution& exec){ *norm=implicit::krylov_update_norm(exec,count,step); },device);
}
const char* aero_device_krylov_reduce(int device, MeshIndex count, const double* x, const double* y, double* scalar) {
    return on_device(nullptr, [&](const Execution& exec) {
        *scalar = implicit::krylov_reduce(exec, count, x, y);
    }, device);
}
const char* aero_device_pressure_rate(const partition_detail::PressureRate& rate, Index count) {
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: pressure time derivative", count, rate);
    });
}
const char* aero_device_sample(const diagnostic::Sample& input,Index count) {
    const auto sample=input;
    return on_device(nullptr,[&](const Execution& exec){
        for_indices(exec,"Aero: accepted time sample",count,KOKKOS_LAMBDA(Index i){sample(i);});
    });
}
#ifdef AERO_HAS_VISCOSITY
#ifdef AERO_HAS_TURBULENCE
// Размерность известна всей задаче, поэтому выбираем ядро до обхода узлов.
// Внутри остаётся только формула для нужного числа компонент скорости.
template<unsigned dim,bool same_gas>
void sa_rows(const Execution& exec,const turbulence::Transport& transport,bool save_rotation) {
    for_indices(exec,"Aero: SA residual",transport.count,KOKKOS_LAMBDA(Index i) {
        transport.residual_row<dim,same_gas>(i,save_rotation);
    });
}
template<bool same_gas>
void sa_residual(const Execution& exec,const turbulence::Step& step) {
    switch (step.transport.dim) {
    case 1: sa_rows<1,same_gas>(exec,step.transport,step.keep_gas); break;
    case 2: sa_rows<2,same_gas>(exec,step.transport,step.keep_gas); break;
    default: sa_rows<3,same_gas>(exec,step.transport,step.keep_gas); break;
    }
}
const char* aero_device_sa_step(const turbulence::Step& input,turbulence::Operation op) {
    const auto step=input;
    return on_device(nullptr,[&](const Execution& exec){
        if (op==turbulence::Operation::LineRelax && step.transport.sa.lines.count) {
            for_indices(exec,"Aero: SA line relaxation",step.transport.sa.lines.count,KOKKOS_LAMBDA(Index k){step.relax_line(k);});
            return;
        }
        if (op==turbulence::Operation::Residual) {
            sa_residual<false>(exec,step);
            return;
        }
        for_indices(exec,"Aero: SA transport",step.transport.count,KOKKOS_LAMBDA(Index i){step.apply(op,i);});
    });
}
template<unsigned dim,bool refresh>
double sa_norm(const Execution& exec,const turbulence::Step& step,turbulence::Norm mode) {
    double norm=0.;
    if (step.transport.count) {
        using Range=Kokkos::RangePolicy<Execution,Kokkos::IndexType<Index>,Kokkos::LaunchBounds<128>>;
        Kokkos::parallel_reduce("Aero: SA maximum",Range(exec,0,step.transport.count),
            KOKKOS_LAMBDA(Index i,double& best) {
                const double v=refresh ? step.updated_defect<dim,false>(i) : step.magnitude(mode,i);
                if (v>best) best=v;
            },Kokkos::Max<double>(norm));
    }
    return norm;
}
const char* aero_device_sa_norm(const turbulence::Step& input,turbulence::Norm mode,bool refresh,double* result) {
    const auto step=input;
    return on_device(nullptr,[&](const Execution& exec){
        if (!refresh) { *result=sa_norm<0,false>(exec,step,mode); return; }
        for_indices(exec,"Aero: SA element diffusivity",step.transport.cell_count,
                    KOKKOS_LAMBDA(Index e){step.transport.refresh_diffusivity(e);});
        switch (step.transport.dim) {
        case 1: *result=sa_norm<1,true>(exec,step,mode); break;
        case 2: *result=sa_norm<2,true>(exec,step,mode); break;
        default: *result=sa_norm<3,true>(exec,step,mode); break;
        }
    });
}
#endif
#ifdef AERO_HAS_TURBULENCE
// Тип замыкания выбираем один раз на стадии, до запуска ядер Kokkos.
template<bool ddes,unsigned dimension>
void subgrid_on_device(const Execution& exec,const viscosity::Operator& g) {
    if (dimension==3 && g.hex_filter.nodes)
        for_indices(exec,"Aero: hexahedral filter",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_hex<ddes>(e);});
    else if (dimension==3 && g.box_nodes)
        for_indices(exec,"Aero: rectangular filter",g.cell_count,KOKKOS_LAMBDA(Index i){g.sample_les<ddes,3,true>(i);});
    else
        for_indices(exec,"Aero: element filter",g.cell_count,KOKKOS_LAMBDA(Index i){g.sample_les<ddes,dimension>(i);});
    // Отдельный короткий обход оставляет регистры большого ядра SA свободными.
    for_indices(exec,"Aero: nodal filter",g.row_count,KOKKOS_LAMBDA(Index i){g.gather_filter<ddes>(i);});
}
#endif
const char* aero_device_viscosity(const viscosity::Operator& input, const double* state, const EulerState* cached,viscosity::Evaluation phase) {
    const auto g = input;
    return on_device(nullptr, [&](const Execution& exec) {
        const viscosity::EvaluationPlan work(phase);
        if (g.needs_gas(phase))
            for_indices(exec,"Aero: viscous gas",g.node_count,KOKKOS_LAMBDA(Index i) { g.prepare_gas(i,state,cached); });
#ifdef AERO_HAS_TURBULENCE
        if (work.filter && !g.skip_ddes_filter) {
            if (g.c_des==0.) subgrid_on_device<false,3>(exec,g);
            else if (g.dim==3) subgrid_on_device<true,3>(exec,g);
            else subgrid_on_device<true,0>(exec,g);
        }
        if (work.sa_material)
            for_indices(exec,"Aero: SA eddy viscosity",g.node_count,KOKKOS_LAMBDA(Index i){g.prepare_sa_material(i);});
#endif
        if (work.cell_material) {
#ifdef AERO_HAS_TURBULENCE
            if (g.material_nodes) for_indices(exec,"Aero: turbulent material",g.cell_count,KOKKOS_LAMBDA(Index e){g.average_turbulent_material(e);});
            else
#endif
                for_indices(exec,"Aero: viscous material",g.cell_count,KOKKOS_LAMBDA(Index e){g.average_material(e);});
        }
        if (!work.rows) return;
        using Mode=viscosity::Operator::Gradients;
        if (g.element_order) {
            if (!work.reuse_gradients)
                for_indices(exec,"Aero: AES gradients by element",g.vertex_count,KOKKOS_LAMBDA(Index k) { g.prepare_hex_gradient(k); });
            for_indices(exec,"Aero: AES assembled gradients",g.row_count,KOKKOS_LAMBDA(Index n) { g.assemble_row<3,Mode::Element>(n); });
        } else if (!g.gradient_cache)
            for_indices(exec,"Aero: AES edge contributions",g.row_count,KOKKOS_LAMBDA(Index i) { g.assemble(i); });
        else if (work.reuse_gradients)
            for_indices(exec,"Aero: AES fixed gas",g.row_count,KOKKOS_LAMBDA(Index i) { g.assemble<Mode::Reuse>(i); });
        else
            for_indices(exec,"Aero: AES save gradients",g.row_count,KOKKOS_LAMBDA(Index i) { g.assemble<Mode::Save>(i); });
    });
}
#endif
const char* aero_device_normals(const Geometry& geometry) {
    const Geometry g = geometry;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: face normals", g.edges, KOKKOS_LAMBDA(Index i) {
            euler_unit_direction(g.dim, g.surfaces[i].area, g.packed.normals[i].axis);
        });
    });
}
const char* aero_device_faces(const FaceBatch& input, Index* bad, void* stream, const FaceBatch* device_faces) {
    return on_device(stream, [&](const Execution& exec) {
        FaceBatch batch = input;
        const auto& packed = batch.geometry.packed;
        if (batch.needs_gas_update()) {
            // Ядру нужны только поля узлов. Передаём адреса и два параметра газа,
            // чтобы nvcc не включал в параметры ядра всю геометрию рёбер.
            const unsigned dimension = batch.geometry.dim;
            const double gamma = batch.geometry.options.gamma;
            const double* state = batch.state;
            EulerState* gas = packed.gas;
            for_indices(exec, "Aero: nodal gas", packed.state_nodes, KOKKOS_LAMBDA(Index i) {
                decode_nodal_gas(dimension, state, gamma, i, gas);
            });
        }
        if (packed.tiles)
            compute_packed_fluxes(exec, batch, bad, device_faces);
        else
            compute_unpacked_fluxes(exec, batch, bad);
    });
}
const char* aero_device_gather(const Geometry& geometry, const EulerFluxResult* flux, double* residual,
                               Index* bad, void* stream) {
    const Geometry g = geometry;
    return on_device(stream, [&](const Execution& exec) {
        for_indices(exec, "Aero: node residuals", g.nodes, KOKKOS_LAMBDA(Index i) {
            if (!assemble_node_residual(g, i, flux, g.node_row(residual, i)))
                Kokkos::atomic_fetch_min(bad, g.faces+i);
        });
    });
}
const char* aero_device_local_steps(const LocalSteadySteps& steps, StepRange* range) {
    return on_device(nullptr, [&](const Execution& exec) {
        Kokkos::parallel_reduce("Aero: local steady steps", Range(exec, 0, steps.gas.count), steps, *range);
    });
}
const char* aero_device_limit_local_steps(const LimitLocalSteps& limit, Index count) {
    return on_device(nullptr, [&](const Execution& exec) {
        Kokkos::parallel_for("Aero: limit local steady steps", Range(exec, 0, count), limit);
    });
}
const char* aero_device_flux_step(const FluxCfl& input, double courant, CflStep* answer, void* stream) {
    const FluxCfl grid = input;
    return on_device(stream, [&](const Execution& exec) {
        minimum_step(exec, grid.count, KOKKOS_LAMBDA(Index k) { return grid.local(k, courant); }, *answer);
    });
}
const char* aero_device_wave_step(cfl_detail::Grid grid, WaveSpeeds speeds, double courant, CflStep* answer,
                                  void* stream) {
    return on_device(stream, [&](const Execution& exec) {
        minimum_step(exec, grid.nodes, KOKKOS_LAMBDA(Index k) {
            return cfl_detail::local(grid, speeds, courant, k);
        }, *answer);
    });
}
const char* aero_device_stage(explicit_detail::FieldStageUpdate update, size_t count, int* bad, void* stream) {
    return on_device(stream, [&](const Execution& exec) {
        for_indices(exec, "Aero: explicit stage", count, KOKKOS_LAMBDA(Index k) {
            if (!update.apply(k))
                Kokkos::atomic_exchange(bad, 1);
        });
    });
}
// На итерации Ньютона новое состояние формирует другое ядро. Проверяем его после обмена.
const char* aero_device_failed_nodes(const partition_detail::Grid& input) {
    const auto g = input;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: validate Newton state", g.owned_count, KOKKOS_LAMBDA(Index k) {
            const Index node = g.owned ? g.owned[k] : k;
            if (!partition_detail::admissible_node(g.geometry, g.state, node)) record_failed_node(g, node);
        });
    });
}
const char* aero_device_owned_stage(const partition_detail::Grid& input, explicit_detail::Stage stage,
                                    double dt, bool save) {
    const auto g = input;
    return on_device(nullptr, [&](const Execution& exec) {
        switch (g.geometry.dim) {
        case 1: update_nodes<3>(exec, g, stage, dt, save); break;
        case 2: update_nodes<4>(exec, g, stage, dt, save); break;
        default: update_nodes<5>(exec, g, stage, dt, save); break;
        }
    });
}
const char* aero_device_records(const partition_detail::RecordField& data, const MeshIndex* ids, Index count,
                                const double* input, double* output) {
    const auto field = data;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: MPI records", count, KOKKOS_LAMBDA(Index k) {
            field.copy_record(ids ? ids[k] : k, k, input, output);
        });
    });
}
}
}
