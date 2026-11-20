// На NVIDIA Kokkos выполняет эти обходы в выбранном потоке устройства.
// Заголовки солвера содержат те же формулы, которые вызываются при обходе на CPU.
#define AERO_BUILD_DEVICE_WORK
#include <Kokkos_Core.hpp>
#include "device_work.h"
#include "mpweno5_edges.h"
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
using Execution = Kokkos::Cuda;
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
    std::call_once(initialized, [&] { runtime.reset(new DeviceRuntime(device)); });
    return runtime->execution(device, stream);
}

// Исключения библиотеки остаются внутри модуля. На хост уходит только текст ошибки.
template <class Work>
const char* on_device(void* stream, Work work) {
    static thread_local std::string error;
    try {
        int device;
        const auto status = cudaGetDevice(&device);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
        const Execution exec = select_execution(device, static_cast<cudaStream_t>(stream));
        work(exec);
        return nullptr;
    } catch (const std::exception& e) {
        error = e.what();
        return error.c_str();
    }
}

// Одна итерация соответствует одному индексу. Kokkos выбирает сетку запуска.
// HintLightWeight помещает аргументы небольшого ядра непосредственно в его параметры.
template <unsigned min_blocks = 0, class Work>
void for_indices(const Execution& exec, const char* name, Index count, Work work) {
    using Policy = Kokkos::RangePolicy<Execution, Kokkos::IndexType<Index>, Kokkos::LaunchBounds<128, min_blocks>>;
    if (count)
        Kokkos::parallel_for(name, Kokkos::Experimental::require(Policy(exec, 0, count),
                             Kokkos::Experimental::WorkItemProperty::HintLightWeight), work);
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
        const Index* indices = batch.indices;
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
            mpweno_edges::project(*stored, face, work[k / MpEdgeTile::width], k % MpEdgeTile::width, bad);
        });
        mpweno_edges::reconstruct(exec, work, count, batch.geometry.stencil.mp5);
        for_indices(exec, "Aero: MPWENO5 complete edge flux", count, KOKKOS_LAMBDA(Index k) {
            const Index face = indices ? indices[first + k] : first + k;
            mpweno_edges::write_flux(*stored, face, work[k / MpEdgeTile::width], k % MpEdgeTile::width, bad);
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
template <bool roe_2d, bool nodal_only = false, unsigned linear_width = 0>
void other_faces(const Execution& exec, FaceBatch batch, Index* bad) {
    if (!batch.indices && batch.count == batch.geometry.faces) {
        // В однопроцессном расчёте рёбра и внешние площадки уже лежат двумя непрерывными группами.
        // У каждого обхода свои регистры. Граничная характеристика не занимает регистры потока на ребре.
        const Index edges = batch.geometry.edges;
        for_indices(exec, "Aero: interior edges", edges, KOKKOS_LAMBDA(Index face) {
            if (!batch.compute_edge<roe_2d, nodal_only, linear_width>(face))
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
        if (!(roe_2d ? batch.compute_roe_2d<nodal_only, linear_width>(face) : batch.compute(face)))
            Kokkos::atomic_fetch_min(bad, face);
    });
}

// Полные шаблоны WENO и MP уложены по 32 ребра. Выбираем ядро один раз на пакет.
void compute_packed_fluxes(const Execution& exec, FaceBatch batch, Index* bad, const FaceBatch* device_faces) {
    const auto& packed = batch.geometry.packed;
    if (device_faces && packed.mp_work)
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
    const Index handled_mixed = device_faces && packed.mp_work ? packed.mixed_edges : 0;
    remaining_group<RemainingFace::Sparse>(exec, batch, device_faces, 0, packed.sparse_edges - handled_mixed, bad);
    remaining_group<RemainingFace::Nodal>(exec, batch, device_faces, packed.sparse_edges, packed.nodal_edges, bad);
    remaining_group<RemainingFace::Boundary>(exec, batch, device_faces, boundary, packed.count - boundary, bad);
}

// Для первого порядка, EBR и сеток без упаковки используем обычный список площадок.
void compute_unpacked_fluxes(const Execution& exec, FaceBatch batch, Index* bad) {
    // Первый порядок читает два узла без обращения к строкам шаблона.
    if (batch.nodal_roe_2d())
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
template <unsigned width>
void update_nodes(const Execution& exec, partition_detail::Grid g, explicit_detail::Stage stage,
                  double dt, bool save) {
    for_indices(exec, "Aero: owned nodes", g.owned_count, KOKKOS_LAMBDA(Index k) {
        if (!partition_detail::update_node<width>(g.geometry, g.flux, g.volumes, g.owned, g.state,
                                                  g.original, k, stage, dt, save))
            Kokkos::atomic_fetch_min(g.bad, Index(g.owned ? g.owned[k] : k));
    });
}
}

void aero_device_shutdown() {
    runtime_holder().reset();
}
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
        const auto packed = batch.geometry.packed;
        if (batch.needs_gas_update())
            for_indices(exec, "Aero: nodal gas", packed.state_nodes, KOKKOS_LAMBDA(Index i) {
                decode_nodal_gas(batch.geometry.dim, batch.state, batch.geometry.options.gamma, i, packed.gas);
            });
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
const char* aero_device_records(const partition_detail::Grid& input, const MeshIndex* ids, Index count,
                                double* records, bool edges, bool pack) {
    const auto g = input;
    return on_device(nullptr, [&](const Execution& exec) {
        const unsigned width = g.geometry.dim + 2;
        for_indices(exec, "Aero: MPI records", count, KOKKOS_LAMBDA(Index k) {
            partition_detail::transfer_record(ids ? ids[k] : k, width, g.state, g.flux,
                                               records + k * (width + unsigned(edges)), edges, pack);
        });
    });
}
}
}
