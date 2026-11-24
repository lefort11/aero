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
#ifdef AERO_HAS_MULTIGRID
#include "../multigrid/transfer.h"
#endif
#include "device_execution.h"
#include "device_faces.h"
#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/jacobian.h"
#include "device_affine_aes.h"
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
// Подготавливает CUDA Runtime до создания контекстов Kokkos.
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
    // Инициализирует Kokkos и контексты выполнения на заданной карте.
    explicit DeviceRuntime(int device) : primary_(device) {
        Kokkos::InitArguments args;
        args.device_id = device;
        args.disable_warnings = true;
        Kokkos::initialize(args);
    }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
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
    // Возвращает контекст исполнения для требуемого устройства или потока.
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

// Возвращает единственный объект управления средой GPU в текущем модуле.
std::unique_ptr<DeviceRuntime>& runtime_holder() {
    static std::unique_ptr<DeviceRuntime> holder;
    return holder;
}
// Выбирает устройство и соответствующий контекст выполнения ядра.
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

// Запускает поузловое обновление поля на выбранном контексте Kokkos.
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

// Возвращает контекст Kokkos для выбранного устройства и потока CUDA.
Execution current_execution(void* stream) {
    int device;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    return select_execution(device, static_cast<cudaStream_t>(stream));
}
// Завершает среду выполнения вычислительных ядер GPU.
void aero_device_shutdown() {
    runtime_holder().reset();
}
// Выделяет память диагональных блоков предобусловливателя на GPU.
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
// Освобождает память диагональных блоков GPU.
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
// Разлагает диагональные блоки параллельно на GPU.
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
// Решает независимые диагональные системы на GPU.
const char* aero_device_jacobi_apply(int device, const implicit::JacobiBlocks& input, const double* residual,
                                     double* correction) {
    const auto blocks = input;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: apply block Jacobi", blocks.nodes, KOKKOS_LAMBDA(Index node) {
            blocks.solve_node(node, residual, correction);
        });
    }, device);
}
// Упаковывает коэффициенты матрицы в порядок проходов CSGS.
const char* aero_device_pack_csgs(int device, const implicit::CsgsPacking& packing, const double* matrix) {
    const auto plan=packing;
    return on_device(nullptr,[&](const Execution& exec) {
        for_indices(exec,"Aero: prepare CSGS coefficients",plan.count,KOKKOS_LAMBDA(Index p){plan.copy(p,matrix);});
    },device);
}
// Вычисляет матричный образ для прохода CSGS на GPU.
const char* aero_device_csgs_image(int device,const implicit::CsgsSweep& input,MeshIndex count,double* image) {
    const auto sweep=input;
    return on_device(nullptr,[&](const Execution& exec) {
        if (sweep.lanes==32) for_indices(exec,"Aero: CSGS matrix product",count,KOKKOS_LAMBDA(Index k){sweep.image<32>(k,image);});
        else for_indices(exec,"Aero: CSGS matrix product",count,KOKKOS_LAMBDA(Index k){sweep.image<1>(k,image);});
    },device);
}
// Обновляет независимые строки одного цвета CSGS на GPU.
const char* aero_device_csgs_color(int device, const implicit::CsgsSweep& sweep,
                                   MeshIndex first, MeshIndex count) {
    const auto operation = sweep;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: CSGS color", count,
                    KOKKOS_LAMBDA(Index i) { operation(first+i); });
    }, device);
}
#ifdef AERO_HAS_VISCOSITY
// Собирает вязкие производные в блоки пространственной матрицы на GPU.
const char* aero_device_viscous_matrix(int device, const viscosity::Jacobian& matrix, const double* state) {
    const auto v = matrix;
    const auto aes = v.aes;
    const unsigned width = aes.dim+2;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec,"Aero: clear viscous matrix",v.value_count,KOKKOS_LAMBDA(Index k) { v.values[k] = 0.; });
        if (v.single_count) for_indices(exec,"Aero: clear mixed off-diagonal",v.single_count,
                                        KOKKOS_LAMBDA(Index k) { v.layout.neighbours[k] = 0.f; });
        for_indices(exec,"Aero: viscous Jacobian states",aes.node_count,KOKKOS_LAMBDA(Index n) { aes.prepare_gas(n,state,nullptr); });
        if (aes.affine || aes.tetra.cells) {
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
                return;
            }
            affine_aes::gradients(exec,aes,v.gradients,0,aes.cell_count,0);
        } else {
            for_indices(exec,"Aero: viscous Jacobian gradients",aes.cell_count,KOKKOS_LAMBDA(Index e) { v.prepare_cell(e); });
        }
        for_indices(exec,"Aero: viscous Jacobian columns",aes.node_count*width,KOKKOS_LAMBDA(Index k) {
            v.assemble_column(k/width,static_cast<unsigned>(k%width));
        });
    }, device);
}
#endif
// Вычисляет производные потоков площадок на GPU.
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
template<unsigned Width,bool direct=false>
void assemble_matrix_rows(const Execution& exec, implicit::SpatialBlocks blocks,
                          const double* volumes, double dt, MeshIndex* bad) {
    // При вычислении производной на месте один исполнитель сразу записывает
    // весь блок: пять компонент не повторяют вычисление потока.
    constexpr unsigned lanes=direct ? 1u : Width;
    const auto range=Kokkos::Experimental::require(Range(exec,0,blocks.row_count*lanes),
                           Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    Kokkos::parallel_reduce("Aero: assemble Jacobian rows",range,KOKKOS_LAMBDA(Index i, MeshIndex& first) {
        const auto row=i/lanes;
        if (!blocks.assemble_equations<Width,direct ? Width : 1,direct>(row,static_cast<unsigned>(i%lanes),volumes,dt)) {
            const auto node=blocks.node(row);
            if (node<first) first=node;
        }
    }, Kokkos::Min<MeshIndex>(*bad));
}
// Собирает строки пространственного якобиана из производных площадок.
const char* aero_device_spatial_rows(int device, const implicit::SpatialBlocks& matrix,
                                     const double* volumes, double dt, MeshIndex* bad) {
    return on_device(nullptr, [&](const Execution& exec) {
        if (matrix.direct_faces) return assemble_matrix_rows<5,true>(exec,matrix,volumes,dt,bad);
        switch (matrix.grid.dim) {
        case 1: return assemble_matrix_rows<3>(exec,matrix,volumes,dt,bad);
        case 2: return assemble_matrix_rows<4>(exec,matrix,volumes,dt,bad);
        default: return assemble_matrix_rows<5>(exec,matrix,volumes,dt,bad);
        }
    }, device);
}
// Обновляет временные диагональные коэффициенты системы на GPU.
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
// Умножает строки блочной матрицы на вектор с заданным типом хранения коэффициентов.
template<unsigned Width>
void multiply_matrix_rows(const Execution& exec, implicit::SpatialBlocks blocks, const double* x, double* y) {
    for_indices(exec,"Aero: spatial Jacobian times vector",blocks.row_count*Width,KOKKOS_LAMBDA(Index i) {
        blocks.multiply_equations<Width,1>(i/Width,static_cast<unsigned>(i%Width),x,y);
    });
}
// Умножает распределённую пространственную матрицу на вектор на GPU.
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
#ifdef AERO_HAS_MULTIGRID
// Выполняет ограничение или продление многосеточного вектора на GPU.
const char* aero_device_multigrid_transfer(int device, MeshIndex count, const multigrid::Transfer& transfer) {
    return on_device(nullptr,[&](const Execution& exec) {
        for_indices(exec,"Aero: steady multigrid transfer",count,transfer);
    },device);
}
// Обновляет рабочие векторы грубосеточной коррекции на GPU.
const char* aero_device_multigrid_update(int device, MeshIndex count, const multigrid::VectorUpdate& update) {
    return on_device(nullptr,[&](const Execution& exec) {
        for_indices(exec,"Aero: steady multigrid coarse vectors",count,update);
    },device);
}
#endif
// Вычисляет невязку неявного шага и её норму на GPU.
const char* aero_device_newton_residual(int device, MeshIndex rows, const implicit::NewtonUpdate& update, double* norm) {
    return on_device(nullptr,[&](const Execution& exec) {
        *norm = implicit::form_residual_and_norm(exec,rows,update);
    },device);
}
// Применяет выбранное обновление полей Ньютона на GPU.
const char* aero_device_newton_update(int device, MeshIndex nodes, const implicit::NewtonUpdate& update) {
    const auto operation = update;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: BDF Newton", nodes, operation);
    }, device);
}
// Обновляет рабочие векторы BiCGSTAB на GPU.
const char* aero_device_krylov_update(int device, MeshIndex count, const implicit::KrylovUpdate& update) {
    const auto operation = update;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: BiCGSTAB update", count, operation);
    }, device);
}
// Вычисляет скалярные произведения для стабилизации BiCGSTAB на GPU.
const char* aero_device_krylov_project(int device,MeshIndex count,const double* t,const double* r,double* pair) {
    return on_device(nullptr,[&](const Execution& exec){ implicit::krylov_project(exec,count,t,r,pair); },device);
}
// Совмещает обновление вектора Крылова с вычислением нормы на GPU.
const char* aero_device_krylov_update_norm(int device,MeshIndex count,const implicit::KrylovUpdate& step,double* norm) {
    return on_device(nullptr,[&](const Execution& exec){ *norm=implicit::krylov_update_norm(exec,count,step); },device);
}
// Вычисляет скалярное произведение или максимальную норму вектора на GPU.
const char* aero_device_krylov_reduce(int device, MeshIndex count, const double* x, const double* y, double* scalar) {
    return on_device(nullptr, [&](const Execution& exec) {
        *scalar = implicit::krylov_reduce(exec, count, x, y);
    }, device);
}
// Вычисляет диагностическую производную давления по времени на GPU.
const char* aero_device_pressure_rate(const partition_detail::PressureRate& rate, Index count) {
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: pressure time derivative", count, rate);
    });
}
// Добавляет вклад текущего поля во временные моменты на GPU.
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
// Собирает локальные невязки уравнения SA на GPU.
template<bool same_gas>
void sa_residual(const Execution& exec,const turbulence::Step& step) {
    // На первом обходе неявного газа сохраняем ротор. Во внутренних итерациях
    // меняется только nu_tilde, поэтому берём уже сохранённый ротор.
    const bool save_rotation=!same_gas && step.keep_gas;
    const auto launch = for_dimension(step.transport.dim, sa_rows<1,same_gas>,
                                      sa_rows<2,same_gas>, sa_rows<3,same_gas>);
    launch(exec, step.transport, save_rotation);
}
// Собирает конвективный перенос рабочей вязкости SA на GPU.
const char* aero_device_sa_convection(const turbulence::SaEdgeConvection& edges,MeshIndex count) {
    return on_device(nullptr,[&](const Execution& exec){
        for_indices(exec,"Aero: reconstructed SA convection",count,edges);
    });
}
// Выполняет выбранную операцию шага SA на GPU.
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
// Выполняет редукцию максимальной величины SA на GPU.
double sa_norm(const Execution& exec,const turbulence::Step& step,turbulence::Norm mode) {
    double norm=0.;
    if (step.transport.count) {
        using Range=Kokkos::RangePolicy<Execution,Kokkos::IndexType<Index>,Kokkos::LaunchBounds<128>>;
        Kokkos::parallel_reduce("Aero: SA maximum",Range(exec,0,step.transport.count),
            KOKKOS_LAMBDA(Index i,double& best) {
                const double v=step.magnitude(mode,i);
                if (v>best) best=v;
            },Kokkos::Max<double>(norm));
    }
    return norm;
}
// Вычисляет максимальную норму выбранной величины SA на GPU.
const char* aero_device_sa_norm(const turbulence::Step& input,turbulence::Norm mode,bool refresh,double* result) {
    const auto step=input;
    return on_device(nullptr,[&](const Execution& exec){
        if (refresh) {
            for_indices(exec,"Aero: SA element diffusivity",step.transport.cell_count,
                        KOKKOS_LAMBDA(Index e){step.transport.refresh_diffusivity(e);});
            // Сборка строки читает площадки и пары AES. Её запускаем по всем
            // узлам обычным parallel_for, у которого достаточно блоков и на
            // большой сетке. Максимум использует готовые rate и diagonal.
            sa_residual<true>(exec,step);
        }
        *result=sa_norm(exec,step,refresh ? turbulence::Norm::Bdf : mode);
    });
}
#endif
#ifdef AERO_HAS_TURBULENCE
// На сетке из однотипных элементов число вершин и наличие уравнения SA
// одинаковы у всей группы. Выбираем их до запуска, чтобы в цикле по вершинам
// сразу собирать нужные коэффициенты переноса.
template<unsigned vertices,bool with_sa>
void cell_material(const Execution& exec, const viscosity::Operator& g) {
    for_indices(exec,"Aero: turbulent material",g.cell_count,KOKKOS_LAMBDA(Index e){
        g.average_cached_material<vertices,with_sa>(e);
    });
}
// На сетке из однотипных элементов число вершин и наличие уравнения SA
// одинаковы у всей группы. Выбираем их до запуска, чтобы в цикле по вершинам
// сразу собирать нужные коэффициенты переноса.
template<bool with_sa>
void cell_material(const Execution& exec, const viscosity::Operator& g) {
    switch (g.vertices_per_cell) {
    case 3: return cell_material<3,with_sa>(exec,g);
    case 4: return cell_material<4,with_sa>(exec,g);
    case 6: return cell_material<6,with_sa>(exec,g);
    default: return cell_material<8,with_sa>(exec,g);
    }
}

// Тип замыкания выбираем один раз на стадии, до запуска ядер Kokkos.
template<bool ddes,unsigned dimension>
void subgrid_on_device(const Execution& exec,const viscosity::Operator& g) {
    if (dimension==3 && g.coordinate_filters && !g.tetra.all_hex)
        for_indices(exec,"Aero: mixed AES filter",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_compact_filter<ddes,0>(e);});
    else if (dimension==3 && ((g.affine && g.filter_extents) || g.coordinate_filters))
        for_indices(exec,"Aero: shared AES filter basis",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_compact_filter<ddes,8>(e);});
    else if (dimension==3 && g.hex_filter.coefficients)
        for_indices(exec,"Aero: hexahedral filter",g.cell_count,KOKKOS_LAMBDA(Index e){g.sample_hex<ddes>(e,g.filter_extents);});
    else if (dimension==3 && g.box_nodes)
        for_indices(exec,"Aero: rectangular filter",g.cell_count,KOKKOS_LAMBDA(Index i){g.sample_les<ddes,3,true>(i);});
    else
        for_indices(exec,"Aero: element filter",g.cell_count,KOKKOS_LAMBDA(Index i){g.sample_les<ddes,dimension>(i);});
    // Отдельный короткий обход оставляет регистры большого ядра SA свободными.
    for_indices(exec,"Aero: nodal filter",g.row_count,KOKKOS_LAMBDA(Index i){g.gather_filter<ddes>(i);});
}
#endif
// Одно ядро собирает невязку узла. Выбираем размерность до запуска, чтобы
// на плоской сетке временные массивы и циклы занимали только две компоненты.
template<unsigned dimension, viscosity::Operator::Gradients mode>
void viscous_rows(const Execution& exec, const viscosity::Operator& g) {
    for_indices(exec, "Aero: viscous residual", g.row_count, KOKKOS_LAMBDA(Index node) {
        g.assemble_row<dimension,mode>(node);
    });
}

// Запускает сборку узловых вязких строк после подготовки элементных данных.
template<viscosity::Operator::Gradients mode>
void launch_viscous_rows(const Execution& exec, const viscosity::Operator& g) {
    if (g.dim == 2) viscous_rows<2,mode>(exec,g);
    else if (g.dim == 3) viscous_rows<3,mode>(exec,g);
    else viscous_rows<1,mode>(exec,g);
}

// Подготавливает объёмные множители тетраэдрального разбиения на GPU.
const char* aero_device_prepare_tetra_volumes(const viscosity::Operator& input) {
    return on_device(nullptr,[&](const Execution& exec){coordinate_aes::prepare_volumes(exec,input.tetra);});
}
// Запускает требуемые фазы вязкого оператора AES на GPU.
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
            else if (g.dim == 2) subgrid_on_device<true,2>(exec,g);
            else subgrid_on_device<true,1>(exec,g);
        }
        if (work.sa_material)
            for_indices(exec,"Aero: SA eddy viscosity",g.node_count,KOKKOS_LAMBDA(Index i){g.prepare_sa_material(i);});
#endif
        if (work.cell_material) {
#ifdef AERO_HAS_TURBULENCE
            if (g.material_nodes) {
                if (g.sa.active) cell_material<true>(exec,g);
                else cell_material<false>(exec,g);
            }
            else
#endif
                for_indices(exec,"Aero: viscous material",g.cell_count,KOKKOS_LAMBDA(Index e){g.average_material(e);});
        }
        if (!work.rows) return;
        using Mode=viscosity::Operator::Gradients;
        if (g.gradient_tile_count) {
            affine_aes::for_gradient_tiles(exec,g,[&](viscosity::GradientTile tile) {
                for_indices(exec,"Aero: accumulate AES tile",tile.slices,KOKKOS_LAMBDA(Index k) {
                    g.assemble_slice(g.gradient_slices[tile.first_slice+k],tile.first_vertex);
                });
            });
        } else if (g.element_order) {
            if (!work.reuse_gradients) {
                if (g.affine || g.tetra.cells) affine_aes::gradients(exec,g,g.gradient_cache,0,g.cell_count,0);
                else for_indices(exec,"Aero: AES gradients by element",g.vertex_count,KOKKOS_LAMBDA(Index k) { g.prepare_hex_gradient(k); });
            }
            if (g.tetra.cells && !g.tetra.all_hex)
                launch_viscous_rows<Mode::MixedElement>(exec,g);
            else
                launch_viscous_rows<Mode::Element>(exec,g);
        } else if (!g.gradient_cache)
            launch_viscous_rows<Mode::Direct>(exec,g);
        else if (work.reuse_gradients)
            launch_viscous_rows<Mode::Reuse>(exec,g);
        else
            launch_viscous_rows<Mode::Save>(exec,g);
    });
}
#endif
// Подготавливает единичные нормали площадок в базисе решателя.
const char* aero_device_normals(const Geometry& geometry) {
    const Geometry g = geometry;
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: face normals", g.edges, KOKKOS_LAMBDA(Index i) {
            euler_unit_direction(g.dim, g.surfaces[i].area, g.packed.normals[i].axis);
        });
    });
}
// Именованный тип даёт одному и тому же ядру одинаковый символ на хосте и на CUDA.
template<unsigned W>
struct SplitFluxFaces {
    const FaceBatch* batch;
    const MeshIndex* indices;
    Index* bad;
    KOKKOS_INLINE_FUNCTION void operator()(Index k) const {
        const Index face=FaceBatch::face(indices,k);
        if (batch->is_boundary(face)) {
            if (!batch->compute_boundary(face)) Kokkos::atomic_fetch_min(bad,face);
        } else if (residual_detail::characteristic_flux<W>(batch->geometry,face,batch->state,batch->flux[face])) {
            batch->save_speed(face);
        } else Kokkos::atomic_fetch_min(bad,face);
    }
};
// Размерность и вид шаблона выбираются до запуска. Укороченные стороны и
// маски восстановления читаются по конкретной грани из той же геометрии.
template<unsigned W,bool linear>
struct StencilSplitFaces {
    const FaceBatch* faces;
    const MeshIndex* indices;
    Index* bad;
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
    const FaceBatch* faces;
    const MeshIndex* indices;
    MeshIndex first;
    Index* bad;
    KOKKOS_INLINE_FUNCTION void operator()(MeshIndex k) const {
        const Index f=FaceBatch::face(indices,first+k);
        if (faces->is_boundary(f) && !faces->compute_boundary(f)) Kokkos::atomic_fetch_min(bad,f);
    }
};
// Вычисляет численные потоки выбранного набора площадок на GPU.
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
        const auto variables = batch.geometry.options.variables;
        if (residual_detail::uses_split_flux(batch.geometry)) {
            if (!device_faces) {
                for_indices(exec,"Aero: split flux",batch.count,KOKKOS_LAMBDA(Index k) {
                    const Index face=batch.face(k);
                    if (!residual_detail::split_face(batch,face)) Kokkos::atomic_fetch_min(bad,face);
                });
            } else if (!batch.geometry.stencil.weno_order) {
                // На цельной сетке диапазоны не пересекаются. На разделе MPI
                // фильтруем выданный список по типу площадки внутри каждого ядра.
                const Index edges=batch.geometry.edges;
                const MeshIndex* indices=batch.indices;
                const Index inner=indices ? batch.count : std::min(batch.count,edges);
                const Index first=indices ? 0 : inner;
                for_indices(exec,"Aero: EBR5 boundary flux",batch.count-first,
                            SplitBoundaryFaces{device_faces,indices,first,bad});
                if (batch.geometry.dim==2) for_indices(exec,"Aero: EBR5 split flux",inner,StencilSplitFaces<4,true>{device_faces,batch.indices,bad});
                else if (batch.geometry.dim==3) for_indices(exec,"Aero: EBR5 split flux",inner,StencilSplitFaces<5,true>{device_faces,batch.indices,bad});
                else for_indices(exec,"Aero: EBR5 split flux",inner,StencilSplitFaces<3,true>{device_faces,batch.indices,bad});
            } else if (batch.geometry.stencil.shared_edge_slopes && batch.geometry.stencil.central_from_edges &&
                       !batch.geometry.options.mp6_hybrid &&
                       (batch.geometry.stencil.mp5.weno || batch.geometry.options.solver!=ResidualSolver::Roe)) {
                if (batch.geometry.dim==2) for_indices(exec,"Aero: full MP split flux",batch.count,StencilSplitFaces<4,false>{device_faces,batch.indices,bad});
                else if (batch.geometry.dim==3) for_indices(exec,"Aero: full MP split flux",batch.count,StencilSplitFaces<5,false>{device_faces,batch.indices,bad});
                else for_indices(exec,"Aero: full MP split flux",batch.count,StencilSplitFaces<3,false>{device_faces,batch.indices,bad});
            } else if (batch.geometry.dim==2)
                for_indices(exec,"Aero: 2D split flux",batch.count,SplitFluxFaces<4>{device_faces,batch.indices,bad});
            else if (batch.geometry.dim==3)
                for_indices(exec,"Aero: 3D split flux",batch.count,SplitFluxFaces<5>{device_faces,batch.indices,bad});
            else
                for_indices(exec,"Aero: 1D split flux",batch.count,SplitFluxFaces<3>{device_faces,batch.indices,bad});
        } else if (packed.tiles && variables == ResidualVariables::Characteristic)
            face_kernels::compute_packed_fluxes(exec, batch, bad, device_faces);
        else if (packed.tiles && device_faces) {
            if (variables == ResidualVariables::Flux) face_kernels::cartesian_fluxes<true>(exec,batch,bad,device_faces);
            else face_kernels::cartesian_fluxes<false>(exec,batch,bad,device_faces);
        } else
            face_kernels::compute_unpacked_fluxes(exec, batch, bad);
    });
}
// Собирает узловую невязку из ориентированных потоков площадок на GPU.
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
// Собирает ограничения скорости изменения по опорам реконструкции на GPU.
const char* aero_device_stencil_rates(const StencilRates& work, Index count) {
    return on_device(nullptr, [&](const Execution& exec) {
        Kokkos::parallel_for("Aero: stencil residual maxima", Range(exec,0,count), work);
    });
}
// Вычисляет локальные стационарные шаги и их диапазон на GPU.
const char* aero_device_local_steps(const LocalSteadySteps& steps, StepRange* range) {
    return on_device(nullptr, [&](const Execution& exec) {
        Kokkos::parallel_reduce("Aero: local steady steps", Range(exec, 0, steps.gas.count), steps, *range);
    });
}
// Применяет верхние ограничения локальных шагов на GPU.
const char* aero_device_limit_local_steps(const LimitLocalSteps& limit, Index count) {
    return on_device(nullptr, [&](const Execution& exec) {
        Kokkos::parallel_for("Aero: limit local steady steps", Range(exec, 0, count), limit);
    });
}
// Выполняет проход ограничения отношений шагов соседних узлов на GPU.
const char* aero_device_neighbour_steps(const NeighbourStepSweep& sweep, Index count, NeighbourStepRange* result) {
    return on_device(nullptr, [&](const Execution& exec) {
        Kokkos::parallel_reduce("Aero: neighbouring local steps", Range(exec, 0, count), sweep, *result);
    });
}
// Выбирает шаг CFL по спектральным скоростям потоков на GPU.
const char* aero_device_flux_step(const FluxCfl& input, double courant, CflStep* answer, void* stream) {
    const FluxCfl grid = input;
    return on_device(stream, [&](const Execution& exec) {
        minimum_step(exec, grid.count, KOKKOS_LAMBDA(Index k) { return grid.local(k, courant); }, *answer);
    });
}
// Выбирает шаг CFL по узловым скоростям волн на GPU.
const char* aero_device_wave_step(cfl_detail::Grid grid, WaveSpeeds speeds, double courant, CflStep* answer,
                                  void* stream) {
    return on_device(stream, [&](const Execution& exec) {
        minimum_step(exec, grid.nodes, KOKKOS_LAMBDA(Index k) {
            return cfl_detail::local(grid, speeds, courant, k);
        }, *answer);
    });
}
// Обновляет узловое поле одной явной стадией на GPU.
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
// Обновляет собственные узлы подобласти одной явной стадией на GPU.
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
// Упаковывает или распаковывает выбранные узловые записи на GPU.
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
