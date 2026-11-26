// Kokkos launches the same numerical operators on CUDA and HIP.
#define AERO_BUILD_DEVICE_WORK
#include <Kokkos_Core.hpp>
#include "device_work.h"
#include "../implicit/block_jacobi_detail.h"
#include "../implicit/csgs_sweep.h"
#include "../implicit/matrix_execution.h"
#include "../implicit/bicgstab_vectors.h"
#include "../implicit/newton_vectors.h"
#include "../implicit/time_prediction.h"
#include "../implicit/prediction_execution.h"
#include "../io/steady_report.h"
#ifdef AERO_HAS_MULTIGRID
#include "../multigrid/transfer.h"
#endif
#include "device_execution.h"
#include "../aero/partition_execution.h"
#include "../explicit/stage_execution.h"
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/step_execution.h"
#endif
#include "../residual/face_execution.h"
#include "../residual/step_execution.h"
#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/jacobian.h"
#include "../viscosity/row_execution.h"
#endif

namespace aero {
namespace device_work {
namespace {
using namespace residual_detail;
using Range = DeviceRange<>;
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
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::factor(exec,input,diagonal,row_starts,first_bad); },device);
}
// Решает независимые диагональные системы на GPU.
const char* aero_device_jacobi_apply(int device, const implicit::JacobiBlocks& input, const double* residual,
                                     double* correction) {
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::solve(exec,input,residual,correction); },device);
}
// Упаковывает коэффициенты матрицы в порядок проходов CSGS.
const char* aero_device_pack_csgs(int device, const implicit::CsgsPacking& packing, const double* matrix) {
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::pack(exec,packing,matrix); },device);
}
// Вычисляет матричный образ для прохода CSGS на GPU.
const char* aero_device_csgs_image(int device,const implicit::CsgsSweep& input,MeshIndex count,double* image) {
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::image(exec,input,count,image); },device);
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
// Передаёт вязкие столбцы общему сборщику в потоке активного устройства.
const char* aero_device_viscous_matrix(int device, const viscosity::Jacobian& matrix, const double* state) {
    return on_device(nullptr, [&](const Execution& exec) {
        viscosity::assemble_jacobian(matrix,state,viscosity::execution::MatrixAssembly<Execution>{exec});
    }, device);
}
#endif
// Вычисляет производные потоков площадок на GPU.
const char* aero_device_spatial_faces(int device, const implicit::SpatialBlocks& matrix, const double* u,
                                      const double* initial, MeshIndex* bad) {
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::differentiate(exec,matrix,u,initial,bad); },device);
}
// Собирает строки пространственного якобиана из производных площадок.
const char* aero_device_spatial_rows(int device, const implicit::SpatialBlocks& matrix,
                                     const double* volumes, double dt, MeshIndex* bad) {
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::assemble(exec,matrix,volumes,dt,bad); },device);
}
// Обновляет временные диагональные коэффициенты системы на GPU.
const char* aero_device_time_diagonal(int device, const implicit::SpatialBlocks& matrix,
                                      const double* volumes, double dt, MeshIndex* bad) {
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::time_diagonal(exec,matrix,volumes,dt,bad); },device);
}
// Умножает распределённую пространственную матрицу на вектор на GPU.
const char* aero_device_spatial_apply(int device, const implicit::SpatialBlocks& matrix,
                                      const double* x, double* y) {
    return on_device(nullptr,[&](const Execution& exec) { implicit::matrix_rows::multiply(exec,matrix,x,y); },device);
}
#ifdef AERO_HAS_MULTIGRID
// Выполняет ограничение или продление многосеточного вектора на GPU.
const char* aero_device_multigrid_transfer(int device, MeshIndex count, const multigrid::Transfer& transfer) {
    return run_rows(device, "Aero: steady multigrid transfer", count, transfer);
}
// Обновляет рабочие векторы грубосеточной коррекции на GPU.
const char* aero_device_multigrid_update(int device, MeshIndex count, const multigrid::VectorUpdate& update) {
    return run_rows(device, "Aero: steady multigrid coarse vectors", count, update);
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
    return run_rows(device, "Aero: BDF Newton", nodes, update);
}
// Строит временное приближение по принятым слоям собственных узлов.
const char* aero_device_time_prediction(int device, MeshIndex rows, const implicit::TimePrediction& prediction) {
    return run_rows(device, "Aero: time prediction", rows, prediction);
}
// Запускает локальную подстановку на выбранной карте с постоянным описанием граней.
const char* aero_device_local_gas_prediction(int device, MeshIndex rows,const implicit::NewtonUpdate& gas,
                                             EulerFluxResult* flux,const FaceBatch* stored,const double* initial,double threshold) {
    return on_device(nullptr,[&](const Execution& exec) {
        implicit::predict_local_gas(exec,rows,gas,flux,stored,initial,threshold);
    },device);
}
// Каждая полоса записывает сводку в свою часть свободной правой части газа.
const char* aero_device_steady_report(const SteadyReport& report) {
    return run_rows(-1, "Aero: steady report", report.strips, report);
}
// Обновляет рабочие векторы BiCGSTAB на GPU.
const char* aero_device_krylov_update(int device, MeshIndex count, const implicit::KrylovUpdate& update) {
    return run_rows(device, "Aero: BiCGSTAB update", count, update);
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
    return run_rows(-1, "Aero: pressure time derivative", count, rate);
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
// Вызывает общий оператор конвекции с адресами активной карты.
const char* aero_device_sa_convection(const turbulence::SaEdgeConvection& edges,MeshIndex count,
                                      const void* tiles,unsigned order) {
    return on_device(nullptr,[&](const Execution& exec){ turbulence::execution::convection(exec,edges,count,tiles,order); });
}
// Ставит общий шаг SA в очередь устройства.
const char* aero_device_sa_step(const turbulence::Step& step,turbulence::Operation op) {
    return on_device(nullptr,[&](const Execution& exec){ turbulence::execution::apply_step(exec,step,op); });
}
// Возвращает на хост скаляр общей редукции SA.
const char* aero_device_sa_norm(const turbulence::Step& step,turbulence::Norm mode,bool refresh,double* result) {
    return on_device(nullptr,[&](const Execution& exec){ *result=turbulence::execution::measure_norm(exec,step,mode,refresh); });
}
#endif
// Подготавливает объёмные множители тетраэдрального разбиения на GPU.
const char* aero_device_prepare_tetra_volumes(const viscosity::Operator& input) {
    return on_device(nullptr,[&](const Execution& exec){coordinate_aes::prepare_volumes(exec,input.tetra);});
}
// Запускает требуемые фазы вязкого оператора AES на GPU.
const char* aero_device_viscosity(const viscosity::Operator& input, const double* state, const EulerState* cached,viscosity::Evaluation phase) {
    return on_device(nullptr,[&](const Execution& exec) {
        viscosity::execution::evaluate(exec,input,state,cached,phase,input.needs_gas(phase));
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
// Запускает общий пакет потоков в потоке устройства вызывающей стороны.
const char* aero_device_faces(const FaceBatch& batch, Index* bad, void* stream, const FaceBatch* stored) {
    return on_device(stream, [&](const Execution& exec) { compute_flux_batch(exec,batch,bad,stored); });
}
// Собирает узловые невязки в том же потоке, что и потоки площадок.
const char* aero_device_gather(const Geometry& g, const EulerFluxResult* flux, double* residual,
                               Index* bad, void* stream) {
    return on_device(stream, [&](const Execution& exec) { gather_flux_batch(exec,g,flux,residual,bad); });
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
        residual_detail::collect_local_steps(Range(exec, 0, steps.gas.count), steps, *range);
    });
}
// Применяет верхние ограничения локальных шагов на GPU.
const char* aero_device_limit_local_steps(const LimitLocalSteps& limit, Index count) {
    return on_device(nullptr, [&](const Execution& exec) {
        residual_detail::cap_local_steps(Range(exec, 0, count), limit);
    });
}
// Выполняет проход ограничения отношений шагов соседних узлов на GPU.
const char* aero_device_neighbour_steps(const NeighbourStepSweep& sweep, Index count, NeighbourStepRange* result) {
    return on_device(nullptr, [&](const Execution& exec) {
        residual_detail::sweep_neighbour_steps(Range(exec, 0, count), sweep, *result);
    });
}
// Выбирает шаг CFL по спектральным скоростям потоков на GPU.
const char* aero_device_flux_step(const FluxCfl& input, double courant, CflStep* answer, void* stream) {
    const FluxCfl grid = input;
    return on_device(stream, [&](const Execution& exec) {
        cfl_detail::minimum_step(exec, grid.count, KOKKOS_LAMBDA(Index k) { return grid.local(k, courant); }, *answer);
    });
}
// Выбирает шаг CFL по узловым скоростям волн на GPU.
const char* aero_device_wave_step(cfl_detail::Grid grid, WaveSpeeds speeds, double courant, CflStep* answer,
                                  void* stream) {
    return on_device(stream, [&](const Execution& exec) {
        cfl_detail::minimum_step(exec, grid.nodes, KOKKOS_LAMBDA(Index k) {
            return cfl_detail::local(grid, speeds, courant, k);
        }, *answer);
    });
}
// Обновляет узловое поле одной явной стадией на GPU.
const char* aero_device_stage(explicit_detail::FieldStageUpdate update, size_t count, int* bad, void* stream) {
    return on_device(stream, [&](const Execution& exec) {
        explicit_detail::update_field(exec,update,count,bad);
    });
}
// На итерации Ньютона новое состояние формирует другое ядро. Проверяем его после обмена.
const char* aero_device_failed_nodes(const partition_detail::Grid& input) {
    const auto g = input;
    auto* cache = partition_detail::validation_cache(g.geometry, !g.owned, g.owned_count);
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, "Aero: validate Newton state", g.owned_count, KOKKOS_LAMBDA(Index k) {
            const Index node = g.owned ? g.owned[k] : k;
            if (!partition_detail::admissible_node(g.geometry, g.state, node, cache)) partition_detail::record_failed_node(g,node);
        });
    });
}
// Обновляет собственные узлы подобласти одной явной стадией на GPU.
const char* aero_device_owned_stage(const partition_detail::Grid& input, explicit_detail::Stage stage,
                                    double dt, bool save) {
    const auto g = input;
    return on_device(nullptr, [&](const Execution& exec) {
        const viscosity::Operator* material = nullptr;
#ifdef AERO_HAS_TURBULENCE
        material = g.filtered_gas;
#endif
        partition_detail::advance_rows<true>(exec,g,stage,dt,save,material,nullptr);
    });
}
// Упаковывает или распаковывает выбранные узловые записи на GPU.
const char* aero_device_records(const partition_detail::RecordField& data, const MeshIndex* ids, Index count,
                                const double* input, double* output) {
    return on_device(nullptr,[&](const Execution& exec) {
        partition_detail::transfer_records(exec,data,ids,count,input,output);
    });
}
}
}
