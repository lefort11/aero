#include "solver.h"
#include "case_geometry.h"
#include "partition_state.h"
#include "mpi_balance.h"
#include "partition_operator.h"
#ifdef AERO_WITH_CUDA
#include "partition_gpu.h"
#endif
#include "time_loop.h"
#include "partition_output.h"
#include "recovery.h"
#include "mpi_periodic.h"
#include "mpi_stencil.h"
#include "partition_report.h"
#include "startup.h"
#include "../explicit/stage_sequence.h"
#include "../io/flowreport.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>

namespace aero {
namespace {
using Id = MeshIndex;

// Собирает локальные измерения и печатает один отчёт за шаг на процессе 0.
void report_step(const Processes& world, const FlowSummary& local, unsigned dimension, std::uint64_t step,
                 double time, double cfl, double dt) {
    std::string message;
    if (world.size == 1) {
        write_flow_summary(local, step, time, cfl, dt, dimension, std::cout);
        return;
    }
    world.local([&] {
        if (!local.volume) {
            return;
        }
        // Представление long double в библиотеке MPI может отличаться от принятого компилятором.
        std::ostringstream s;
        s.imbue(std::locale::classic());
        s << std::setprecision(std::numeric_limits<long double>::max_digits10);
        s << local.volume << ' ' << local.rate_scale << ' ' << local.scaled_squares;
        for (unsigned k = 0; k < dimension + 3; ++k) {
            s << ' ' << local.field[k].low << ' ' << local.field[k].high << ' ' << local.field[k].integral;
        }
        message = s.str();
    });
    const auto incoming = world.gather_text(std::move(message));
    world.local([&] {
        if (world.rank) {
            return;
        }
        FlowSummary total;
        for (const auto& record : incoming) {
            if (record.empty()) {
                continue;
            }
            std::istringstream s(record);
            s.imbue(std::locale::classic());
            FlowSummary part;
            s >> part.volume >> part.rate_scale >> part.scaled_squares;
            for (unsigned k = 0; k < dimension + 3; ++k) {
                s >> part.field[k].low >> part.field[k].high >> part.field[k].integral;
            }
            if (!s) {
                throw std::runtime_error("Invalid MPI flow report");
            }
            total.merge(part, dimension + 3);
        }
        write_flow_summary(total, step, time, cfl, dt, dimension, std::cout);
    });
}

// Выполняет шаг на своей подобласти. Буфер начальных значений стадий живёт весь расчёт.
class PartitionStepper {
public:
    // Оператор и поля принадлежат расчёту и используются всеми стадиями.
    PartitionStepper(PartitionOperator& op, PartitionState& fields, const Processes& world,
                     explicit_detail::Stages stages)
        : op_(op), fields_(fields), world_(world), stages_(stages) {}

    // Для отчёта каждый процесс учитывает только объёмы узлов, которыми владеет.
    void prepare_report(const RunSettings& run, unsigned dimension, const double* volumes) {
        if (!run.flowreport_every || op_.owned.empty()) {
            return;
        }
        std::vector<double> owned_volumes;
        owned_volumes.reserve(op_.owned.size());
        for (Id node : op_.owned) {
            owned_volumes.push_back(volumes[node]);
        }
        report_.reset(new FlowReport(run.flowreport_every, run.gamma, dimension, owned_volumes.size(),
                                     owned_volumes.data()));
    }

    // На GPU многостадийная схема хранит промежуточные поля на карте. На CPU нужен рабочий буфер.
    void prepare_storage(bool use_cuda, unsigned width) {
        if ((!use_cuda && stages_.count > 1) || report_) {
            owned_state_.resize(op_.owned.size() * width);
        }
    }

    // Первые потоки нужны и для CFL, и для обновления решения на первой стадии.
    double prepare_first_stage(double cfl, SolverTimings& timing) {
        assemble_fluxes(timing);
        WallTimer timer(timing.counter(SolverTimings::Cfl));
        return op_.cfl_step(cfl);
    }

    // Выполняет стадии выбранной схемы. После каждой стадии соседи получают обновлённые узлы.
    void advance(const TimeStep& step, SolverTimings& timing) {
        const auto assemble = [&](double) { assemble_fluxes(timing); };
        const auto update = [&](explicit_detail::StepStage stage) {
            world_.local([&] { update_stage(stage, step, timing); });
        };
        const auto exchange = [&] {
            if (world_.size > 1) {
                op_.halo(fields_.current, timing.counter(SolverTimings::Halo));
            }
        };
        explicit_detail::advance_stages(stages_, step.start, step.dt,
                                        explicit_detail::FirstResidual::Prepared, assemble, update, exchange);
    }

    // Даже процесс без собственных узлов участвует в сборе общей сводки.
    void write_report(const TimeStep& step, unsigned dimension, double cfl) const {
        report_step(world_, summary_, dimension, step.number, step.end, cfl, step.dt);
    }

private:
    // Пересчитывает потоки по текущим полям и согласует их на границах подобластей.
    void assemble_fluxes(SolverTimings& timing) {
        op_.assemble_fluxes(fields_.current, fields_.initial, timing);
    }

    // Снимки до первой и после последней стадии дают изменение полей за целый шаг.
    void update_stage(explicit_detail::StepStage stage, const TimeStep& step, SolverTimings& timing) {
        const bool measure = step.reporting && report_;
        if (stage.first() && measure) {
            op_.copy_owned(fields_.current, owned_state_);
            report_->capture(step.number, owned_state_);
        }
        {
            WallTimer timer(timing.counter(SolverTimings::Update));
            // При обновлении на месте старое U_n понадобится во второй и третьей стадиях SSPRK3.
            op_.update_owned(stage.formula, owned_state_, step.dt, fields_.current,
                             stage.save_start_for_later());
        }
        if (stage.last() && measure) {
            op_.copy_owned(fields_.current, owned_state_);
            summary_ = report_->measure(step.number, owned_state_);
        }
    }

    PartitionOperator& op_;              // Потоки, обновление узлов и обмен с соседями.
    PartitionState& fields_;             // Исходное и текущее состояния подобласти.
    const Processes& world_;             // Процессы, участвующие в общем шаге.
    explicit_detail::Stages stages_;     // Коэффициенты и число стадий схемы по времени.
    std::unique_ptr<FlowReport> report_; // Измерения на узлах этого процесса.
    std::vector<double> owned_state_;    // Буфер для стадий и снимков собственных узлов.
    FlowSummary summary_;                // Локальное измерение последнего отчётного шага.
};

// Обычная сетка образует одну подобласть с исходными номерами узлов и границ.
MeshPartition whole_mesh(const std::string& path) {
    MeshPartition part;
    part.mesh = Mesh::read(path);
    const auto m = part.mesh.view();
    part.nodes.resize(m.node_count);
    std::iota(part.nodes.begin(), part.nodes.end(), Id(0));
    part.node_owners.resize(m.node_count, 0);
    part.boundary_order.resize(m.boundary_face_count);
    std::iota(part.boundary_order.begin(), part.boundary_order.end(), Id(0));
    part.global_counts = {{m.node_count, m.face_count, m.cell_count}};
    part.physical_patches = part.mesh.data().patch_names.size();
    return part;
}

// Строит поверхности и шаблоны реконструкции. Временные данные архива освобождаются при выходе.
std::unique_ptr<PartitionOperator>
prepare_partition_operator(const RunSettings& run, const ReconstructionOptions& reconstruction,
                           const Processes& world, const MeshPartition& part, const CaseGeometry& geometry,
                           const ResidualFrame& frame, std::unique_ptr<PartitionInput> input,
                           bool needs_stencil) {
    const bool single = world.size == 1;
    const auto* seam = geometry.periodic_partition();
    const auto& boundaries = geometry.boundaries;
    if (single) {
        stage("Building reconstruction stencils");
    }

    StencilRequests requests;
    std::unique_ptr<PartitionOperator> op;
    world.local([&] {
        auto physics = residual_options(run.reconstruction_variables, run.euler_solver, run.gamma,
                                              run.trace_ratio_min, run.trace_ratio_max);
        physics.upw_coeff = run.upw_coeff;
        physics.mp6_hybrid = run.mp6_hybrid;
        op.reset(new PartitionOperator(part, geometry.dual->view(), seam, boundaries.patches, physics, frame,
                                       world, needs_stencil ? &requests : nullptr, run.use_cuda));
    });
    Id downgraded = 0;
    if (needs_stencil) {
        downgraded =
            op->prepare_stencil(part, seam, boundaries.periodic_pairs, input.get(), requests, reconstruction);
    }
    if (single) {
        done();
        const auto edges = seam ? seam->edges.size() : part.mesh.view().edge_count;
        std::cout << "Computational nodes = " << op->state_nodes() << "  edges = " << edges << "  ";
    }
    if (!world.rank) {
        std::cout << "downgraded reconstruction sides = " << downgraded << '\n';
    }
    return op;
}

#ifdef AERO_WITH_CUDA
// Сначала все карты пробуют зарезервировать память. При отказе хотя бы одной карты
// вызывающий код освободит пробные буферы и повторит разбиение сетки.
bool prepare_gpu_memory(PartitionOperator& op, const MeshPartition& part, const RunSettings& run,
                        const Processes& world, bool compact, GpuMemoryFit& fit) {
    const bool single = world.size == 1;
    world.local([&] {
        if (single) {
            fit.budget = std::numeric_limits<Id>::max();
        } else {
            query_gpu_memory(fit, run.mpi_gpu_memory_mib, world);
        }
        fit.first = part.first_leaf;
        fit.end = part.end_leaf;
        op.reserve_gpu(compact, fit);
    });
    if (single) {
        if (fit.failed) {
            throw std::runtime_error("Insufficient GPU memory for the mesh");
        }
    } else {
        const bool retry = world.sum(fit.failed ? 1 : 0) != 0;
        const auto lines = world.gather_text(
            "rank " + std::to_string(world.rank) + " leaves [" + std::to_string(fit.first) + "," +
            std::to_string(fit.end) + ") required_bytes " + std::to_string(fit.required) + " budget_bytes " +
            std::to_string(fit.budget) + (fit.failed ? " rejected" : " reserved"));
        if (!world.rank) {
            for (const auto& line : lines) {
                std::cout << "MPI GPU fit: " << line << '\n';
            }
        }
        if (retry) {
            return false;
        }
    }
    world.local([&] { op.layout_gpu(compact); });
    return true;
}
#endif

// Вывод идёт после успешной подготовки устройства: по этой строке QA проверяет выбранного исполнителя.
void report_active_backend(const RunSettings& run, const Processes& world) {
    if (world.size == 1) {
        std::cout << "Active backend = "
                  << (run.use_cuda ? "CUDA device " + std::to_string(run.devices.front()) : "CPU") << '\n';
    } else if (!world.rank) {
        std::cout << "Active backend = " << (run.use_cuda ? "CUDA" : "CPU")
                  << " (MPI), processes = " << world.size << '\n';
    }
}

// Общий цикл времени вызывает эти операции и в обычном расчёте, и при разбиении.
// Поля остаются на выбранном исполнителе. Копию с GPU загружаем на CPU, когда её запросил вывод.
void integrate_partition(const RunSettings& run, const Processes& world, unsigned dimension,
                         PartitionStepper& stepper, PartitionOperator& op, PartitionState& fields,
                         PartitionOutput& output, Recovery* recovery) {
    StepOperations operations;
    operations.cfl_step = [&](SolverTimings& timing) { return stepper.prepare_first_stage(run.cfl, timing); };
    operations.advance = [&](const TimeStep& step, SolverTimings& timing) { stepper.advance(step, timing); };
    operations.output_state = [&]() -> const std::vector<double>& {
        if (run.use_cuda) {
            world.local([&] { op.download(fields.current); });
        }
        return fields.current;
    };
    operations.report = [&](const TimeStep& step) { stepper.write_report(step, dimension, run.cfl); };
    operations.write_fields = [&](const TimeStep& step, const std::vector<double>& snapshot) {
        output.write(step.number, step.end, snapshot);
    };
    integrate_time(run, world, output.schedule, recovery, fields.initial, operations);
}

// Готовит одну подобласть и ведёт расчёт до конечного времени.
// Возвращает -1, если для MPI с CUDA нужно заново распределить сетку по картам.
int run_partition(RunSettings run, const ReconstructionOptions& reconstruction, const Processes& world,
                  const LeafLayout& layout, GpuMemoryFit* gpu, bool first_attempt, bool archive) {
#ifndef AERO_WITH_CUDA
    (void)gpu;
#endif
    const bool reconstruct = reconstruction.scheme != ReconstructionScheme::FirstOrder;
    std::unique_ptr<PartitionInput> input;
    const bool single = world.size == 1;
    // Для распределённой реконструкции сохраняем данные архива до сборки удалённых шаблонов.
    auto part = archive
                    ? load_partition(run.mesh_path, world, reconstruct && !single ? &input : nullptr, layout)
                    : whole_mesh(run.mesh_path);
    const auto mesh = part.mesh.view();
    const unsigned width = mesh.dimension + 2;
    CaseGeometry geometry;
    if (!prepare_case_geometry(run, part, world, first_attempt, geometry)) {
        return 2;
    }

    const auto cv = geometry.dual->view();
    const bool periodic = geometry.periodic();
    auto& seam = geometry.seam;
    const auto frame = momentum_frame(mesh, world, &part);
    const auto stages = explicit_detail::stages_for(explicit_detail::scheme_from_name(run.time_scheme));
    PartitionState fields;
    std::uint64_t stencil_peak = 0;
    auto op = prepare_partition_operator(run, reconstruction, world, part, geometry, frame, std::move(input),
                                         reconstruct);
    PartitionStepper stepper(*op, fields, world, stages);
    if (!single) {
        op->connect(run.use_cuda);
    }
#ifdef AERO_WITH_CUDA
    GpuMemoryFit single_fit;
    if (single && run.use_cuda) {
        gpu = &single_fit;
    }
    if (gpu) {
        const bool compact = stages.count > 1 || run.flowreport_every;
        if (!prepare_gpu_memory(*op, part, run, world, compact, *gpu)) {
            return -1;
        }
    }
#endif
    const PartitionFileLayout file_layout(run, part, geometry.periodic_partition(), world);
    // Контрольная точка и общий файл полей используют одну таблицу исходных номеров.
    std::unique_ptr<Recovery> recovery;
    if (run.recovery_every_steps || run.directories.recover) {
        const auto hash = archive ? part.mesh_sha256 : FieldSnapshot::mesh_fingerprint(part.mesh);
        recovery.reset(
            new Recovery(run, world, part.global_counts[0], mesh.dimension, hash, file_layout.rows));
    }
    world.local([&] {
        if (!single) {
            stencil_peak = process_peak_kib();
        }
        stepper.prepare_report(run, mesh.dimension, periodic ? seam.volumes.data() : cv.volumes);
        geometry.release_stencil_geometry();
        stepper.prepare_storage(run.use_cuda, width);
        fields.prepare(run, part, geometry.periodic_partition(), file_layout.selection, op->state_nodes(),
                       world);
    });

    // Контрольная точка хранит только собственные узлы. Соседние значения восстанавливаем обменом.
    if (run.directories.recover) {
        recovery->restore(fields.current, fields.initial);
        if (!single) {
            op->halo_host(fields.current);
        }
    }
    if (!single) {
        op->halo_host(fields.initial);
    }

    const auto& base = recovery ? recovery->fields_directory() : run.directories.output;
    std::unique_ptr<PartitionOutput> output;
    world.local([&] {
        if (!run.directories.recover) {
            fields.start_from_initial(part, geometry.periodic_partition());
        }
#ifdef AERO_WITH_CUDA
        if (gpu) {
            op->upload_state(fields.current, fields.initial);
            // Замер цикла начинается после завершения начальной загрузки на карту.
            if (run.mpi_timing)
                PartitionGpu::synchronize();
        }
#endif
        output.reset(
            new PartitionOutput(run, part, file_layout, world, base, seam.node_dof, op->state_nodes()));
    });
    if (!single) {
        write_partition_report(world, op->work(part.end_leaf - part.first_leaf), stencil_peak, std::cout);
    }
    report_active_backend(run, world);
    integrate_partition(run, world, mesh.dimension, stepper, *op, fields, *output, recovery.get());
    return 0;
}
}

int calculate_case(RunSettings run, const ReconstructionOptions& reconstruction, const Processes& world,
                   bool archive) {
#ifdef AERO_WITH_CUDA
    if (world.size > 1 && run.use_cuda) {
        select_partition_gpu(world, run.devices);
        GpuPlacement gpu;
        int automatic = run.mpi_rank_weights.empty() ? 1 : 0;
        world.broadcast(&automatic, 1, 0);
        if (automatic) {
            // Для калибровки берём один и тот же газ, чтобы выбор разбиения не зависел от начального поля.
            auto physics = residual_options(run.reconstruction_variables, run.euler_solver, 1.4);
            world.local([&] { query_gpu_memory(gpu.local, run.mpi_gpu_memory_mib, world); });
            run.mpi_rank_weights = PartitionGpu::measure_weights(world, physics, gpu.local.budget);
        }
        for (bool first = true;; first = false) {
            world.local([&] { query_gpu_memory(gpu.local, run.mpi_gpu_memory_mib, world); });
            const auto plan = plan_partition(run, world, &gpu);
            const int result = run_partition(run, reconstruction, world, plan, &gpu.local, first, archive);
            if (result >= 0) {
                return result;
            }
        }
    }
#endif
    return run_partition(run, reconstruction, world, plan_partition(run, world), nullptr, true, archive);
}
}
