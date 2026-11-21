#include "solver.h"
#include "run_control.h"
#include "exact_error.h"
#include "case_geometry.h"
#include "partition_state.h"
#include "mpi_balance.h"
#include "partition_operator.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "partition_gpu.h"
#include "../parallel/device_check.h"
#endif
#include "time_loop.h"
#include "partition_output.h"
#include "recovery.h"
#include "mpi_periodic.h"
#include "mpi_stencil.h"
#include "partition_report.h"
#include "startup.h"
#include "../explicit/stage_sequence.h"
#include "../io/report.h"
#include "../io/settings_text.h"
#include <iomanip>
#include <cctype>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>
#include <set>

namespace aero {
namespace {
using Id = MeshIndex;

// Диагностика относится ко всей постановке. На MPI каждое сообщение печатает только ведущий процесс.
void print_input_warnings(const RunSettings& run, const Processes& world, std::ostream& out) {
    if (world.rank) return;
    for (const auto& warning : run.input_warnings) out << "WARNING: " << warning << '\n';
}

// Собирает локальные измерения и печатает один отчёт за шаг на процессе 0.
void report_step(const Processes& world, const FlowSummary& local, unsigned dimension, std::uint64_t step,
                 double time, double cfl, double dt, const ReportDetails& details) {
    std::string message;
    if (world.size == 1) {
        write_flow_summary(local, step, time, cfl, dt, dimension, std::cout, details);
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
                throw std::runtime_error("Invalid MPI report");
            }
            total.merge(part, dimension + 3);
        }
        write_flow_summary(total, step, time, cfl, dt, dimension, std::cout, details);
    });
}

// Выполняет шаг на своей подобласти. Буфер начальных значений стадий живёт весь расчёт.
struct RejectedGasState {};

class PartitionStepper {
public:
    // Оператор и поля принадлежат расчёту и используются всеми стадиями.
    PartitionStepper(PartitionOperator& op, PartitionState& fields, const Processes& world,
                     RunSettings& run)
        : op_(op), fields_(fields), world_(world), run_(run), courant_(run.stabilization, run.cfl),
          stages_(run.uses_implicit_time() ? explicit_detail::Stages{} :
                  explicit_detail::stages_for(explicit_detail::scheme_from_name(run.time_scheme))) {
        if (!run.background.active() && (run.t_fo > 0. || run.reconstruction == "first_order"))
            total_cells_ = world.sum(op.owned.size());
    }

    bool needs_compact_buffer() const { return stages_.count > 1 || run_.report_every; }

    // Для отчёта каждый процесс учитывает только объёмы узлов, которыми владеет.
    void prepare_report(const RunSettings& run, unsigned dimension, const double* volumes) {
        if (!run.report_every || op_.owned.empty()) {
            return;
        }
        std::vector<double> owned_volumes;
        owned_volumes.reserve(op_.owned.size());
        for (Id node : op_.owned) {
            owned_volumes.push_back(volumes[node]);
        }
        report_.reset(new Report(run.report_every, run.gamma, dimension, owned_volumes.size(),
                                     owned_volumes.data(), run.background));
    }

    // На GPU многостадийная схема хранит промежуточные поля на карте. На CPU нужен рабочий буфер.
    void prepare_storage(bool use_cuda, unsigned width) {
        if ((!use_cuda && stages_.count > 1) || report_) {
            owned_state_.resize(op_.owned.size() * width);
        }
    }

    // Общий буфер потоков используют CFL, первая стадия интегратора и вывод dp/dt.
    // Проверяем также порядок реконструкции: на границе t_fo он меняется при том же поле.
    void prepare_state_fluxes(bool first_order_start, SolverTimings& timing) {
        if (op_.first_order_start != first_order_start) {
            op_.first_order_start = first_order_start;
            // У BDF потоки прошлого шага могли сохраниться для следующего CFL.
            // После смены порядка их надо вычислить заново на том же поле.
            accepted_fluxes_ = false;
            if (!first_order_start && !world_.rank)
                std::cout << "Reconstruction = " << run_.reconstruction << " from t_fo = " << run_.t_fo << '\n';
        }
        if (!accepted_fluxes_) {
            assemble_fluxes(timing);
            accepted_fluxes_ = true;
        }
    }

    double prepare_first_stage(bool first_order_start, SolverTimings& timing) {
        prepare_state_fluxes(first_order_start, timing);
        WallTimer timer(timing.counter(SolverTimings::Cfl));
        return op_.cfl_step(courant());
    }

    // Выполняет стадии выбранной схемы. После каждой стадии соседи получают обновлённые узлы.
    double courant() const { return courant_.value(); }

    // Снимаем локальные метки также на обменных узлах MPI и в таблице граней на GPU.
    void reset_marks() {
        if (!have_marks_) return;
        std::fill(marks_.begin(), marks_.end(), 0.);
        op_.set_low_order(marks_);
        marks_.clear();
        master_low_.clear();
        have_marks_ = false;
        accepted_fluxes_ = false;
    }

    // После принятия шага все ранги уже прочитали один набор команд. Меняем
    // только правило выбора dt: поле и накопленная пространственная матрица остаются пригодными.
    void set_steady_steps(bool local, double max_ratio) {
        run_.local_time_step = local;
        run_.local_time_step_max_ratio = max_ratio;
        op_.set_steady_steps(local, max_ratio);
        runtime_steady_steps_ = true;
    }

    bool control(Id step) {
        RunControl command(run_, courant());
        if (!command.read(run_.directories.output, world_)) return false;
        if (command.change_steady_steps) set_steady_steps(command.local_time_step, command.local_time_step_max_ratio);
        if (command.reset) reset_marks();
        if (command.change_cfl) {
            run_.stabilization = command.limits;
            courant_.configure(command.limits, command.cfl);
            op_.set_courant(courant(), run_.uses_implicit_time());
            runtime_settings_ = true;
            accepted_fluxes_ = false;
        }
        if (run_.upw_coeff != command.upwind || run_.mp6_hybrid != command.hybrid) {
            run_.upw_coeff = command.upwind;
            run_.mp6_hybrid = command.hybrid;
            world_.local([&] { op_.set_flux_weights(command.upwind, command.hybrid); });
            runtime_settings_ = true;
            accepted_fluxes_ = false;
        }
        if (command.change_cfl && run_.stabilization.mode == "low" && courant() >= .99 * command.limits.maximum)
            reset_marks();
        if (!world_.rank) {
            command.report_accepted(std::cout);
            std::cout << "Control applied after step " << step << ", CFL = " << courant()
                      << ", upw_coeff = " << command.upwind << ", mp6_hybrid = " << command.hybrid
                      << ", cfl_min = " << command.limits.minimum << ", cfl_max = " << command.limits.maximum
                      << ", cfl_num_steps = " << command.limits.interval << ", cfl_coeff = " << command.limits.factor
                      << ", first-order cells = " << first_order_cells() << std::endl;
        }
        return command.stop;
    }

    void accepted() {
        if (!courant_.accept()) return;
        op_.set_courant(courant(), run_.uses_implicit_time());
        accepted_fluxes_ = false;
        if (run_.stabilization.mode == "low" && have_marks_ &&
            courant() >= .99 * run_.stabilization.maximum) {
            reset_marks();
            if (!world_.rank) std::cout << "Restored reconstruction, first-order cells = " << first_order_cells() << '\n';
        }
    }

    // Сохраняем все узлы, включая обменные. При отказе на поздней стадии
    // SSPRK3 или итерации Ньютона возвращаемся именно к началу шага.
    bool attempt(const TimeStep& step, SolverTimings& timing) {
        if (!run_.stabilization.active()) {
            advance(step, timing);
            return true;
        }
        world_.local([&] { op_.copy_backup(fields_.current, backup_, false); });
        try {
            advance(step, timing);
            return true;
        } catch (const RejectedGasState&) {
            world_.local([&] { op_.copy_backup(fields_.current, backup_, true); });
            if (run_.stabilization.local_order()) {
                op_.set_low_order(marks_);
                have_marks_ = true;
            }
            const double previous_cfl = courant();
            courant_.reject();
            op_.set_courant(courant(), run_.uses_implicit_time());
            accepted_fluxes_ = false;
            if (!world_.rank) {
                // Сообщаем сразу при каждом откате, даже если периодический отчёт выключен.
                std::ostringstream message;
                message << std::setprecision(17) << "RETRY step " << step.number
                        << " | CFL " << previous_cfl << " -> " << courant() << " | failed nodes = " << failed_count_;
                std::cout << message.str() << std::endl;
            }
            return false;
        }
    }

    // Храним глобальные номера. После перезапуска разбиение MPI может измениться.
    std::string checkpoint() const {
        if (world_.rank || (!run_.stabilization.active() && !runtime_settings_ && !runtime_steady_steps_)) return {};
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(17) << "ready " << runtime_settings_ << ' ';
        if (runtime_settings_) {
            const auto& limits = run_.stabilization;
            out << run_.upw_coeff << ' ' << run_.mp6_hybrid << ' ' << limits.minimum << ' '
                << limits.maximum << ' ' << limits.factor << ' ' << limits.interval << ' ';
        }
        out << run_.stabilization.mode << ' ' << std::setprecision(17) << courant() << ' ' << courant_.successes();
        for (const auto* group : {&master_low_, &report_failed_}) {
            out << ' ' << group->size();
            for (Id node : *group) out << ' ' << node;
        }
        // Старые recovery заканчиваются списками узлов. Дописываем режим только
        // после ручной команды, чтобы в остальных случаях действовал текущий implicit.txt.
        if (runtime_steady_steps_) {
            out << " local_time_step " << (run_.local_time_step ? "on" : "off") << ' ';
            if (run_.local_time_step_max_ratio) out << run_.local_time_step_max_ratio;
            else out << "off";
        }
        return out.str();
    }

    void resume(const std::string& data) {
        if (data.empty()) return; // Старый снимок содержит только газодинамические поля.
        std::istringstream in(data);
        in.imbue(std::locale::classic());
        std::string mode;
        double value;
        Id count;
        in >> mode;
        const bool ready = mode == "ready";
        if (ready) {
            in >> runtime_settings_;
            if (runtime_settings_) {
                auto& limits = run_.stabilization;
                in >> run_.upw_coeff >> run_.mp6_hybrid >> limits.minimum >> limits.maximum >> limits.factor >> limits.interval;
                world_.local([&] { op_.set_flux_weights(run_.upw_coeff, run_.mp6_hybrid); });
            }
            in >> mode;
        }
        if (!(in >> value >> count) || mode != run_.stabilization.mode)
            throw std::runtime_error("Incompatible checkpoint stabilization mode");
        courant_.configure(run_.stabilization, value, count);
        op_.set_courant(value, run_.uses_implicit_time());
        for (auto* group : {&master_low_, &report_failed_}) {
            if (!(in >> count)) throw std::runtime_error("Invalid checkpoint cell count");
            for (Id i = 0, node; i < count; ++i) {
                if (!(in >> node)) throw std::runtime_error("Invalid checkpoint cell ID");
                group->insert(node);
            }
        }
        std::string key, local, ratio;
        if (in >> key) {
            if (key != "local_time_step" || !(in >> local >> ratio))
                throw std::runtime_error("Invalid checkpoint local time step settings");
            if (run_.steady()) set_steady_steps(settings_detail::parse_local_time_step(local, key),
                settings_detail::parse_local_step_ratio(ratio, "checkpoint local_time_step_max_ratio"));
        }
        have_marks_ = !master_low_.empty();
        if (have_marks_) {
            marks_.assign(fields_.current.size(), 0.);
            const Id width = fields_.current.size() / std::max<Id>(1, op_.state_nodes());
            for (Id node : op_.owned)
                if (master_low_.count(op_.global_node(node))) marks_[width*node] = 1.;
            op_.set_low_order(marks_);
        }
        if (world_.rank) { master_low_.clear(); report_failed_.clear(); }
        if (!ready) accepted(); // В прежнем формате повышение CFL выполнялось после записи.
    }

    void advance(const TimeStep& step, SolverTimings& timing) {
        if (!run_.uses_implicit_time()) {
            advance_explicit(step, timing);
            return;
        }
        if (step.reporting) world_.local([&] { capture_report(step); });
        try {
            newton_ = op_.advance_bdf(fields_.current, fields_.initial, step.dt,
                                       run_.implicit_settings, timing, run_.stabilization.active() ?
                                       std::function<void()>([&] { validate_stage(); }) : std::function<void()>());
            // Критерий Ньютона проверен по потокам принятого поля. До следующего CFL поле сохраняется.
            accepted_fluxes_ = true;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(run_.implicit_settings.second_order ? "BDF2 step " : "BDF1 step ") + std::to_string(step.number) + ": " + e.what());
        }
        if (step.reporting) world_.local([&] { finish_report(step); });
    }

    // Даже процесс без собственных узлов участвует в сборе общей сводки.
    void write_report(const TimeStep& step, unsigned dimension, double cfl, double steady_residual, double relative_residual) {
        print_input_warnings(run_, world_, std::cout);
        ReportDetails details;
        details.scheme = run_.time_scheme;
        for (char& c : details.scheme) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        details.linear = run_.background.active();
        if (run_.uses_implicit_time()) {
            details.iterations = newton_.iterations;
            details.max_iterations = run_.implicit_settings.max_iterations;
            details.newton_residual = newton_.relative_residual;
            details.newton_tol = run_.implicit_settings.newton_relative;
        }
        if (run_.steady()) {
            details.steady_criterion = run_.steady_absolute ? "absolute" : "relative";
            details.steady_residual = steady_residual;
            details.relative_steady_residual = relative_residual;
            details.steady_tol = run_.steady_tol;
            details.max_local_step = op_.largest_step(step.dt);
        }
        if (!details.linear) {
            // Ведущий процесс хранит уникальные номера отказавших узлов со всех разделов.
            details.stabilization = run_.stabilization.mode;
            details.failed = report_failed_.size();
            details.first_order = first_order_cells();
        }
        report_step(world_, summary_, dimension, step.number, step.end, cfl, step.dt, details);
        report_failed_.clear();
    }

private:
    // Сохраняем энергию перед первой стадией и измеряем её изменение после последней.
    // В MPI обе операции выполняются внутри уже существующей синхронизации стадии.
    void capture_report(const TimeStep& step) {
        if (!step.reporting || !report_) return;
        op_.copy_owned(fields_.current, owned_state_);
        report_->capture(step.number, owned_state_);
    }
    void finish_report(const TimeStep& step) {
        if (!step.reporting || !report_) return;
        op_.copy_owned(fields_.current, owned_state_);
        summary_ = report_->measure(step.number, owned_state_);
    }
    // Проверяем собственные контрольные объёмы. Одного отказа на любом
    // процессе достаточно для повтора шага на всей распределённой сетке.
    Id first_order_cells() const {
        return (op_.first_order_start || run_.reconstruction == "first_order") ? total_cells_ : master_low_.size();
    }
    void validate_stage() {
        std::vector<Id> failed;
        const unsigned width = static_cast<unsigned>(fields_.current.size() / std::max<Id>(1, op_.state_nodes()));
        world_.local([&] {
            op_.failed_nodes(fields_.current, failed, run_.uses_implicit_time());
            for (Id& node : failed) {
                if (run_.stabilization.local_order()) {
                    if (marks_.empty()) marks_.resize(fields_.current.size());
                    marks_[node * width] = 1.;
                }
                node = op_.global_node(node);
            }
        });
        failed_count_ = world_.size == 1 ? failed.size() : world_.sum(failed.size());
        if (failed_count_) {
            // Владельцы передают только глобальные номера отказавших узлов.
            // По ним master знает и число, и состав проблемных объёмов всей сетки.
            std::ostringstream ids;
            for (Id node : failed) ids << node << ' ';
            const auto records = world_.gather_text(ids.str());
            if (!world_.rank) {
                for (const auto& record : records) {
                    std::istringstream in(record);
                    Id node;
                    while (in >> node) {
                        if (run_.report_every) report_failed_.insert(node);
                        if (run_.stabilization.local_order()) master_low_.insert(node);
                    }
                }
            }
            if (run_.stabilization.local_order() && marks_.empty()) marks_.resize(fields_.current.size());
            throw RejectedGasState{};
        }
    }

    void advance_explicit(const TimeStep& step, SolverTimings& timing) {
        const auto assemble = [&](double) { assemble_fluxes(timing); };
        const auto update = [&](explicit_detail::StepStage stage) {
            world_.local([&] {
                if (stage.first()) capture_report(step);
                update_stage(stage, step, timing);
                if (!run_.stabilization.active() && stage.last()) finish_report(step);
            });
            if (run_.stabilization.active()) {
                validate_stage();
                if (stage.last() && step.reporting) world_.local([&] { finish_report(step); });
            }
        };
        const auto exchange = [&] {
            if (world_.size > 1) {
                op_.halo(fields_.current, timing.counter(SolverTimings::Halo));
            }
        };
        explicit_detail::advance_stages(stages_, step.start, step.dt,
                                        explicit_detail::FirstResidual::Prepared, assemble, update, exchange);
    }

    // Пересчитывает потоки по текущим полям и согласует их на границах подобластей.
    void assemble_fluxes(SolverTimings& timing) {
        op_.assemble_fluxes(fields_.current, fields_.initial, timing);
    }

    // Для SSPRK3 сохраняем U_n в начале шага и используем его на следующих стадиях.
    void update_stage(explicit_detail::StepStage stage, const TimeStep& step, SolverTimings& timing) {
        // Стадия использует потоки входного поля, затем меняет само поле.
        // Для выходного состояния следующая сборка понадобится один раз.
        accepted_fluxes_ = false;
        WallTimer timer(timing.counter(SolverTimings::Update));
        op_.update_owned(stage.formula, owned_state_, step.dt, fields_.current, stage.save_start_for_later());
    }

    PartitionOperator& op_;              // Потоки, обновление узлов и обмен с соседями.
    PartitionState& fields_;             // Исходное и текущее состояния подобласти.
    const Processes& world_;             // Процессы, участвующие в общем шаге.
    bool runtime_steady_steps_ = false; // Параметры псевдошага из control.txt сохраняются и при stabilization=off.
    bool runtime_settings_ = false; // Записываем ручные настройки вместе с полем для --recover.
    RunSettings& run_;             // Схема по времени, допуски и выбранная карта.
    std::set<Id> master_low_; // Глобальные номера узлов с пониженным порядком на процессе 0.
    std::set<Id> report_failed_; // Уникальные отказавшие узлы с предыдущего Report.
    Id failed_count_ = 0, total_cells_ = 0;
    bool have_marks_ = false; // Одинаков на всех процессах, включая пустые подобласти.
    AdaptiveCourant courant_;
    std::vector<double> backup_, marks_; // Принятый слой и постоянные до сброса метки узлов.
    bool accepted_fluxes_ = false; // После чтения начального поля потоки ещё предстоит вычислить.
    implicit::NewtonResult newton_{};    // Итог последнего принятого неявного шага.
    explicit_detail::Stages stages_;     // Коэффициенты и число стадий схемы по времени.
    std::unique_ptr<Report> report_; // Измерения на узлах этого процесса.
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
        physics.retry_nonphysical = run.stabilization.active();
        physics.background = run.background;
        op.reset(new PartitionOperator(part, geometry.dual->view(), seam, boundaries.patches, physics, frame,
                                       world, needs_stencil ? &requests : nullptr, run.use_cuda));
    });
    for (const auto& zone:run.output_zones) for (const auto& name:zone.variables) {
        if (diagnostic::needs_time(name)) op->enable_time_samples();
        if (diagnostic::column(name) == diagnostic::DpDt) op->enable_pressure_rate();
    }
    if (run.steady()) world.local([&] { op->prepare_steady_steps(run.use_cuda, run.local_time_step, run.local_time_step_max_ratio); });
    Id downgraded = 0;
    if (needs_stencil) {
        auto options = reconstruction;
        if (options.geometry == StencilGeometry::Pl || options.geometry == StencilGeometry::Auto) {
            for (Id p=0;p<part.physical_patches;++p)
                if (solid_wall(boundaries.patches.by_patch[p].type()))
                    options.wall_labels.push_back(part.mesh.data().patch_names[p]);
        }
        double seconds = 0.;
        {
            WallTimer preparation(run.mpi_timing ? &seconds : nullptr);
            downgraded = op->prepare_stencil(part, seam, boundaries.periodic_pairs, input.get(), requests, options);
        }
        if (run.mpi_timing) {
            const double slowest = world.reduce_double(seconds, true);
            if (!world.rank) std::cout << "Stencil preparation (s) = " << slowest << '\n';
        }
    }
#ifdef AERO_HAS_VISCOSITY
    op->prepare_viscosity(part,seam,frame,run);
#ifdef AERO_HAS_TURBULENCE
    if(run.turbulence.uses_sa())
        op->prepare_sa(part,seam,boundaries.patches,boundaries.periodic_pairs,run.turbulence,run.use_cuda,run.uses_implicit_time(),run.time_scheme=="steady");
#endif
#endif
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

#ifdef AERO_WITH_KOKKOS_GPU
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
        op.reserve_gpu(compact, fit, run.uses_implicit_time() ? &run.implicit_settings : nullptr);
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
#ifdef AERO_WITH_KOKKOS_GPU
    if (run.use_cuda) {
        std::string description;
        world.local([&] {
            int device = 0;
            char name[256] = {};
            gpu::checked(gpu::current(&device));
            gpu::checked(gpu::selected_device_name(name, sizeof(name)));
            description = "CUDA device" + std::to_string(device) + ": " + name;
            if (world.size > 1) description += " (MPI rank " + std::to_string(world.rank) + ")";
        });
        if (world.size == 1) {
            std::cout << description << '\n';
            return;
        }
        const auto lines = world.gather_text(description);
        if (!world.rank) for (const auto& line : lines) std::cout << line << '\n';
    }
#endif
}

// Общий цикл времени вызывает эти операции и в обычном расчёте, и при разбиении.
// Поля остаются на выбранном исполнителе. Копию с GPU загружаем на CPU, когда её запросил вывод.
void integrate_partition(const RunSettings& run, const Processes& world, unsigned dimension,
                         PartitionStepper& stepper, PartitionOperator& op, PartitionState& fields,
                         PartitionOutput& output, Recovery& recovery, ExactError* error) {
    StepOperations operations;
    operations.control = [&](Id step) { return stepper.control(step); };
    operations.cfl_step = [&](bool first_order_start, SolverTimings& timing) {
        return stepper.prepare_first_stage(first_order_start, timing);
    };
    if (run.steady())
        operations.steady_residual = [&] { return op.steady_residual(fields.current, fields.initial); };
    bool sample_time=false, write_rotation=false, write_sa=false;
    for (const auto& zone:run.output_zones) for (const auto& name:zone.variables) {
        sample_time |= diagnostic::needs_time(name);
        write_rotation |= name=="vorticity";
        write_sa |= diagnostic::turbulent(name);
    }
    op.start_samples(fields.current,recovery.samples,recovery.has_samples);
    recovery.has_samples=sample_time;
    operations.advance = [&](const TimeStep& step, SolverTimings& timing) {
        if (!stepper.attempt(step,timing)) return false;
        if (sample_time) world.local([&]{op.sample_accepted(fields.current,step.dt);});
        return true;
    };
    if (run.stabilization.active()) operations.accepted = [&] { stepper.accepted(); };
    operations.output_state = [&]() -> const std::vector<double>& {
        if (run.use_cuda) {
            world.local([&] { op.download(fields.current); });
        }
        return fields.current;
    };
    operations.checkpoint_history = [&](implicit::BdfHistory& history) {
        world.local([&] {
            if (run.implicit_settings.second_order) op.implicit_step().export_history(history);
            recovery.stabilization_state = stepper.checkpoint();
            if (sample_time) op.copy_samples(recovery.samples);
#ifdef AERO_HAS_TURBULENCE
            if(recovery.scalar) op.sa_checkpoint(recovery.scalar->current,recovery.scalar->previous,false,history.dt);
#endif
        });
    };
    if (error) operations.exact_error = [&](const TimeStep& step, const std::vector<double>& state) {
        error->write(step.number, step.end, state);
    };
    operations.report = [&](const TimeStep& step, double criterion) { stepper.write_report(step, dimension, stepper.courant(), criterion,
            run.steady_absolute ? recovery.relative_steady_residual(criterion) : criterion); };
    operations.write_fields = [&](const TimeStep& step, const std::vector<double>& snapshot,
                                  bool final_field, SolverTimings& timing) {
        diagnostic::Fields data;
        std::vector<diagnostic::Moments> samples;
        std::vector<double> rotation, scalars, pressure_rate;
        if (output.schedule.column_due(diagnostic::DpDt, step.number, step.end, final_field)) {
            // У BDF уже есть потоки последней итерации Ньютона. У явной схемы
            // готовим потоки принятого состояния и оставляем их для следующего шага.
            stepper.prepare_state_fluxes(run.first_order_start(step.end), timing);
            pressure_rate = op.output_pressure_rate(snapshot);
            data.pressure_rate = pressure_rate.data();
        }
        if (sample_time) world.local([&]{op.copy_samples(samples);data.moments=samples.data();});
        if (write_rotation) { rotation=op.output_vorticity(snapshot);data.rotation=rotation.data(); }
#ifdef AERO_HAS_TURBULENCE
        if (write_sa && dimension==3 && run.turbulence.uses_sigma()) stepper.prepare_state_fluxes(run.first_order_start(step.end),timing);
        if(write_sa) world.local([&]{scalars=op.sa_output(snapshot);data.sa=scalars.data();});
#endif
        output.write(step.number, step.end, snapshot, final_field,data);
    };
    integrate_time(run, world, output.schedule, recovery, fields.initial, operations);
}

// Готовит одну подобласть и ведёт расчёт до конечного времени.
// Возвращает -1, если для MPI с CUDA нужно заново распределить сетку по картам.
int run_partition(RunSettings run, const ReconstructionOptions& reconstruction, const Processes& world,
                  const LeafLayout& layout, GpuMemoryFit* gpu, bool first_attempt, bool archive) {
#ifndef AERO_WITH_KOKKOS_GPU
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
    PartitionState fields;
    std::uint64_t stencil_peak = 0;
    auto op = prepare_partition_operator(run, reconstruction, world, part, geometry, frame, std::move(input),
                                         reconstruct);
    PartitionStepper stepper(*op, fields, world, run);
    if (!single) {
        op->connect(run.use_cuda, run.uses_implicit_time());
    }
#ifdef AERO_WITH_KOKKOS_GPU
    GpuMemoryFit single_fit;
    if (single && run.use_cuda) {
        gpu = &single_fit;
    }
    if (gpu) {
        const bool compact = stepper.needs_compact_buffer();
        if (!prepare_gpu_memory(*op, part, run, world, compact, *gpu)) {
            return -1;
        }
    }
#endif
    const PartitionFileLayout file_layout(run, part, geometry.periodic_partition(), world);
    // Контрольная точка и общий файл полей используют одну таблицу исходных номеров.
    const auto hash = archive ? part.mesh_sha256 : FieldSnapshot::mesh_fingerprint(part.mesh);
    Recovery recovery(run, world, part.global_counts[0], mesh.dimension, hash, file_layout.rows);
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
        recovery.restore(fields.current, fields.initial);
        if (!single) {
            op->halo_host(fields.current);
        }
    }
    if (!single) {
        op->halo_host(fields.initial);
    }

    const auto& base = recovery.fields_directory();
    std::unique_ptr<ExactError> error;
    std::unique_ptr<PartitionOutput> output;
    world.local([&] {
        if (!run.directories.recover) fields.start_from_initial(part, geometry.periodic_partition());
    });
#ifdef AERO_HAS_VISCOSITY
    if (run.viscosity.law != "off") {
        op->apply_wall(fields.initial);
        op->apply_wall(fields.current);
        if (!single) {
            op->halo_host(fields.initial);
            op->halo_host(fields.current);
        }
    }
#endif
    // Старый формат recovery содержит исходное поле, но не его норму.
    // Вычисляем её до восстановления рабочего поля SA и параметров из control.txt.
    if (run.steady() && run.directories.recover && recovery.steady_reference < 0.) {
        SolverTimings timing(world, false, run.use_cuda, 0);
        op->first_order_start = run.first_order_start(0.);
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu) world.local([&] { op->upload_state(fields.initial, fields.initial); });
#endif
        op->assemble_fluxes(fields.initial, fields.initial, timing);
        recovery.steady_reference = op->steady_residual(fields.initial, fields.initial);
    }
    world.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu) {
            op->upload_state(fields.current, fields.initial);
            // Замер цикла начинается после завершения начальной загрузки на карту.
            if (run.mpi_timing)
                PartitionGpu::synchronize();
        }
#endif
        if (run.implicit_settings.second_order && recovery.history.dt > 0.) {
            op->implicit_step().import_history(recovery.history);
        }
#ifdef AERO_HAS_TURBULENCE
        if(run.directories.recover && recovery.scalar)
            op->sa_checkpoint(recovery.scalar->current,recovery.scalar->previous,true,recovery.history.dt);
#endif
        recovery.history = implicit::BdfHistory{};
        if (run.exact_solution != ExactSolution::Off)
            error.reset(new ExactError(run, world, part, geometry.periodic_partition(), file_layout.selection,
                                       periodic ? seam.volumes.data() : cv.volumes, base));
        output.reset(
            new PartitionOutput(run, part, file_layout, world, base, seam.node_dof, op->state_nodes(),geometry.boundaries.patches));
    });
#ifdef AERO_HAS_TURBULENCE
    if (!single && run.turbulence.enabled) op->halo(fields.current,nullptr);
#endif
    if (!single) {
        write_partition_report(world, op->work(part.end_leaf - part.first_leaf), stencil_peak, std::cout);
    }
    if (run.directories.recover) {
        op->first_order_start = run.first_order_start(recovery.time);
        stepper.resume(recovery.stabilization_state);
    }
    print_input_warnings(run, world, std::cerr);
    report_active_backend(run, world);
    integrate_partition(run, world, mesh.dimension, stepper, *op, fields, *output, recovery, error.get());
    return 0;
}
}

int calculate_case(RunSettings run, const ReconstructionOptions& reconstruction, const Processes& world,
                   bool archive) {
#ifdef AERO_WITH_KOKKOS_GPU
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
