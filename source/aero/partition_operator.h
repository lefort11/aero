#ifndef AERO_PARTITION_OPERATOR_H
#define AERO_PARTITION_OPERATOR_H
#include "partition_transfers.h"
#include "../timestep/local_step_mode.h"
#include "../timestep/steady_step_spread.h"
#include "mpi_stencil.h"
#include "../io/diagnostic_fields.h"
#include "solver_timings.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/transport.h"
#endif
#include "../implicit/bdf1_step.h"
#ifdef AERO_HAS_MULTIGRID
#include "../multigrid/schedule.h"
#endif

#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/aes.h"
#endif

namespace aero {
struct RunSettings;
class Report;
struct FlowSummary;
// Все ранги откатывают одну попытку. Каждый помечает только свои отказавшие узлы.
struct RejectedGasState {
    // Номера недопустимых узлов на текущем ранге.
    std::vector<MeshIndex> local_nodes;
    // Полное число зарегистрированных недопустимых состояний.
    std::size_t total;
    // Согласует число отказов всех рангов до перехода к обработке неудачного шага.
    static void check(const Processes& world, std::vector<MeshIndex>& failed);

};
namespace residual_detail { struct FluxCfl; }
// Пространственный оператор одной подобласти. Сетка и её владельцы живут дольше оператора.
class PartitionOperator {
    // Тип номера узла, ребра или строки подобласти.
    using Id = MeshIndex;

public:
    // Строит локальные поверхности, определяет владельцев потоков и готовит геометрию оператора.
    PartitionOperator(const MeshPartition& part, CvView cv, const PeriodicPartition* periodic,
                      const BoundaryPatches& patches, ResidualOptions options, const ResidualFrame& frame,
                      const Processes& world, StencilRequests* requests, bool cuda, bool cell_height_cfl);
    // Число узлов состояния с учётом удалённых узлов шаблонов реконструкции.
    Id state_nodes() const;
    // Средние нужны только при заказе пульсаций. Раскладка GPU учитывает их размер.
    void enable_time_samples() { track_time_ = true; }
    // Инициализирует накопление статистики с учётом восстановленных моментов.
    void start_samples(const std::vector<double>& state, std::vector<diagnostic::Moments>& saved, bool restored);
    // Накопление выполняется там же, где хранится принятое состояние. На GPU
    // за один шаг читаем только узел и его историю, без копирования поля на CPU.
    void sample_accepted(const std::vector<double>& state, double dt);
    // Копирует накопленные моменты поля для вывода или восстановления.
    void copy_samples(std::vector<diagnostic::Moments>& target) const;
    // Включает вычисление производной давления по времени.
    void enable_pressure_rate() { need_pressure_rate_ = true; }
    // Берёт потоки записываемого слоя и возвращает dp/dt в нумерации полей вывода.
    std::vector<double> output_pressure_rate(const std::vector<double>& state);
    // Заполняет запрошенные поля из одного обхода площадок контрольного объёма.
    void output_gradients(const std::vector<double>& state, std::vector<double>* rotation,
                          std::vector<double>* schlieren);
    // Переводит локальный номер узла в исходную глобальную нумерацию.
    Id global_node(Id local) const { return ids_[local]; }
    // Возвращает объём работы и обменов процесса для отчёта о разбиении.
    PartitionWork work(Id leaves) const;
    // Строит шаблоны и добавляет в обмен удалённые узлы, нужные для реконструкции.
    Id prepare_stencil(const MeshPartition& part, const PeriodicPartition* seam,
                       const std::vector<PeriodicPair>& pairs, PartitionInput* input,
                       const StencilRequests& requests, const ReconstructionOptions& options);
    // Согласует с соседями порядок узлов и рёбер и выделяет буферы для повторных обменов.
    void connect(bool cuda, bool implicit_time);
#ifdef AERO_HAS_VISCOSITY
    // Подготавливает геометрию и рабочие массивы вязкого оператора.
    void prepare_viscosity(const MeshPartition&, const PeriodicPartition*, const ResidualFrame&,
                           const RunSettings&);
#ifdef AERO_HAS_TURBULENCE
    // Готовит поле SA, расстояния до стенок и геометрию переноса.
    void prepare_sa(const MeshPartition&,const PeriodicPartition*,const BoundaryPatches&,
                    const std::vector<PeriodicPair>&,const turbulence::Settings&,bool gpu,bool implicit,bool steady,bool has_stencil);
    // Формирует рабочую и физическую вихревую вязкости для вывода.
    std::vector<double> sa_output(const std::vector<double>& state);
    // Сохраняет или восстанавливает текущее и предыдущее поля SA.
    void sa_checkpoint(std::vector<double>& current,std::vector<double>& previous,bool restore,double previous_dt);
#endif
    // При задании u=0 убираем только кинетическую энергию: начальные p и T сохраняются.
    void apply_wall(std::vector<double>& field) const;
#endif
#ifdef AERO_WITH_KOKKOS_GPU
    // Размещает данные сетки и индексы обмена в общей области памяти GPU.
    void layout_gpu(bool compact_buffer);
    // Оценивает объём по той же раскладке и резервирует память карты.
    void reserve_gpu(bool compact_buffer, GpuMemoryFit& fit, const implicit::NewtonSettings* settings);
    // Передаёт начальное и текущее состояния на GPU перед первым шагом.
    void upload_state(const std::vector<double>& state, const std::vector<double>& initial);
#endif
    // Обновляет порядок со стороны отмеченных узлов после согласования маски по MPI.
    bool first_order_start = false; // На текущем шаге все рёбра используют узловые состояния до t_fo.
    // Подключает метки принудительного первого порядка.
    void set_low_order(std::vector<double>& marks);
    // Пересобираем матрицу только при ручном изменении потока. Для BDF2 переносим предыдущий слой.
    void set_flux_weights(double upwind, double hybrid);
    // Обновляет CFL и согласованный с интегратором параметр MP.
    void set_courant(double cfl, bool implicit_time);
    // Сохраняет поле перед попыткой шага или восстанавливает его после отказа.
    void copy_backup(std::vector<double>& state, std::vector<double>& backup, bool restore);
    // Собирает номера узлов с недопустимым газодинамическим состоянием.
    // Возвращает true, если за эту проверку подготовлен газ всех узлов для следующего пакета потоков.
    bool failed_nodes(const std::vector<double>& state, std::vector<Id>& failed, bool implicit);
    // Собирает значения только собственных узлов в компактный буфер.
    void copy_owned(const std::vector<double>& state, std::vector<double>& compact) const;
#ifdef AERO_WITH_KOKKOS_GPU
    // Сводка принятого steady-шага на одной карте с запасным путём расширенной точности.
    FlowSummary steady_report(std::uint64_t number, Report& report, std::vector<double>& host) const;
#endif
    // Обновляет копию текущих полей на CPU для записи результатов.
    void download(std::vector<double>& state) const;
    // Обменивает значения узлов после стадии на активном вычислительном устройстве.
    void halo(std::vector<double>& u, double* seconds);
    // Заполняет узлы соседей в памяти CPU при подготовке и восстановлении расчёта.
    void halo_host(std::vector<double>& u);
    // Вычисляет потоки на принадлежащих процессу поверхностях и получает остальные от соседей.
    bool assemble_fluxes(const std::vector<double>& u, const std::vector<double>& initial,
                         SolverTimings& timing, bool trial = false, bool retry_boundary = false,
                         bool solve_trial_sa = false, bool validated_gas = false);
    // Находит допустимый шаг по собственным узлам и берёт минимум по всем процессам.
    double cfl_step(double cfl);
    // После отказа повторяем ту же попытку с половиной фактически использованного шага.
    void retry_auto_step(double failed_dt);
    // После отката сокращаем уже согласованные шаги всех узлов в одной пропорции.
    double reduce_auto_attempt(double failed_dt, double factor) {
#ifdef AERO_HAS_TURBULENCE
        sa_rate_dt_ = 0.; // Баланс отказавшего шага не входит в следующий прогноз SA.
#endif
        retry_dt_ = failed_dt * factor;
        if (!(retry_dt_ > 0. && retry_dt_ < failed_dt))
            throw std::runtime_error("Auto LTS: cannot reduce the rejected pseudo-time step further");
        return retry_dt_;
    }
    // Сбрасывает историю адаптации и расписание коррекций после смены схемы.
    void reset_rate_history() {
        rate_samples_ = 0;
        if (bdf_) bdf_->clear_prediction_history();
#ifdef AERO_HAS_TURBULENCE
        sa_rate_dt_ = 0.;
#endif
        if (step_spread_) step_spread_->reset();
#ifdef AERO_HAS_MULTIGRID
        coarse_schedule_ = multigrid::CycleSchedule{}; // Новая пространственная схема начинает свой цикл.
#endif
    }
#ifdef AERO_HAS_MULTIGRID
    // Возвращает расписание попыток многосеточной коррекции.
    multigrid::CycleSchedule& coarse_schedule() { return coarse_schedule_; }
#endif
    // work, cfl и stencil используют общий регулятор диапазона.
    SteadyStepSpread* step_spread() {
        return step_mode_ == LocalStepMode::Auto ? step_spread_.get() : nullptr;
    }
    // Возвращает фактическое отношение шагов последнего обновления.
    double applied_step_ratio() const { return applied_ratio_; }
    // Проверяет ограничение локальных шагов по скорости изменения поля.
    bool has_node_rate_control() const { return rate_control_; }
    // Суммирует по процессам число узлов, ограниченных по невязке.
    std::uint64_t limited_rate_nodes() const { return world_.sum(rate_limited_); }
    // Суммирует по процессам число узлов, достигших предела шага.
    std::uint64_t capped_rate_nodes() const { return world_.sum(rate_capped_); }
    // Ищем владельца один раз. Для отчёта читаем только данные выбранного узла.
    void set_rate_probe(MeshIndex id);
    // Считывает диагностику шага и число узлов с растущей невязкой.
    void read_rate_probe(double accepted_dt, double* values, MeshIndex* growth_nodes) const;
    // Готовит данные ограничения шагов по опорам реконструкции.
    void prepare_stencil_rates(bool cuda);
    // Настраивает локальные псевдошаги стационарного расчёта.
    void prepare_steady_steps(bool gpu, LocalStepMode mode, double max_ratio, ResidualLimit residual_limit,
                              const AdaptiveStepSettings& adaptive_step);
    // Готовит второй слой шагов и обмен скалярами на рёбрах между процессами.
    void prepare_neighbour_step_limit(double ratio, bool cuda);
    // Возвращает отношение соседних шагов до сглаживания.
    double neighbour_ratio_before() const { return neighbour_limit_ && steady_steps_->local ? neighbour_limit_->before : 0.; }
    // Возвращает отношение соседних шагов после сглаживания.
    double neighbour_ratio_after() const { return neighbour_limit_ && steady_steps_->local ? neighbour_limit_->after : 0.; }
    // Меняет режим локальных шагов и сбрасывает несовместимую историю адаптации.
    void set_steady_steps(LocalStepMode mode, double max_ratio);
    // Масштабирует наибольший локальный шаг по фактически принятому минимуму.
    double largest_step(double accepted_minimum) const {
        return steady_steps_ && steady_steps_->local ? (accepted_minimum / minimum_step_) * maximum_step_ : 0.;
    }
    // Геометрия и обмены, которыми будет пользоваться матрица на этом процессе.
    implicit::LinearGrid linear_grid();
    // Лениво создаёт неявный интегратор подобласти.
    implicit::BdfStep& implicit_step(); // Один решатель для интегрирования и контрольных точек.
    // Решает уравнение BDF, используя потоки, подготовленные для CFL.
    // Норма пространственной невязки по текущим потокам всех процессов.
    double steady_residual(std::vector<double>& state, const std::vector<double>& initial, bool accepted = false,
                           bool observe = false);
    // Выполняет неявный шаг газа с согласованным обновлением турбулентности.
    implicit::NewtonResult advance_bdf(std::vector<double>& state, const std::vector<double>& initial,
                                        double dt, const implicit::NewtonSettings& settings,
                                        SolverTimings& timing, const std::function<bool()>& validate);
    // Обновляет собственные узлы на одной стадии интегратора.
    void update_owned(explicit_detail::Stage stage, std::vector<double>& start, double dt,
                      std::vector<double>& state, bool save_original);

    std::vector<MeshIndex> owned; // Локальные номера узлов, которые обновляет этот процесс.
private:
    // Добавляет рёбра с собственным узлом и запоминает, кому нужно передать поток.
    void add_edge_surfaces(MeshView mesh, CvView cv, const PeriodicPartition* periodic,
                           const ResidualFrame& frame, StencilRequests* requests);
    // Добавляет физические границы, за которые отвечает эта подобласть.
    void add_boundary_surfaces(CvView cv, const PeriodicPartition* periodic, Id physical_patches,
                               const BoundaryPatches& patches, const ResidualFrame& frame,
                               double gamma, StencilRequests* requests);
    // Возвращает активное устройство или nullptr для обмена в памяти CPU.
    PartitionGpu* device() const;
#ifdef AERO_WITH_KOKKOS_GPU
    std::unique_ptr<PartitionGpu> gpu_; // Раскладка, рабочая память и ядра текущей карты.
    std::shared_ptr<implicit::GpuMatrixPlan> matrix_plan_; // Фактический граф расчёта памяти переходит в решатель.
    std::shared_ptr<const multigrid::Aggregates> coarse_graph_; // Граф агрегатов из оценки памяти GPU.
    bool shared_csgs_ = false; // Способ хранения определяем до создания системы Ньютона.
    bool shared_step_start_ = false; // BDF1 и откат используют одно неизменное начальное поле.
#endif
    // Подготавливает представление распределённой системы. Формулы и итерации общие для всех режимов.
    struct ImplicitFields {
        // Изменяемое поле консервативных переменных.
        double* state;
        // Начальное поле для задания граничных условий.
        const double* initial;
        // Объёмы, входящие во временную диагональ.
        const double* volumes;
        // Потоки текущего приближения на гранях.
        const EulerFluxResult* flux;
    };
    // Подключает актуальные CPU- или GPU-поля к неявному интегратору.
    ImplicitFields implicit_fields(std::vector<double>& state, const std::vector<double>& initial);
    // Возвращает максимальную невязку SA либо ноль при выключенной модели.
    double steady_sa_rate(); // Глобальная пространственная норма переноса SA, ноль для чистого газа.
#ifdef AERO_HAS_MULTIGRID
    multigrid::CycleSchedule coarse_schedule_; // Следующая проба только после требуемых обычных шагов.
    // Пробует грубосеточную коррекцию по текущему расписанию.
    void try_steady_multigrid(std::vector<double>& state, const std::vector<double>& initial,
                              double dt, const implicit::NewtonSettings& settings, SolverTimings& timing);
#endif
    // Задаёт обмены и редукции распределённой линейной системы.
    implicit::Partition linear_partition();
    // Раскладываем только собственные компоненты x, затем используем обычный обмен узлами.
    // На GPU и рабочее поле, и индексы упаковки остаются в памяти выбранной карты.
    const double* extend_linear_vector(const double* compact);
    std::unique_ptr<PartitionTransfers> derivative_exchange_; // Два блока LF на каждое общее ребро.
    std::vector<double> linear_input_; // Вход A*x в локальной нумерации для CPU, включая соседние узлы.
    PartitionTransfers nodes_; // Значения узлов, общие для соседних подобластей.
    PartitionTransfers edges_; // Результаты расчёта потоков на рёбрах между подобластями.
#ifdef AERO_HAS_VISCOSITY
    // Владелец вязкой геометрии подобласти.
    std::unique_ptr<viscosity::Aes> aes_;
    bool compact_jacobian_ = false; // Ширину шаблона матрицы задаём до построения таблиц на CPU и GPU.
#endif
#ifdef AERO_HAS_TURBULENCE
    // Обновляет замыкание LES по текущему газовому полю.
    void update_les(const std::vector<double>& state, SolverTimings& timing,
                    viscosity::Evaluation phase=viscosity::Evaluation::LesClosure);
    // Проверяет, включено ли LES-замыкание без DDES.
    bool les_active() const { return aes_ && aes_->view().les.scale!=turbulence::Scale::Off && !aes_->view().c_des; }
    // Проверяет наличие активного переноса рабочей вязкости SA.
    bool sa_active() const { return aes_ && aes_->view().sa.active; }
    // Формирует данные одного шага уравнения SA.
    turbulence::Step sa_step() const;
    // Выполняет выбранную операцию SA на CPU или GPU.
    void sa_apply(const turbulence::Step&, turbulence::Operation) const;
    // Вычисляет глобальную максимальную норму SA.
    double sa_max(const turbulence::Step&, turbulence::Norm,bool refresh=false) const;
    // Обменивает рабочую вязкость с соседними подобластями.
    void sa_exchange();
    // Пересчитывает поправку при каждом изменении газа или рабочей вязкости, затем передаёт её соседу.
    void prepare_sa_convection(const turbulence::Step&) const;
    // Собирает оператор SA с учётом уже подготовленных потоков и строк.
    void evaluate_sa(const std::vector<double>& state,SolverTimings& timing,bool rates_ready=false,bool rows_ready=false,
                     bool reference_only=false);
    // Решает стадию SA с учётом готовности потоков и диффузионных строк.
    void solve_sa_stage(const std::vector<double>& state,bool rates_ready,bool rows_ready,bool reference_only);
    std::unique_ptr<PartitionTransfers> turbulence_exchange_; // Те же владельцы узлов, один скаляр на запись.
    std::unique_ptr<PartitionTransfers> sa_edge_exchange_; // Одна поправка на общее ребро MPI.
    // Использовать газовый шаблон для конвекции SA.
    bool sa_high_order_ = false;
    std::vector<double> host_sa_correction_; // Хранилище потока для CPU.
    double* sa_correction_ = nullptr; // Адрес на активном исполнителе, включая GPU.
    // Текущий и предыдущий шаг SA, целевое значение и допуск итераций.
    double sa_dt_ = 0., sa_previous_dt_ = 0., sa_target_ = 0., sa_tol_ = 0.;
    double sa_rate_dt_ = 0.; // Интервал между двумя последними балансами SA для пробного поля steady LTS.
    // Число итераций уточнения SA.
    unsigned sa_iterations_ = 0;
    bool sa_lines_ = false; // Один выбор пробной поправки на всех MPI-процессах.
    bool sa_multigrid_ = false; // Ускоренный сглаживатель SA во время стационарного цикла.
    // Признаки двухслойной истории и неявного интегрирования SA.
    bool sa_bdf2_ = false, sa_implicit_time_ = false;
    bool sa_prediction_pending_ = false; // Газовый предиктор изменил газ при фиксированной nu_tilde.
#endif
    // Для общего и локального шага используем те же геометрию, газ и вязкие ограничения.
    residual_detail::FluxCfl cfl_grid() const;
    // Вычисляет локальные шаги по CFL и возвращает общий минимум.
    double prepare_local_steps(double cfl);
    // Уменьшает шаги до выполнения ограничения на соседних узлах.
    void limit_neighbour_steps();
    // Вся дополнительная память принадлежит опции local_time_step_neighbor_ratio.
    // Два поля dt меняются ролями: обход ни на CPU, ни на GPU не перезаписывает вход.
    struct NeighbourStepLimit {
        // Предел отношения соседних шагов и экстремумы до и после ограничения.
        double ratio = 0., before = 0., after = 0.;
        // Рабочий массив итерации распространения ограничений на CPU.
        std::vector<double> host_scratch;
        double* scratch = nullptr; // Свободное поле; второе всегда доступно через steady_steps_->values.
        // Обмен граничными значениями шагов между подобластями.
        std::unique_ptr<PartitionTransfers> halo;
    };
    // Рабочие данные ограничения шага между соседями.
    std::unique_ptr<NeighbourStepLimit> neighbour_limit_;
    std::unique_ptr<SteadyNodeSteps> steady_steps_; // Для steady резервируем память сразу, даже при старте с общего шага.
    std::vector<double> host_steps_; // При работе на GPU массив входит в общую раскладку карты.
    // Действующий способ выбора локального псевдошага.
    LocalStepMode step_mode_ = LocalStepMode::Off;
    const double* accepted_field_ = nullptr; // Поле после принятого шага на активном устройстве.
    double current_step_cfl_ = 0.; // CFL, с которым предложен шаг до ограничения расписанием.
    // Принятый множитель узлового массива, CFL принятой попытки и шаг повтора.
    double accepted_fraction_ = 1., previous_cfl_ = 0., retry_dt_ = 0.;
    unsigned retries_ = 0; // Сбрасывается только после успешного шага или переключения режима.
    // Признак загруженного состояния регулятора шага.
    bool recovered_step_ = false;
    std::uint64_t rate_samples_ = 0; // После сброса первая принятая невязка инициализирует историю.
    MeshIndex rate_capped_ = 0; // Действующая доля местного CFL меньше единицы.
    MeshIndex rate_limited_ = 0; // Сколько собственных шагов уменьшено до согласования по рёбрам.
    MeshIndex probe_row_ = ~MeshIndex(0); // Только владелец хранит номер строки для наблюдения.
    int probe_owner_ = -1; // После выбора узла его владелец известен всем процессам.
    // Максимальные узловые скорости изменения газа и рабочей вязкости.
    double gas_rate_max_ = 0., sa_rate_max_ = 0.;
    std::unique_ptr<SteadyStepSpread> step_spread_; // Одно состояние на весь процесс, независимо от числа узлов.
    bool cfl_after_limit_ = false; // В режиме auto + cfl узловой массив хранит шаги при CFL=1.
    AdaptiveStepSettings adaptive_step_; // Параметры auto действуют также после переключения из control.txt.
    double applied_ratio_ = 0.; // Отношение шагов, разрешённое при выборе принятого шага.
    // Поля пар (газ, SA) нужны только при проверке по шаблону. Строки реконструкции общие с потоком.
    struct StencilRateStorage {
        // Рабочие значения критерия в узлах и на рёбрах.
        double *nodes = nullptr, *edges = nullptr;
        // Владельцы соответствующих массивов при расчёте на CPU.
        std::vector<double> host_nodes, host_edges;
        unsigned stride = 2; // При наблюдении за узлом пересылаем также источник каждой оценки.
        // Обмен значениями критерия на стыке подобластей.
        std::unique_ptr<PartitionTransfers> node_exchange, edge_exchange;
    };
    // Данные проверки невязки на шаблоне реконструкции.
    std::unique_ptr<StencilRateStorage> stencil_rates_;
    // Межпроцессное ребро вычисляет тот же владелец, который строил обе строки его реконструкции.
    void collect_stencil_rates(const double* sa_rate, const EulerFluxResult* flux);
    bool rate_control_ = false; // Память под историю нужна только при включённом ограничителе.
    // Сохранённые узловые значения для сравнения соседних шагов.
    double* rate_history_ = nullptr;
    std::vector<double> host_rate_history_; // Три double на собственный узел для steady, включая переключение из control.txt.
    double minimum_step_ = 0., maximum_step_ = 0.; // Границы в единицах узлового массива.
    bool need_pressure_rate_ = false; // Резервируем рабочую память только при заказе dp_dt.
    std::unique_ptr<PartitionTransfers> pressure_exchange_; // Передаёт dp/dt от владельцев узлов.
    // Включает сбор времени операций подобласти.
    bool track_time_ = false;
    std::vector<diagnostic::Moments> moments_; // На CPU хранит средние в расчётной нумерации.
#ifdef AERO_WITH_KOKKOS_GPU
    double* device_pressure_rate_ = nullptr; // По одному double на собственный узел.
    const double* device_heights_ = nullptr; // Высоты загружаются один раз при подготовке GPU.
    diagnostic::Moments* device_moments_ = nullptr; // Владельцем памяти остаётся PartitionGpu.
#endif
    unsigned width_;           // Число консервативных переменных в одном узле.
    const double* volumes_;    // Объёмы контрольных ячеек в расчётной нумерации.
    const double* heights_;    // Наименьшая высота у узла; nullptr для CFL по площадкам.
    const std::vector<Id>&
        ids_; // Исходный номер каждого локального узла или его периодического представителя.
    const std::vector<int>& owners_;                 // MPI-процесс, отвечающий за каждый узел.
    const Processes& world_;                         // Все процессы этого расчёта.
    // Выделяет узловой кэш газовых параметров для расчёта потоков.
    void prepare_nodal_gas(); // Резервирует одну запись на собственный или удалённый узел.
    // Упаковывает рёбра и строки WENO в блоки совместной обработки.
    template <unsigned order> void prepare_weno_tiles(std::vector<residual_detail::WenoEdgeTile<order>>& tiles); // Готовит общую для CPU и GPU геометрию полных двумерных шаблонов WENO3 и WENO5.
    std::vector<residual_detail::WenoEdgeTile<3>> weno3_tiles_; // По 32 шаблона третьего порядка.
    std::vector<residual_detail::WenoEdgeTile<5>> weno5_tiles_; // Дальние наклоны нужны пятому порядку.
    std::vector<MeshIndex> other_faces_; // Оставшиеся поверхности этого процесса и физические границы.
    std::vector<residual_detail::FaceNormal> normals_; // Направление потока на каждом ребре, на CPU.
    std::vector<EulerState> nodal_gas_; // Величины текущей стадии для рёбер на CPU.
    bool gas_current_ = false; // В однопроцессном расчёте стадия обновляет сразу и поле, и этот массив.
    PartitionStencil stencil_;                       // Строки реконструкции и дополнительные удалённые узлы.
    std::vector<ProfileState> boundary_profiles_; // Профили физических площадок в порядке сборки потоков.
    std::vector<residual_detail::Surface> surfaces_; // Сначала рёбра, затем части физических границ.
    std::vector<EulerFluxResult> flux_;              // Потоки для каждой поверхности при расчёте на CPU.
    std::vector<Id> compute_; // Индексы поверхностей, на которых процесс сам вычисляет поток.
    std::vector<residual_detail::Index> offsets_;          // Начало списка поверхностей каждого узла.
    std::vector<residual_detail::OrientedFace> incidence_; // Связи узлов с прилегающими поверхностями.
    residual_detail::Geometry grid_{}; // Представление этих массивов для вычислительных ядер.
    std::unique_ptr<implicit::BdfStep> bdf_; // Матрица и рабочие поля между неявными шагами.
};
}
#endif
