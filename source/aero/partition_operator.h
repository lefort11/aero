#ifndef AERO_PARTITION_OPERATOR_H
#define AERO_PARTITION_OPERATOR_H
#include "partition_transfers.h"
#include "../timestep/local_step_mode.h"
#include "mpi_stencil.h"
#include "../io/diagnostic_fields.h"
#include "solver_timings.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/transport.h"
#endif
#include "../implicit/bdf1_step.h"

#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/aes.h"
#endif

namespace aero {
struct RunSettings;
namespace residual_detail { struct FluxCfl; }
// Пространственный оператор одной подобласти. Сетка и её владельцы живут дольше оператора.
class PartitionOperator {
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
    void start_samples(const std::vector<double>& state, std::vector<diagnostic::Moments>& saved, bool restored);
    void sample_accepted(const std::vector<double>& state, double dt);
    void copy_samples(std::vector<diagnostic::Moments>& target) const;
    void enable_pressure_rate() { need_pressure_rate_ = true; }
    // Берёт потоки записываемого слоя и возвращает dp/dt в нумерации полей вывода.
    std::vector<double> output_pressure_rate(const std::vector<double>& state);
    std::vector<double> output_vorticity(const std::vector<double>& state);
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
    void prepare_viscosity(const MeshPartition&, const PeriodicPartition*, const ResidualFrame&,
                           const RunSettings&);
#ifdef AERO_HAS_TURBULENCE
    void prepare_sa(const MeshPartition&,const PeriodicPartition*,const BoundaryPatches&,
                    const std::vector<PeriodicPair>&,const turbulence::Settings&,bool gpu,bool implicit,bool steady,bool has_stencil);
    std::vector<double> sa_output(const std::vector<double>& state);
    void sa_checkpoint(std::vector<double>& current,std::vector<double>& previous,bool restore,double previous_dt);
#endif
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
    void set_low_order(std::vector<double>& marks);
    void set_flux_weights(double upwind, double hybrid);
    void set_courant(double cfl, bool implicit_time);
    void copy_backup(std::vector<double>& state, std::vector<double>& backup, bool restore);
    void failed_nodes(const std::vector<double>& state, std::vector<Id>& failed, bool implicit);
    // Собирает значения только собственных узлов в компактный буфер.
    void copy_owned(const std::vector<double>& state, std::vector<double>& compact) const;
    // Обновляет копию текущих полей на CPU для записи результатов.
    void download(std::vector<double>& state) const;
    // Обменивает значения узлов после стадии на активном вычислительном устройстве.
    void halo(std::vector<double>& u, double* seconds);
    // Заполняет узлы соседей в памяти CPU при подготовке и восстановлении расчёта.
    void halo_host(std::vector<double>& u);
    // Вычисляет потоки на принадлежащих процессу поверхностях и получает остальные от соседей.
    bool assemble_fluxes(const std::vector<double>& u, const std::vector<double>& initial,
                         SolverTimings& timing, bool trial = false);
    // Находит допустимый шаг по собственным узлам и берёт минимум по всем процессам.
    double cfl_step(double cfl);
    // После отказа повторяем ту же попытку с половиной фактически использованного шага.
    void retry_auto_step(double failed_dt);
    void limit_auto_growth(bool hold) { hold_growth_ = hold; }
    void reset_auto_history();
    void prepare_steady_steps(bool gpu, LocalStepMode mode, double max_ratio);
    void set_steady_steps(LocalStepMode mode, double max_ratio);
    double largest_step(double accepted_minimum) const {
        return steady_steps_ && steady_steps_->local ? (accepted_minimum / minimum_step_) * maximum_step_ : 0.;
    }
    // Геометрия и обмены, которыми будет пользоваться матрица на этом процессе.
    implicit::LinearGrid linear_grid();
    implicit::BdfStep& implicit_step(); // Один решатель для интегрирования и контрольных точек.
    // Решает уравнение BDF, используя потоки, подготовленные для CFL.
    // Норма пространственной невязки по текущим потокам всех процессов.
    double steady_residual(std::vector<double>& state, const std::vector<double>& initial);
    implicit::NewtonResult advance_bdf(std::vector<double>& state, const std::vector<double>& initial,
                                        double dt, const implicit::NewtonSettings& settings,
                                        SolverTimings& timing, const std::function<void()>& validate);
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
    bool shared_csgs_ = false; // Способ хранения определяем до создания системы Ньютона.
#endif
    // Подготавливает представление распределённой системы. Формулы и итерации общие для всех режимов.
    struct ImplicitFields {
        double* state;
        const double* initial;
        const double* volumes;
        const EulerFluxResult* flux;
    };
    ImplicitFields implicit_fields(std::vector<double>& state, const std::vector<double>& initial);
    implicit::Partition linear_partition();
    const double* extend_linear_vector(const double* compact);
    std::unique_ptr<PartitionTransfers> derivative_exchange_; // Два блока LF на каждое общее ребро.
    std::vector<double> linear_input_; // Вход A*x в локальной нумерации для CPU, включая соседние узлы.
    PartitionTransfers nodes_; // Значения узлов, общие для соседних подобластей.
    PartitionTransfers edges_; // Результаты расчёта потоков на рёбрах между подобластями.
#ifdef AERO_HAS_VISCOSITY
    std::unique_ptr<viscosity::Aes> aes_;
    bool compact_jacobian_ = false; // Ширину шаблона матрицы задаём до построения таблиц на CPU и GPU.
#endif
#ifdef AERO_HAS_TURBULENCE
    void update_les(const std::vector<double>& state, SolverTimings& timing,
                    viscosity::Evaluation phase=viscosity::Evaluation::LesClosure);
    bool les_active() const { return aes_ && aes_->view().les.scale!=turbulence::Scale::Off && !aes_->view().c_des; }
    bool sa_active() const { return aes_ && aes_->view().sa.active; }
    turbulence::Step sa_step() const;
    void sa_apply(const turbulence::Step&, turbulence::Operation) const;
    double sa_max(const turbulence::Step&, turbulence::Norm,bool refresh=false) const;
    void sa_exchange();
    // Пересчитывает поправку при каждом изменении газа или рабочей вязкости, затем передаёт её соседу.
    void prepare_sa_convection(const turbulence::Step&) const;
    void evaluate_sa(const std::vector<double>& state,SolverTimings& timing,bool rates_ready=false,bool rows_ready=false);
    void solve_sa_stage(const std::vector<double>& state,bool rates_ready,bool rows_ready);
    std::unique_ptr<PartitionTransfers> turbulence_exchange_; // Те же владельцы узлов, один скаляр на запись.
    std::unique_ptr<PartitionTransfers> sa_edge_exchange_; // Одна поправка на общее ребро MPI.
    bool sa_high_order_ = false;
    std::vector<double> host_sa_correction_; // Хранилище потока для CPU.
    double* sa_correction_ = nullptr; // Адрес на активном исполнителе, включая GPU.
    double sa_dt_ = 0., sa_previous_dt_ = 0., sa_target_ = 0., sa_tol_ = 0.;
    unsigned sa_iterations_ = 0;
    bool sa_lines_ = false; // Один выбор пробной поправки на всех MPI-процессах.
    bool sa_bdf2_ = false, sa_implicit_time_ = false;
    bool sa_prediction_pending_ = false; // Газовый предиктор изменил газ при фиксированной nu_tilde.
#endif
    residual_detail::FluxCfl cfl_grid() const;
    double prepare_local_steps(double cfl);
    std::unique_ptr<SteadyNodeSteps> steady_steps_; // Для steady резервируем память сразу, даже при старте с общего шага.
    std::vector<double> host_steps_; // При работе на GPU массив входит в общую раскладку карты.
    LocalStepMode step_mode_ = LocalStepMode::Off;
    const double* accepted_field_ = nullptr; // Поле после принятого шага на активном устройстве.
    double current_step_cfl_ = 0.; // CFL, с которым предложен шаг до ограничения расписанием.
    double accepted_fraction_ = 1., previous_cfl_ = 0., retry_dt_ = 0.;
    unsigned retries_ = 0; // Сбрасывается только после успешного шага или переключения режима.
    bool hold_growth_ = false, recovered_step_ = false;
    double minimum_step_ = 0., maximum_step_ = 0.; // Диапазон до ограничения расписанием.
    bool need_pressure_rate_ = false; // Резервируем рабочую память только при заказе dp_dt.
    std::unique_ptr<PartitionTransfers> pressure_exchange_; // Передаёт dp/dt от владельцев узлов.
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
    void prepare_nodal_gas(); // Резервирует одну запись на собственный или удалённый узел.
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
