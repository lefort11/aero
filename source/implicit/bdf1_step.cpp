#include "bdf1_step.h"
#include "linear_iteration_report.h"
#include <utility>
#include "bdf1_linear_system.h"
#include "newton_vectors.h"
#include "time_prediction.h"
#include "workspace.h"
#ifdef AERO_HAS_MULTIGRID
#include "../multigrid/coarse_step.h"
#endif
#include "../timestep/bdf_weights.h"
#include "../parallel/parallel_for.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_api.h"
#endif
#include <cmath>
#include <sstream>

namespace aero {
namespace implicit {
namespace {
// Предиктор пробует поправку от временной диагонали либо от готового предобусловливателя.
// Каждая проба использует те же потоки, тот же обмен и ту же норму, что и Ньютон.
template<class Update, class Share, class Residual>
double predict_initial_field(double norm, double initial_norm, const NewtonSettings& settings, const Update& update,
                             const Share& share, const Residual& residual,
                             const std::function<void()>& refresh, const std::function<bool()>& trial_flux,
                             bool viscous, bool use_history, bool damp, const std::function<bool()>& confirm_prediction,
                             bool local_rans, unsigned limit) {
    if (norm == 0.) return confirm_prediction && confirm_prediction() ? residual() : norm;
    // При быстрой сходимости последовательные подстановки дешевле сборки AES
    // и решения линейной системы. Для вязкого шага оставляем запас по допуску,
    // чтобы точность поля была близка к получаемой после поправки Ньютона.
    const double target_multiplier = local_rans ? 1. : (viscous ? .02 : (use_history ? 1. : 4.));
    const double target = initial_norm * std::min(.8, target_multiplier * settings.newton_relative);
    // На первом приближении уточняем связанные уравнения сразу. Затем проверяем их
    // на полпути к допуску по логарифмической шкале и обязательно перед принятием.
    const double coupled_reduction = confirm_prediction ? std::sqrt(settings.newton_relative) : 1.;
    double next_coupled_check = local_rans ? target : norm;
    // После медленной подстановки слегка демпфируем следующую пробу. Это гасит
    // знакочередующиеся пространственные моды, а невязка остаётся полной.
    double fraction = 1.;
    for (unsigned k = 0; k < limit && norm > target; ++k) {
        update(NewtonUpdate::Predict, 1.);
        if (fraction != 1.) update(NewtonUpdate::RelaxPrediction, fraction);
        share();
        double trial = trial_flux() ? residual() : HUGE_VAL;
        if ((viscous || damp) && std::isfinite(trial) && !(trial < .8 * norm)) {
            // На жёстком вязком члене полная подстановка может перебросить поле
            // через корень. При противоположных знаках одной линейной моды
            // доля norm/(norm+trial) даёт нуль этой моды между двумя полями.
            // Для системы это только проба, её проверяем по полной невязке.
            const double relaxation = 1. / (1. + trial/norm);
            update(NewtonUpdate::RelaxPrediction, relaxation);
            share();
            trial = trial_flux() ? residual() : HUGE_VAL;
        }
        if (!(trial < .8 * norm)) {
            // После отката нужны и потоки, и правая часть для восстановленного поля.
            update(NewtonUpdate::UndoPrediction, 1.);
            share();
            refresh();
            return residual();
        }
        if (damp && trial > .2 * norm) fraction = .75;
        norm = trial;
        // Пока подбирается газовое приближение, SA может использовать последнее
        // рабочее поле. Перед принятием ещё раз проверяем оба уравнения совместно.
        if (confirm_prediction && (norm <= next_coupled_check || norm <= target)) {
            if (confirm_prediction()) norm = residual();
            next_coupled_check = coupled_reduction * norm;
        }
    }
    if (confirm_prediction && confirm_prediction()) norm = residual();
    return norm;
}
}

// Матрица, геометрия, история и рабочие векторы неявного шага.
struct BdfStep::Storage {
    BdfLinearSystem linear;         // Матрица и рабочая память BiCGSTAB используются на всех шагах.
    PredictionHistory predictions;  // Принятые слои в свободных векторах линейной системы.
    bool prediction_capacity_set = false; // Все ранги согласовали глубину истории перед первой пробой.
    bool predicts_rates = false; // В кольце хранятся пространственные балансы газа при автоматическом LTS.
    residual_detail::Geometry grid; // Геометрия пространственной невязки на выбранном устройстве.
    Partition partition;            // Списки собственных строк и обмен обновлённым полем.
    MeshIndex rows;                 // Число узлов, за которые отвечает процесс.
    int device;                     // Отрицательный номер означает расчёт на CPU.
    std::vector<double> host;       // Три непересекающихся поля для расчёта на CPU.
    // Владелец выделений на выбранном устройстве.
    Workspace workspace;
    double* start;       // Копия принятого поля в начале текущего шага.
    std::vector<double> detached_start; // Собственная копия при последующем включении BDF2.
    std::vector<double> history_host; // Выделяется только при первом использовании BDF2 на CPU.
    double* previous = nullptr;    // Предыдущий слой BDF2. После принятия шага меняется местами со start.
    // Длительность предыдущего принятого шага для коэффициентов BDF2.
    double previous_dt = 0.;
    bool cached = false;            // В матрице сохранены производные для последующих шагов.
    bool cached_diagonal = false;   // Чистая диагональ нужна только при переносе матрицы между шагами.
    // Сколько принятых шагов прошло с последней сборки матрицы.
    unsigned accepted_since_build = 0;
    MatrixSettings cached_settings; // Переключение потока или предобуславливателя требует полной сборки.
    double* rhs;         // Полная невязка с обратным знаком для решения системы.
    bool saved_balance = false; // В rhs лежит подписанный пространственный баланс принятого слоя steady.
    double* correction;  // Последняя поправка BiCGSTAB к состоянию узлов.
#ifdef AERO_HAS_MULTIGRID
    residual_detail::Geometry host_grid; // Адреса связности на CPU для однократного построения агрегатов.
    // Владелец грубой коррекции в режиме steady.
    std::unique_ptr<multigrid::CoarseStep> coarse;
    bool matrix_on_step = false; // Предиктор мог закончить шаг без единой сборки матрицы.
#endif

    // Создаёт матрицу и размечает рабочие поля Ньютона для собственной части подобласти.
    Storage(const residual_detail::Geometry& topology, const residual_detail::Geometry& active, int id, const Partition& part)
        : linear(topology, active, id, part), predictions(linear.prediction_storage()), grid(active), partition(part),
          rows(part.extend_vector ? part.rows : grid.nodes), device(id) {
#ifdef AERO_HAS_MULTIGRID
        host_grid = topology;
#endif
        if (rows > host.max_size() / (NewtonFieldCount * grid.components()))
            throw std::length_error("BDF: field too large");
        const auto size = rows * grid.components();
        const auto capacity = std::max<MeshIndex>(1, (NewtonFieldCount-unsigned(part.step_start!=nullptr))*size);
        auto* fields = workspace.allocate(device, host, capacity);
        start = part.step_start ? part.step_start : fields;
        rhs = part.step_start ? fields : fields+size;
        correction = rhs + size;
    }

    // Новая последовательность сборок начинается после изменения настроек или
    // по истечении заданного числа принятых шагов. Потоки для невязки всегда свежие.
    void begin_matrix_step(const NewtonSettings& settings) {
        cached = cached && cached_diagonal &&
                 (grid.options.background.active() || accepted_since_build < settings.jacobian_update_every) &&
                 settings.roe == cached_settings.roe &&
                 settings.simplified_jacobian == cached_settings.simplified_jacobian &&
                 settings.csgs == cached_settings.csgs;
    }

    // Возвращает единицу, если пришлось заново дифференцировать потоки.
    unsigned prepare_matrix(const double* state, const double* initial, const double* volumes,
                            double effective_dt, const NewtonSettings& settings, unsigned iteration,
                            const PhaseTiming& timing) {
#ifdef AERO_HAS_MULTIGRID
        matrix_on_step = true;
#endif
        if (!cached) {
            // Для линейных уравнений пространственные производные постоянны.
            // В упрощённом Ньютоне сохраняем их вместе с готовым предобусловливателем.
            const bool reuse = grid.options.background.active() || settings.simplified_newton();
            // При интервале 1 время и локальные шаги внутри Ньютона неизменны,
            // а следующий принятый шаг соберёт новую матрицу. Копия A_kk не нужна.
            cached_diagonal=grid.options.background.active() || settings.jacobian_update_every>1;
            linear.assemble(state, initial, volumes, effective_dt, settings, cached_diagonal, timing);
            cached = reuse;
            cached_settings = settings;
            accepted_since_build = 0;
            return 1;
        }
        // Для первого решения на новом шаге обновляем V*a/dt и разложение
        // диагонали. Остальные поправки используют ту же готовую матрицу.
        if (iteration == 0) timing.measure(PhaseTiming::PreconditionerSetup,[&] {
            linear.update_time_diagonal(volumes,effective_dt);
        });
        return 0;
    }

    // Выделяет историю предыдущего слоя, когда она требуется формулой BDF2.
    void ensure_history() {
        if (previous) return;
        const auto size = std::max<MeshIndex>(1, rows * grid.components());
        // BDF2 меняет местами start и previous. Заимствованная копия отката
        // не должна стать историей: перед первым BDF2 переходим на своё поле.
        // ensure_history вызывается до InitialRhs, который заново запишет U_n.
        if (start==partition.step_start)
            start=workspace.allocate(device,detached_start,size);
        previous = workspace.allocate(device, history_host, size);
    }

    // Вычисляет глобальную норму полной невязки неявного шага.
    double residual_norm(const NewtonUpdate& operation) const {
        double local = 0.;
        partition.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
            if (device >= 0) {
                device_work::check(device_work::aero_device_newton_residual(device,rows,operation,&local));
                return;
            }
#endif
            host_parallel::ensure_runtime();
            local = form_residual_and_norm(host_parallel::Execution(),rows,operation);
        });
        return partition.reduce ? partition.reduce(local,true) : local;
    }

    // Одним обходом вычисляем норму и, по запросу, сохраняем узловые нормы
    // для LTS либо подписанную интегральную невязку для грубой системы.
    double measure_spatial_balance(const double* volumes, const EulerFluxResult* flux,
                                   double* rates, double* negative_balance) const {
        const double norm = residual_norm({grid,flux,volumes,nullptr,nullptr,negative_balance,rates,nullptr,
                                           0.,0.,0.,NewtonUpdate::SpatialRate,partition.owned,1.,{}});
        if (!std::isfinite(norm)) throw std::domain_error("Steady: nonfinite spatial residual");
        return norm;
    }

    // Сохраняет начало принятого шага до перестановки слоёв BDF2.
    void remember(double dt) {
        double* dest = predictions.append(dt);
        if (!dest || !rows) return;
        const auto size = rows * grid.components();
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            gpu::DeviceScope scope(device);
            gpu::checked(scope.status);
            gpu::checked(gpu::copy(dest, start, size * sizeof(double), gpu::Copy::OnDevice));
            return;
        }
#endif
        std::copy_n(start, size, dest);
    }

    // Если часть векторов на каком-либо ранге совмещена с потоками, все ранги
    // должны выбрать ту же степень и вступить в одни и те же обмены поля.
    void configure_prediction_capacity() {
        if (prediction_capacity_set) return;
        if (partition.reduce) {
            const double borrowed = partition.reduce(partition.krylov_scratch.count, true);
            predictions.limit_capacity(KrylovFieldCount - static_cast<unsigned>(borrowed));
        }
        prediction_capacity_set = true;
    }

    // Применяет полином к собственным строкам. Halo обновляет вызывающий код.
    void extrapolate(const TimePrediction& trial) const {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            device_work::check(device_work::aero_device_time_prediction(device, rows, trial));
            return;
        }
#endif
        parallel_for(rows, trial);
    }

#ifdef AERO_WITH_KOKKOS_GPU
    // На цельной плоской сетке уточняем наиболее активные строки до общей
    // проверки газа и SA. В остальных раскладках достаточно временной пробы.
    void predict_local_rows(const NewtonUpdate& gas,const double* initial,double threshold) const {
        if (!partition.trial_faces || partition.owned || rows!=grid.nodes || grid.faces>=rows*4 ||
            grid.dim!=2 || !grid.packed.tiles || !grid.packed.gas ||
            grid.options.variables!=ResidualVariables::Characteristic || grid.stencil.mp5.weno ||
            grid.options.mp6_hybrid!=0.) return;
        device_work::check(device_work::aero_device_local_gas_prediction(device,rows,gas,
                                              partition.trial_flux,partition.trial_faces,initial,threshold));
    }
#endif

    // Выполняет выбранное обновление полей Ньютона на CPU или GPU.
    void update(const NewtonUpdate& operation) const {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            device_work::check(device_work::aero_device_newton_update(device, rows, operation));
            return;
        }
#endif
        parallel_for(rows, operation);
    }
};

// Создаёт неявный интегратор с геометрией и обменами заданной подобласти.
BdfStep::BdfStep(const residual_detail::Geometry& host, const residual_detail::Geometry& active, int device, const Partition& part)
    : storage_(new Storage(host, active, device, part)) {}
// Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
BdfStep::~BdfStep() = default;

// Возвращает сохранённое поле начала текущего шага.
const double* BdfStep::start_field() const { return storage_->start; }
// Возвращает компактные строки для промежуточного результата отчёта на GPU.
double* BdfStep::report_workspace() const {
    storage_->saved_balance = false;
    return storage_->rhs;
}

// Копирует предыдущий слой BDF и его шаг для контрольной точки.
void BdfStep::export_history(BdfHistory& history) const {
    const auto& s = *storage_;
    history.dt = s.previous_dt;
    if (!history.dt) {
        history.previous.clear();
        return;
    }
    const unsigned width = s.grid.components();
    history.previous.resize(s.grid.nodes * width);
    const double* source = s.previous;
#ifdef AERO_WITH_KOKKOS_GPU
    std::vector<double> staging;
    if (s.device >= 0 && s.rows) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        staging.resize(s.rows * width);
        gpu::checked(gpu::copy(staging.data(), source, staging.size() * sizeof(double), gpu::Copy::ToHost));
        source = staging.data();
    }
#endif
    // Файл индексируется номерами узлов исходной сетки через полное локальное поле.
    for (MeshIndex i = 0; i < s.rows; ++i) {
        const auto node = s.partition.host_owned ? s.partition.host_owned[i] : i;
        std::copy_n(source + i * width, width, history.previous.data() + node * width);
    }
}
// Восстанавливает предыдущий слой BDF и длительность его шага.
void BdfStep::import_history(const BdfHistory& history) {
    auto& s = *storage_;
    const unsigned width = s.grid.components();
    if (!(history.dt > 0.) || !std::isfinite(history.dt) || history.previous.size() < s.grid.nodes * width)
        throw std::invalid_argument("BDF2: invalid checkpoint history");
    s.ensure_history();
    std::vector<double> compact(s.rows * width);
    for (MeshIndex i = 0; i < s.rows; ++i) {
        const auto node = s.partition.host_owned ? s.partition.host_owned[i] : i;
        std::copy_n(history.previous.data() + node * width, width, compact.data() + i * width);
    }
#ifdef AERO_WITH_KOKKOS_GPU
    if (s.device >= 0 && s.rows) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        gpu::checked(
            gpu::copy(s.previous, compact.data(), compact.size() * sizeof(double), gpu::Copy::ToDevice));
    } else
#endif
        std::copy(compact.begin(), compact.end(), s.previous);
    s.previous_dt = history.dt;
    clear_prediction_history();
    s.cached = false; // Контрольная точка могла содержать другое поле.
}

// Смена реконструкции или режима LTS начинает новую последовательность проб.
void BdfStep::clear_prediction_history() {
    storage_->predictions.clear();
    invalidate_spatial_balance();
}

// Изменение поля или потоков снимает признак готовности начальной правой части.
void BdfStep::invalidate_spatial_balance() { storage_->saved_balance = false; }

// Вычисляет пространственные скорости изменения поля без временного вклада BDF.
const double* BdfStep::spatial_rates() const { return storage_->correction; }

// Максимум |R_i|/V_i по собственным узлам всех процессов и всем уравнениям.
double BdfStep::spatial_rate_norm(const double* volumes, const EulerFluxResult* flux, bool save_rows, bool save_balance) {
    auto& s = *storage_;
    s.saved_balance = false;
    const double norm = s.measure_spatial_balance(volumes,flux,save_rows ? s.correction : nullptr,
                                                  save_balance ? s.rhs : nullptr);
    s.saved_balance = save_balance;
    return norm;
}

#ifdef AERO_HAS_MULTIGRID
// Пробуем уменьшить невязку стационарного уравнения после обычного шага.
SteadyCorrection BdfStep::correct_steady(double* state,const double* initial,const double* volumes,
                                        const EulerFluxResult* flux,double dt,const NewtonSettings& settings,
                                        double other_rate,const std::function<double()>& trial_norm,
                                        const std::function<void()>& restore_flux) {
    auto& s = *storage_;
    // До сборки якобиана потоки соответствуют принятому полю. Сохраняем полную
    // газовую правую часть и получаем критерий принятия за тот же проход.
    invalidate_spatial_balance();
    const double accepted_norm = std::max(s.measure_spatial_balance(volumes,flux,nullptr,s.rhs),other_rate);
    if (accepted_norm == 0.) return {0.,0.,false};
    s.partition.local([&] {
        if (!s.coarse) s.coarse.reset(new multigrid::CoarseStep(s.host_grid,s.grid,s.rows,s.device,s.partition));
    });
    // Если Ньютон обошёлся предиктором, сборка может занять память потока.
    // Обновляем его ровно один раз при отказе даже в случае нулевой грубой правой части.
    bool pending_refresh = !s.matrix_on_step;
    if (pending_refresh) s.prepare_matrix(state,initial,volumes,dt,settings,0,{});
    // Пересчитывает зависимые потоки или коэффициенты после изменения поля.
    const auto refresh = [&] { restore_flux(); pending_refresh = false; };
    s.predictions.clear(); // Грубая коррекция также использует рабочую память линейной системы.
    const double fraction = s.coarse->correct(s.linear,state,volumes,dt,s.rhs,s.correction,
                                              accepted_norm,trial_norm,refresh,s.start);
    if (pending_refresh && fraction == 0.) refresh();
    return {accepted_norm,fraction,s.coarse->used_history()};
}
#endif

// На входе потоки уже вычислены при выборе dt. После каждой поправки refresh_fluxes
// пересчитывает их тем же пространственным оператором. При ошибке восстанавливаем U_n.
NewtonResult BdfStep::advance(double* state, const double* initial, const double* volumes,
                              const EulerFluxResult* flux, double dt, const NewtonSettings& settings,
                              const std::function<void()>& refresh_fluxes, const std::function<bool()>& try_prediction_fluxes,
                              const std::function<bool()>& confirm_prediction, const PhaseTiming& timing,
                              bool allow_extrapolation, double auto_pseudo_dt, const std::function<void(CoupledStart)>& coupled_start) {
    auto& s = *storage_;
#ifdef AERO_HAS_MULTIGRID
    s.matrix_on_step = false;
#endif
    s.partition.local([&] {
        if (settings.second_order) s.ensure_history();
        if ((s.rows && (!state || !volumes || (!flux && s.grid.faces))) || !(dt > 0.) || !std::isfinite(dt))
            throw std::invalid_argument("BDF: fields and a finite positive time step are required");
        if (!(settings.linear_relative > 0. && settings.linear_relative < 1. &&
              settings.newton_relative > 0. && settings.newton_relative < 1.) || !settings.max_iterations)
            throw std::invalid_argument("BDF: invalid solver tolerance or Newton iteration limit");
    });
    // Интерполяция по трём последним моментам времени. При равных шагах a=3/2, b=1/2.
    const auto weights = BdfWeights::for_step(settings.second_order, dt, s.previous_dt);
    if (!std::isfinite(weights.history)) throw std::domain_error("BDF: invalid time step ratio");
    const NodeSteps node_steps = SteadyNodeSteps::active(s.partition.steady_steps);
    // Формирует параметры выбранного обновления полей неявного шага.
    const auto operation = [&](NewtonUpdate::Operation op, double relaxation) {
        return NewtonUpdate{s.grid, flux, volumes, state, s.start, s.rhs, s.correction, s.previous, dt,
                            weights.leading, weights.history, op, s.partition.owned, relaxation, node_steps};
    };
    // Формирует пробное поле по заданной доле нелинейной поправки.
    const auto update_trial = [&](NewtonUpdate::Operation op, double relaxation) {
        s.partition.local([&] { s.update(operation(op,relaxation)); });
    };
    // Применяет выбранное обновление рабочих полей текущего этапа решателя.
    const auto update = [&](NewtonUpdate::Operation op) { update_trial(op, 1.); };
    // Пока поле ещё принятое, сохраняем U_n и собираем первую правую часть за один обход.
    // Все последующие пробы могут использовать эту копию для отката.
    const bool reuse_balance = s.saved_balance && coupled_start && !settings.second_order;
    s.saved_balance = false;
    double initial_norm = s.residual_norm(operation(reuse_balance ? NewtonUpdate::SavedInitialRhs : NewtonUpdate::InitialRhs, 1.));
    LinearTolerance tolerance;
    tolerance.relative = settings.linear_relative;
    tolerance.max_iterations = settings.linear_max_iterations;
    // Вычисляет согласованную норму невязки текущего пробного состояния.
    const auto residual_norm = [&] {
        return s.residual_norm(operation(NewtonUpdate::FormRhs, 1.));
    };
    // Обновляет соседние копии поля перед вычислением полной невязки.
    const auto share_state = [&] {
        if (s.partition.share_state) s.partition.share_state(state);
    };
    try {
        // Критерий сходимости всегда отсчитывается от принятого U_n, в том
        // числе если предиктор сразу дал хорошее начальное приближение.
        s.begin_matrix_step(settings);
        unsigned assemblies = 0, linear_iterations = 0;
        std::vector<unsigned> correction_iterations;
        double norm = initial_norm;
        if (!std::isfinite(norm)) throw std::domain_error("BDF: nonfinite nonlinear residual");
        bool spatial_predictor = false; // Предобусловливатель пробы пригоден и для первой поправки Ньютона.
        bool use_history = false;       // После принятия сохраним слой для следующего газового предиктора.
        // Один переключатель управляет всеми способами построения начального
        // приближения. Лимит Ньютона применяется ниже, после проверки пробы.
        if (settings.without_predictor || !try_prediction_fluxes) {
            s.predictions.clear();
            if (coupled_start) {
                // При раздельном вызове интегратора завершаем отложенную SA на U_n.
                coupled_start(CoupledStart::Solve);
                norm = initial_norm = residual_norm();
                if (!std::isfinite(norm)) throw std::domain_error("BDF: nonfinite coupled initial residual");
            }
        } else {
            // В уравнении непрерывности нет вязких потоков. Его исходная норма
            // ограничивает снизу полную норму и при уточнённой SA на старом газе.
            // Если этот масштаб достаточен, первую полную SA решаем уже на пробном газе.
            double reference = initial_norm;
            bool mass_reference = false;
            if (coupled_start) {
                const double mass=s.residual_norm(operation(NewtonUpdate::MassNorm,1.));
                mass_reference = mass > 0. && mass >= .05 * initial_norm;
                if (mass_reference) {
                    reference=mass;
                    coupled_start(CoupledStart::Predict);
                }
                else {
                    coupled_start(CoupledStart::Solve);
                    norm=initial_norm=reference=residual_norm();
                    if (!std::isfinite(norm)) throw std::domain_error("BDF: nonfinite coupled initial residual");
                }
            }
            // Для общего шага все узлы делят одну шкалу времени. Связанный перенос SA
            // и индивидуальные псевдошаги используют свой штатный предиктор.
            const bool local_time = s.partition.steady_steps && s.partition.steady_steps->local;
            bool local_rans = false;
#ifdef AERO_HAS_TURBULENCE
            local_rans = auto_pseudo_dt > 0. && local_time && coupled_start && confirm_prediction &&
                         !s.partition.host_aes.c_des;
#endif
            use_history = settings.prediction_order && allow_extrapolation &&
                          !confirm_prediction && !local_time;
            // У LTS параметр dt содержит долю узлового шага. Для интерполяции
            // последовательно принятых полей используем фактически пройденный минимум.
            // Полная невязка каждой пробы по-прежнему считается со всеми узловыми dt_i.
            const double history_dt = local_rans ? auto_pseudo_dt : dt;
            const unsigned history_order = local_rans ? 2u : (use_history ? settings.prediction_order : 0u);
            if (s.predicts_rates != local_rans) s.predictions.clear();
            s.predicts_rates = local_rans;
            if (!history_order) s.predictions.clear();
            else s.configure_prediction_capacity();
            if (history_order && initial_norm > 0.) {
                TimePrediction trial(s.partition.owned,s.grid.components(),state,volumes,node_steps,dt);
                trial.intervals = s.predictions.local_intervals(local_rans);
                s.predictions.prepare(history_dt, history_order, trial);
                if (local_rans) {
                    // В BDF1 U_{n+1}=U_n+dt_i*L(U_{n+1}). Переносим во времени
                    // пространственный баланс; свой текущий dt_i узел применяет один раз.
                    trial.rate=s.rhs;
                    trial.save_rate=s.predictions.append(history_dt);
                }
                if (trial.count || local_rans) {
                    s.partition.local([&] { s.extrapolate(trial); });
#ifdef AERO_WITH_KOKKOS_GPU
                    if (local_rans) s.predict_local_rows(operation(NewtonUpdate::Predict,1.),initial,
                                                         reference*settings.newton_relative);
#endif
                    share_state();
                    norm = try_prediction_fluxes() ? residual_norm() : HUGE_VAL;
                    if (!(norm < .8 * initial_norm)) {
                        update(NewtonUpdate::RestoreStart);
                        share_state();
                        refresh_fluxes();
                        norm = residual_norm();
                    }
                }
            }
            bool viscous = false;
#ifdef AERO_HAS_VISCOSITY
            viscous = s.partition.host_aes.dim != 0; // Один выбор для всех рангов, включая пустые.
#endif
            // В стационарном RANS пристенная диффузия может быть жёстче временной
            // диагонали. В упрощённом Ньютоне одна сборка обслуживает и пробные
            // подстановки, и последующее решение BiCGSTAB, если оно потребуется.
#ifdef AERO_HAS_TURBULENCE
            // При локальном псевдошаге dt_i уже учитывает пристенную диффузию и SA.
            // Сначала пробуем решить газовый баланс временной диагональю. Матрица
            // понадобится только если подтверждённое связанное поле не прошло допуск.
            spatial_predictor = initial_norm > 0. && s.partition.steady_steps && confirm_prediction &&
                                !local_rans && !s.partition.host_aes.c_des && settings.simplified_newton();
#endif
            if (spatial_predictor) assemblies += s.prepare_matrix(state,initial,volumes,dt/weights.leading,settings,0,timing);
            // Применяет пробное обновление предиктора перед проверкой нелинейного баланса.
            const auto predictor_update = [&](NewtonUpdate::Operation op, double relaxation) {
                if (spatial_predictor && op == NewtonUpdate::Predict) {
                    timing.measure(PhaseTiming::Prediction,[&] { s.linear.predict(s.rhs,s.correction); });
                    op = NewtonUpdate::PredictCorrection;
                }
                update_trial(op, relaxation);
            };
            // При BDF2 множитель 1/a временной диагонали уже уменьшает подстановку.
            const bool damp = use_history && !viscous && weights.history == 0.;
            // На жёстком локальном шаге матрица уже собрана. Одна подстановка
            // готовым предобусловливателем даёт начальное поле для её решения.
            const unsigned trials = spatial_predictor && local_time ? 1u : (local_rans ? 6u : (viscous ? 24u : 16u));
            norm = predict_initial_field(norm,reference,settings,predictor_update,share_state,residual_norm,refresh_fluxes,
                                         try_prediction_fluxes,viscous,use_history,damp,confirm_prediction,local_rans,trials);
            if (local_rans && !(norm <= reference*settings.newton_relative)) {
                // Если строгой оценки не хватило, Ньютон начинает с обычного
                // связанного состояния и использует исходный полный критерий.
                update(NewtonUpdate::RestoreStart);
                share_state();
                coupled_start(CoupledStart::Restore);
                norm=initial_norm=residual_norm();
                if (!std::isfinite(norm)) throw std::domain_error("BDF: nonfinite coupled initial residual");
                s.predictions.clear();
                // На жёстком шаге восстанавливаем также штатный пространственный
                // предиктор. Это сохраняет начальное приближение на случай остановки
                // Ньютона по заданному пользователем пределу числа поправок.
                spatial_predictor = settings.simplified_newton() && initial_norm>0.;
                if (spatial_predictor) {
                    assemblies+=s.prepare_matrix(state,initial,volumes,dt/weights.leading,settings,0,timing);
                    norm=predict_initial_field(norm,initial_norm,settings,predictor_update,share_state,residual_norm,
                                               refresh_fluxes,try_prediction_fluxes,viscous,false,false,confirm_prediction,false,1);
                }
            }
        }
        for (unsigned iteration = 0;; ++iteration) {
            // Равновесное поле с нулевой исходной невязкой принимаем без решения системы.
            const double relative = initial_norm == 0. ? 0. : norm / initial_norm;
            if (relative <= settings.newton_relative || iteration == settings.max_iterations) {
                // Оба условия завершают шаг с текущим полем и фактической невязкой.
                // Для следующего шага BDF2 сохраняем историю и при остановке по лимиту.
                // Экстраполяция самого поля использует общий шаг. Локальные
                // интервалы и балансы RANS записаны при построении его пробы.
                if (use_history) s.partition.local([&] { s.remember(dt); });
                if (settings.second_order) std::swap(s.previous, s.start);
                s.previous_dt = settings.second_order ? dt : 0.;
                if (s.cached && !s.grid.options.background.active()) ++s.accepted_since_build;
                return {iteration, relative, linear_iterations, std::move(correction_iterations), assemblies};
            }
            // Диагональ пространственной части берётся из баланса потоков. Временной
            // вклад a*V/dt добавляется при сборке системы для текущей поправки.
            if (!spatial_predictor || iteration != 0)
                assemblies += s.prepare_matrix(state,initial,volumes,dt/weights.leading,settings,iteration,timing);
            // Время BiCGSTAB включает обмены вектора, предобусловливание и
            // коллективные редукции всех скалярных произведений этой поправки.
            LinearResult result;
            s.predictions.clear(); // BiCGSTAB перезапишет векторы, содержащие принятые слои.
            timing.measure(PhaseTiming::Solve,[&] {
                result=s.linear.solve_with_rhs_norm(s.rhs,s.correction,tolerance,norm);
            });
            correction_iterations.push_back(result.iterations);
            linear_iterations += result.iterations;
            if (result.status != LinearStatus::Converged) {
                std::ostringstream message;
                message << "BiCGSTAB " << (result.status == LinearStatus::Breakdown ? "breakdown" : "iteration limit")
                        << " at Newton iteration " << iteration + 1 << ", linear relative residual = "
                        << result.residual_norm / norm;
                message << ", BiCGSTAB iterations: ";
                write_linear_iterations(message, correction_iterations, tolerance.max_iterations);
                throw LinearSolveFailure(message.str());
            }
            update(NewtonUpdate::AddCorrection);
            // Для нового поля снова вычисляем физические потоки. Только по ним
            // можно проверить сходимость полного нелинейного уравнения.
            share_state();
            refresh_fluxes();
            norm = residual_norm();
            if (!std::isfinite(norm)) throw std::domain_error("BDF: nonfinite nonlinear residual");
        }
    } catch (...) {
        s.cached = false;
        s.predictions.clear();
        update(NewtonUpdate::RestoreStart);
        if (s.partition.share_state) s.partition.share_state(state);
        throw;
    }
}
}
}
