#include "bdf1_step.h"
#include "bdf1_linear_system.h"
#include "newton_vectors.h"
#include "../parallel/parallel_for.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#include "../parallel/device_api.h"
#endif
#include <cmath>
#include <sstream>

namespace aero {
namespace implicit {
namespace {
// Для малого шага матрицу первого приближения можно заменить временной диагональю.
// Каждая проба использует те же потоки, тот же обмен и ту же норму, что и Ньютон.
template<class Update, class Share, class Residual>
double predict_initial_field(double norm, const NewtonSettings& settings, const Update& update,
                             const Share& share, const Residual& residual,
                             const std::function<void()>& refresh, const std::function<bool()>& trial_flux,
                             bool viscous) {
    if (!trial_flux || norm == 0.) return norm;
    // При быстрой сходимости последовательные подстановки дешевле сборки AES
    // и решения линейной системы. Для вязкого шага оставляем запас по допуску,
    // чтобы точность поля была близка к получаемой после поправки Ньютона.
    const double target = norm * ((viscous ? .02 : 4.) * settings.newton_relative);
    // Хорошо сокращающаяся невязка допускает несколько дополнительных дешёвых
    // проб перед сборкой якобиана. При лимите в одну поправку сразу вызываем Ньютон.
    const unsigned limit = std::min(viscous ? 24u : 16u, settings.max_iterations - 1);
    for (unsigned k = 0; k < limit && norm > target; ++k) {
        update(NewtonUpdate::Predict, 1.);
        share();
        double trial = trial_flux() ? residual() : HUGE_VAL;
        if (viscous && std::isfinite(trial) && !(trial < .8 * norm)) {
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
        norm = trial;
    }
    return norm;
}
}

struct BdfStep::Storage {
    BdfLinearSystem linear;         // Матрица и рабочая память BiCGSTAB используются на всех шагах.
    residual_detail::Geometry grid; // Геометрия пространственной невязки на выбранном устройстве.
    Partition partition;            // Списки собственных строк и обмен обновлённым полем.
    MeshIndex rows;                 // Число узлов, за которые отвечает процесс.
    int device;                     // Отрицательный номер означает расчёт на CPU.
    std::vector<double> host;       // Три непересекающихся поля для расчёта на CPU.
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;       // Те же поля в памяти выбранной карты.
#endif
    double* start;       // Копия принятого поля в начале текущего шага.
    std::vector<double> history_host; // Выделяется только при первом использовании BDF2 на CPU.
    double* previous = nullptr;    // Предыдущий слой BDF2. После принятия шага меняется местами со start.
    double previous_dt = 0.;
    bool cached = false;            // В матрице сохранены производные для последующих шагов.
    unsigned accepted_since_build = 0;
    MatrixSettings cached_settings; // Переключение потока или предобуславливателя требует полной сборки.
    double* rhs;         // Полная невязка с обратным знаком для решения системы.
    double* correction;  // Последняя поправка BiCGSTAB к состоянию узлов.

    Storage(const residual_detail::Geometry& topology, const residual_detail::Geometry& active, int id, const Partition& part)
        : linear(topology, active, id, part), grid(active), partition(part),
          rows(part.extend_vector ? part.rows : grid.nodes), device(id) {
        if (rows > host.max_size() / (NewtonFieldCount * grid.components()))
            throw std::length_error("BDF: field too large");
        const auto size = rows * grid.components();
        const auto capacity = std::max<MeshIndex>(1, NewtonFieldCount*size);
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            gpu::DeviceScope scope(device);
            gpu::checked(scope.status);
            arrays.device = device;
            start = arrays.allocate<double>(capacity);
        } else
#endif
        {
            host.resize(capacity);
            start = host.data();
        }
        rhs = start + size;
        correction = rhs + size;
    }

    // Новая последовательность сборок начинается после изменения настроек или
    // по истечении заданного числа принятых шагов. Потоки для невязки всегда свежие.
    void begin_matrix_step(const NewtonSettings& settings) {
        cached = cached && (grid.options.background.active() || accepted_since_build < settings.jacobian_update_every) &&
                 settings.roe == cached_settings.roe &&
                 settings.simplified_jacobian == cached_settings.simplified_jacobian &&
                 settings.csgs == cached_settings.csgs;
    }

    // Возвращает единицу, если пришлось заново дифференцировать потоки.
    unsigned prepare_matrix(const double* state, const double* initial, const double* volumes,
                            double effective_dt, const NewtonSettings& settings, unsigned iteration) {
        if (!cached) {
            const bool reuse = grid.options.background.active() || settings.jacobian_update_every > 0;
            linear.assemble(state, initial, volumes, effective_dt, settings, reuse);
            cached = reuse;
            cached_settings = settings;
            accepted_since_build = 0;
            return 1;
        }
        // Для первого решения на новом шаге обновляем V*a/dt и разложение
        // диагонали. Остальные поправки используют ту же готовую матрицу.
        if (iteration == 0) linear.update_time_diagonal(volumes, effective_dt);
        return 0;
    }

    void ensure_history() {
        if (previous) return;
        const auto size = std::max<MeshIndex>(1, rows * grid.components());
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            gpu::DeviceScope scope(device);
            gpu::checked(scope.status);
            previous = arrays.allocate<double>(size);
            return;
        }
#endif
        history_host.resize(size);
        previous = history_host.data();
    }

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

BdfStep::BdfStep(const residual_detail::Geometry& host, const residual_detail::Geometry& active, int device, const Partition& part)
    : storage_(new Storage(host, active, device, part)) {}
BdfStep::~BdfStep() = default;

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
    s.cached = false; // Контрольная точка могла содержать другое поле.
}

double BdfStep::spatial_rate_norm(const double* volumes, const EulerFluxResult* flux) {
    auto& s = *storage_;
    s.partition.local([&] {
        s.update({s.grid, flux, volumes, nullptr, nullptr, s.rhs, nullptr, nullptr,
                  0., 0., 0., NewtonUpdate::SpatialRate, s.partition.owned, 1.});
    });
    const double norm = s.linear.infinity_norm(s.rhs);
    if (!std::isfinite(norm)) throw std::domain_error("Steady: nonfinite spatial residual");
    return norm;
}

NewtonResult BdfStep::advance(double* state, const double* initial, const double* volumes,
                              const EulerFluxResult* flux, double dt, const NewtonSettings& settings,
                              const std::function<void()>& refresh_fluxes, const std::function<bool()>& try_prediction_fluxes) {
    auto& s = *storage_;
    s.partition.local([&] {
        if (settings.second_order) s.ensure_history();
        if ((s.rows && (!state || !volumes || (!flux && s.grid.faces))) || !(dt > 0.) || !std::isfinite(dt))
            throw std::invalid_argument("BDF: fields and a finite positive time step are required");
        if (!(settings.linear_relative > 0. && settings.linear_relative < 1. &&
              settings.newton_relative > 0. && settings.newton_relative < 1.) || !settings.max_iterations)
            throw std::invalid_argument("BDF: invalid solver tolerance or Newton iteration limit");
    });
    // Интерполяция по трём последним моментам времени. При равных шагах a=3/2, b=1/2.
    const double ratio = settings.second_order && s.previous_dt > 0. ? dt/s.previous_dt : 0.;
    const double fraction = ratio/(1.+ratio);
    const double leading = 1.+fraction, history_weight = ratio*fraction;
    if (!std::isfinite(history_weight)) throw std::domain_error("BDF: invalid time step ratio");
    const auto update_trial = [&](NewtonUpdate::Operation op, double relaxation) {
        s.partition.local([&] {
            s.update({s.grid, flux, volumes, state, s.start, s.rhs, s.correction, s.previous, dt, leading, history_weight, op, s.partition.owned, relaxation});
        });
    };
    const auto update = [&](NewtonUpdate::Operation op) { update_trial(op, 1.); };
    // Первое приближение — принятое поле U_n. Его копия нужна в уравнении
    // баланса на каждой итерации и для отката всего шага при отказе решателя.
    update(NewtonUpdate::SaveStart);
    LinearTolerance tolerance;
    tolerance.relative = settings.linear_relative;
    const auto residual_norm = [&] {
        update(NewtonUpdate::FormRhs);
        return s.linear.infinity_norm(s.rhs);
    };
    const auto share_state = [&] {
        if (s.partition.share_state) s.partition.share_state(state);
    };
    try {
        // Критерий сходимости всегда отсчитывается от принятого U_n, в том
        // числе если предиктор сразу дал хорошее начальное приближение.
        s.begin_matrix_step(settings);
        unsigned assemblies = 0;
        const double initial_norm = residual_norm();
        double norm = initial_norm;
        if (!std::isfinite(norm)) throw std::domain_error("BDF: nonfinite nonlinear residual");
        bool viscous = false;
#ifdef AERO_HAS_VISCOSITY
        viscous = s.partition.host_aes.dim != 0; // Один выбор для всех рангов, включая пустые.
#endif
        norm = predict_initial_field(norm, settings, update_trial, share_state, residual_norm, refresh_fluxes,
                                     try_prediction_fluxes, viscous);
        for (unsigned iteration = 0;; ++iteration) {
            // Равновесное поле с нулевой исходной невязкой принимаем без решения системы.
            const double relative = initial_norm == 0. ? 0. : norm / initial_norm;
            if (relative <= settings.newton_relative) {
                // История продвигается только после успешного шага, включая равновесное поле.
                if (settings.second_order) std::swap(s.previous, s.start);
                s.previous_dt = settings.second_order ? dt : 0.;
                if (s.cached && !s.grid.options.background.active()) ++s.accepted_since_build;
                return {iteration, relative, assemblies};
            }
            if (iteration == settings.max_iterations) {
                std::ostringstream message;
                message << "Newton iteration limit (" << iteration << "), relative residual = " << relative;
                throw std::runtime_error(message.str());
            }
            // Диагональ пространственной части берётся из баланса потоков. Временной
            // вклад a*V/dt добавляется при сборке системы для текущей поправки.
            assemblies += s.prepare_matrix(state, initial, volumes, dt / leading, settings, iteration);
            const auto result = s.linear.solve(s.rhs, s.correction, tolerance);
            if (result.status != LinearStatus::Converged) {
                std::ostringstream message;
                message << "BiCGSTAB " << (result.status == LinearStatus::Breakdown ? "breakdown" : "iteration limit")
                        << " at Newton iteration " << iteration + 1 << ", linear relative residual = "
                        << result.residual_norm / norm;
                throw std::runtime_error(message.str());
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
        update(NewtonUpdate::RestoreStart);
        if (s.partition.share_state) s.partition.share_state(state);
        throw;
    }
}
}
}
