#include "bdf1_linear_system.h"
#include "block_jacobi.h"
#include "spatial_jacobian.h"
#include "../parallel/parallel_for.h"
#include "bicgstab_vectors.h"
#include "workspace.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace aero {
namespace implicit {
// Матрица, предобусловливатель и общая рабочая память BiCGSTAB.
struct BdfLinearSystem::Storage {
    SpatialJacobian matrix;    // J пространства плюс временная диагональ текущей сборки.
    BlockJacobi jacobi;          // Разложенные диагональные блоки этой же матрицы.
    const int device;           // -1 для CPU, иначе номер карты.
    const MeshIndex count;      // Полное число скалярных неизвестных.
    bool csgs = false;         // Два прохода Гаусса–Зейделя с теми же разложениями диагонали.
    bool ready = false;         // Система готова после успешной сборки и разложения.
    KrylovVectors work{};       // Шесть непересекающихся участков рабочего буфера.
    std::vector<double> host;   // Рабочая память CPU.
    // Распределяет рабочие векторы между CPU и GPU.
    Workspace workspace;
    // Создаёт матрицу и диагональный предобусловливатель и размечает шесть рабочих векторов BiCGSTAB.
    template<class... GeometryArgs>
    explicit Storage(GeometryArgs&&... geometry)
        : matrix(std::forward<GeometryArgs>(geometry)..., true),
          jacobi(matrix.node_count(), matrix.dimension(), matrix.device(),
                 matrix.device() >= 0 && matrix.partition().krylov_scratch.count != 0),
          device(matrix.device()), count(matrix.node_count() * (matrix.dimension()+2)) {
        if (count > host.max_size()/KrylovFieldCount) throw std::length_error("BiCGSTAB: too many unknowns");
        auto scratch = matrix.partition().krylov_scratch;
        const auto remaining = KrylovFieldCount - scratch.count;
        // На пустом ранге оставляем один допустимый адрес для коллективных операций.
        const auto size = count ? remaining*count : MeshIndex(1);
        double* buffer = workspace.allocate(device, host, size);
        for (unsigned i=0; i<remaining; ++i) scratch.field[scratch.count+i] = buffer+i*count;
        work = {scratch.field[0], scratch.field[1], scratch.field[2],
                scratch.field[3], scratch.field[4], scratch.field[5]};
    }

    // Выполняет выбранную поэлементную операцию BiCGSTAB на CPU или GPU.
    void update(const KrylovUpdate& operation) const {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            device_work::check(device_work::aero_device_krylov_update(device, count, operation));
            return;
        }
#endif
        parallel_for(count, operation);
    }
    // Вычисляет локальное скалярное произведение или максимальную норму на выбранном исполнителе.
    double local_reduction(const double* x, const double* y) const {
        if (!count) return 0.; // Нейтральный вклад пустого ранга и в сумму, и в норму.
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            double scalar;
            device_work::check(device_work::aero_device_krylov_reduce(device, count, x, y, &scalar));
            return scalar;
        }
#endif
        host_parallel::ensure_runtime();
        return krylov_reduce(host_parallel::Execution(), count, x, y);
    }
    // Согласует локальные редукции и ошибки между процессами MPI.
    double reduce(const double* x, const double* y) const {
        const auto& part = matrix.partition();
        if (!part.reduce) return local_reduction(x, y);
        double value = 0.;
        std::exception_ptr failure;
        try {
            value = local_reduction(x, y);
        } catch (...) {
            failure = std::current_exception();
            value = HUGE_VAL;
        }
        // Для исправного поля достаточно одной редукции. Бесконечный вклад при
        // ошибке на любом ранге переводит все процессы в общую обработку сбоя.
        value = part.reduce(value, y == nullptr);
        if (!std::isfinite(value)) part.local([&] { if (failure) std::rethrow_exception(failure); });
        return value;
    }
    // Обновляет векторы BiCGSTAB и одновременно вычисляет глобальную норму невязки.
    double update_norm(const KrylovUpdate& step) const {
        const auto& part=matrix.partition();
        double norm=0.;
        part.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
            if (device>=0) { device_work::check(device_work::aero_device_krylov_update_norm(device,count,step,&norm)); return; }
#endif
            host_parallel::ensure_runtime();
            norm=krylov_update_norm(host_parallel::Execution(),count,step);
        });
        return part.reduce ? part.reduce(norm,true) : norm;
    }
    // Вычисляет коэффициент стабилизации BiCGSTAB из глобальных скалярных произведений.
    double omega_projection() const {
        const auto& part=matrix.partition();
        double pair[2]{};
        part.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
            if (device>=0) { device_work::check(device_work::aero_device_krylov_project(device,count,work.t,work.r,pair)); return; }
#endif
            host_parallel::ensure_runtime();
            krylov_project(host_parallel::Execution(),count,work.t,work.r,pair);
        });
        if (part.reduce) for (auto& value:pair) value=part.reduce(value,false);
        return pair[0]>0. && std::isfinite(pair[0]) ? pair[1]/pair[0] : 0.;
    }
    // Вычисляет глобальное скалярное произведение рабочих векторов.
    double dot(const double* x, const double* y) const { return reduce(x, y); }
    // Возвращает максимальный модуль компоненты вектора по всем процессам.
    double norm(const double* x) const { return reduce(x, nullptr); }

    // Правое предобусловливание: сначала решаем локальные системы M*z=v,
    // затем умножаем z на полную матрицу. Сама поправка x остаётся физической.
    void apply_preconditioned(const double* v, double* image, bool use_sweep_image) {
        if (csgs) {
            matrix.apply_csgs(jacobi, v, work.z, use_sweep_image ? image : nullptr);
            if (use_sweep_image) return;
        } else {
            matrix.partition().local([&] { jacobi.apply(v, work.z); });
        }
        matrix.apply(work.z, image);
    }
};

// Создаёт пространственную матрицу, предобусловливатель и рабочие векторы BiCGSTAB.
BdfLinearSystem::BdfLinearSystem(const SpatialResidual& residual) : storage_(new Storage(residual)) {}
// Создаёт пространственную матрицу, предобусловливатель и рабочие векторы BiCGSTAB.
BdfLinearSystem::BdfLinearSystem(const residual_detail::Geometry& host, const residual_detail::Geometry& active,
                                   int device, const Partition& part) : storage_(new Storage(host, active, device, part)) {}
// Возвращает максимальный модуль компоненты вектора по всем процессам.
double BdfLinearSystem::infinity_norm(const double* values) const { return storage_->norm(values); }
// Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
BdfLinearSystem::~BdfLinearSystem() = default;

// Собирает матрицу текущего шага и подготавливает выбранный предобусловливатель.
void BdfLinearSystem::assemble(const double* state, const double* initial, const double* volumes, double dt,
                               const MatrixSettings& settings, bool cache_diagonal, const PhaseTiming& timing) {
    auto& s = *storage_;
    s.ready = false;
    s.matrix.partition().local([&] {
        if (cache_diagonal) s.matrix.reserve_diagonal();
        if ((!volumes && s.count) || !(dt > 0.) || !std::isfinite(dt))
            throw std::invalid_argument("BDF: volumes and a finite positive time step are required");
    });
    timing.measure(PhaseTiming::Matrix,[&] {
        s.matrix.assemble_matrix(state,initial,volumes,dt,settings.simplified_jacobian,settings.roe,timing);
    });
    s.csgs = settings.csgs;
    timing.measure(PhaseTiming::PreconditionerSetup,[&] {
        s.matrix.partition().local([&] {
            s.matrix.prepare_jacobi(s.jacobi);
            if (s.csgs) s.matrix.prepare_csgs();
        });
    });
    s.ready = true;
}
// Обновляет временной вклад диагонали при повторном использовании пространственной матрицы.
void BdfLinearSystem::update_time_diagonal(const double* volumes, double dt) {
    auto& s = *storage_;
    if (!s.ready) throw std::logic_error("BDF: assemble before reusing the matrix");
    s.ready = false;
    s.matrix.partition().local([&] {
        s.matrix.update_time_diagonal(volumes, dt);
        s.matrix.prepare_jacobi(s.jacobi);
    });
    s.ready = true;
}
// Умножает готовую матрицу системы на переданный вектор.
void BdfLinearSystem::apply(const double* x, double* result) const {
    if (!storage_->ready) throw std::logic_error("BDF: assemble before applying");
    storage_->matrix.apply(x, result);
}

// Применяет предобусловливатель к правой части для начальной поправки.
void BdfLinearSystem::predict(const double* rhs,double* correction) const {
    const auto& s=*storage_;
    if (s.csgs) s.matrix.apply_csgs(s.jacobi,rhs,correction,nullptr);
    else s.matrix.partition().local([&]{s.jacobi.apply(rhs,correction);});
}

// Запускает BiCGSTAB с самостоятельным вычислением нормы правой части.
LinearResult BdfLinearSystem::solve(const double* b, double* x, const LinearTolerance& tol) {
    // Для самостоятельного вызова норма ещё неизвестна. Вычисляем её только
    // после общих проверок готовности матрицы и указателей.
    return solve_with_rhs_norm(b,x,tol,-1.);
}

// Решает систему BiCGSTAB, используя известную норму правой части для критерия останова.
LinearResult BdfLinearSystem::solve_with_rhs_norm(const double* b, double* x, const LinearTolerance& tol, double rhs_norm) {
    auto& s = *storage_;
    const auto& w = s.work;
    if (!s.ready) throw std::logic_error("BDF: assemble before solving");
    s.matrix.partition().local([&] {
        if (s.count && (!b || !x || b == x))
            throw std::invalid_argument("BiCGSTAB: separate input and output required");
    });
    if (!(tol.relative >= 0. && tol.relative < 1. && tol.absolute >= 0.) || !std::isfinite(tol.absolute))
        throw std::invalid_argument("BiCGSTAB: invalid tolerance");
    if (rhs_norm < 0.) rhs_norm = s.norm(b);
    if (!std::isfinite(rhs_norm)) throw std::domain_error("BiCGSTAB: nonfinite right-hand side");
    // Абсолютный допуск защищает почти нулевую правую часть, относительный масштабируется её нормой.
    const double threshold = std::max(tol.absolute, tol.relative * rhs_norm);
    // В итерациях масштабируем невязку, чтобы при малой правой части скалярные
    // произведения не обнулялись. Допуск и итоговый отчёт остаются в исходных единицах.
    const double scale = rhs_norm > 0. ? rhs_norm : 1.;
    // Применяет выбранное обновление рабочих полей текущего этапа решателя.
    const auto update = [&](KrylovStep step, double a, double omega, const double* image) {
        s.matrix.partition().local([&] { s.update({w, b, x, image, scale, a, omega, step}); });
    };
    update(KrylovStep::Begin, 0., 0., nullptr);
    if (rhs_norm <= threshold) return {LinearStatus::Converged, 0, rhs_norm};
    // Независимо вычисляет b−A*x, чтобы проверить рекуррентную оценку BiCGSTAB.
    const auto actual_residual = [&]() {
        s.matrix.apply(x, w.t);
        // Элемент b−A*x уже находится в регистре. В ту же редукцию включаем
        // его модуль, сохраняя саму невязку для возможного перезапуска BiCGSTAB.
        return s.update_norm({w, b, x, w.t, scale, 0., 0., KrylovStep::TrueResidual});
    };
    unsigned iteration = 0;
    LinearStatus reason = LinearStatus::IterationLimit;
    bool first = true, use_sweep_image = true;
    double old_rho = 1., alpha = 1., omega = 1.;
    while (iteration < tol.max_iterations) {
        ++iteration;
        reason = LinearStatus::Breakdown; // До завершения итерации выход возможен при вырождении.
        const double rho = s.dot(w.shadow, w.r);
        if (!std::isfinite(rho) || rho == 0.) break;
        if (!first) {
            const double beta = (rho/old_rho) * (alpha/omega);
            if (!std::isfinite(beta)) break;
            update(KrylovStep::Direction, beta, omega, nullptr);
        }
        // Первая поправка идёт вдоль M^{-1}p. После неё r хранит промежуточную
        // невязку s из формулы BiCGSTAB. При точном попадании второй шаг не нужен.
        s.apply_preconditioned(w.p, w.v, use_sweep_image);
        const double projection = s.dot(w.shadow, w.v);
        if (!std::isfinite(projection) || projection == 0.) break;
        alpha = rho/projection;
        if (!std::isfinite(alpha)) break;
        double estimate = s.update_norm({w,b,x,w.v,scale,alpha,0.,KrylovStep::Advance});
        if (!std::isfinite(estimate)) break;
        if (estimate > threshold/scale) {
            // Вторая поправка выбирает omega, минимизирующую евклидову норму r−omega*t.
            s.apply_preconditioned(w.r, w.t, use_sweep_image);
            omega = s.omega_projection();
            if (!std::isfinite(omega) || omega == 0.) break;
            estimate = s.update_norm({w,b,x,w.t,scale,omega,0.,KrylovStep::Advance});
            if (!std::isfinite(estimate)) break;
        }
        old_rho = rho;
        first = false;
        if (estimate <= threshold/scale) {
            // Округление накапливается в рекуррентной невязке. Принимаем решение
            // только после независимого умножения A*x. Иначе продолжаем с b−A*x.
            const double actual = actual_residual();
            if (actual <= threshold) return {LinearStatus::Converged, iteration, actual};
            if (!std::isfinite(actual)) break;
            // Если рекуррентная оценка оказалась оптимистичной, следующий цикл
            // использует прямое умножение CSR и для промежуточных направлений.
            use_sweep_image = false;
            update(KrylovStep::Restart, 0., 0., nullptr);
            first = true;
        }
        reason = LinearStatus::IterationLimit;
    }
    const double actual = actual_residual();
    return {actual <= threshold ? LinearStatus::Converged : reason, iteration, actual};
}
}
}
