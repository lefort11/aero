#include "bdf1_linear_system.h"
#include "block_jacobi.h"
#include "spatial_jacobian.h"
#include "../parallel/parallel_for.h"
#include "bicgstab_vectors.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#endif
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace aero {
namespace implicit {
struct BdfLinearSystem::Storage {
    SpatialJacobian matrix;    // J пространства плюс временная диагональ текущей сборки.
    BlockJacobi jacobi;          // Разложенные диагональные блоки этой же матрицы.
    const int device;           // -1 для CPU, иначе номер карты.
    const MeshIndex count;      // Полное число скалярных неизвестных.
    bool csgs = false;         // Два прохода Гаусса–Зейделя с теми же разложениями диагонали.
    bool ready = false;         // Система готова после успешной сборки и разложения.
    KrylovVectors work{};       // Шесть непересекающихся участков рабочего буфера.
    std::vector<double> host;   // Рабочая память CPU.
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;   // Рабочая память выбранной карты.
#endif
    template<class... GeometryArgs>
    explicit Storage(GeometryArgs&&... geometry)
        : matrix(std::forward<GeometryArgs>(geometry)...), jacobi(matrix.node_count(), matrix.dimension(), matrix.device()),
          device(matrix.device()), count(matrix.node_count() * (matrix.dimension()+2)) {
        if (count > host.max_size()/KrylovFieldCount) throw std::length_error("BiCGSTAB: too many unknowns");
        const auto size = std::max<MeshIndex>(1, KrylovFieldCount*count); // Пустой ранг тоже передаёт корректные адреса.
        double* buffer;
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            gpu::DeviceScope scope(device);
            gpu::checked(scope.status);
            arrays.device = device;
            buffer = arrays.allocate<double>(size);
        } else
#endif
        {
            host.resize(size);
            buffer = host.data();
        }
        work = {buffer, buffer+count, buffer+2*count, buffer+3*count, buffer+4*count, buffer+5*count};
    }

    void update(const KrylovUpdate& operation) const {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            device_work::check(device_work::aero_device_krylov_update(device, count, operation));
            return;
        }
#endif
        parallel_for(count, operation);
    }
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
    double dot(const double* x, const double* y) const { return reduce(x, y); }
    double norm(const double* x) const { return reduce(x, nullptr); }

    // Правое предобусловливание: сначала решаем локальные системы M*z=v,
    // затем умножаем z на полную матрицу. Сама поправка x остаётся физической.
    void apply_preconditioned(const double* v, double* image) {
        if (csgs) matrix.apply_csgs(jacobi, v, work.z);
        else matrix.partition().local([&] { jacobi.apply(v, work.z); });
        matrix.apply(work.z, image);
    }
};

BdfLinearSystem::BdfLinearSystem(const SpatialResidual& residual) : storage_(new Storage(residual)) {}
BdfLinearSystem::BdfLinearSystem(const residual_detail::Geometry& host, const residual_detail::Geometry& active,
                                   int device, const Partition& part) : storage_(new Storage(host, active, device, part)) {}
double BdfLinearSystem::infinity_norm(const double* values) const { return storage_->norm(values); }
BdfLinearSystem::~BdfLinearSystem() = default;

void BdfLinearSystem::assemble(const double* state, const double* initial, const double* volumes, double dt, const MatrixSettings& settings, bool cache_diagonal) {
    auto& s = *storage_;
    s.ready = false;
    s.matrix.partition().local([&] {
        if (cache_diagonal) s.matrix.reserve_diagonal();
        if ((!volumes && s.count) || !(dt > 0.) || !std::isfinite(dt))
            throw std::invalid_argument("BDF: volumes and a finite positive time step are required");
    });
    s.matrix.assemble_matrix(state, initial, volumes, dt, settings.simplified_jacobian, settings.roe);
    s.csgs = settings.csgs;
    s.matrix.partition().local([&] { s.matrix.prepare_jacobi(s.jacobi);
        if (s.csgs) s.matrix.prepare_csgs(); });
    s.ready = true;
}
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
void BdfLinearSystem::apply(const double* x, double* result) const {
    if (!storage_->ready) throw std::logic_error("BDF: assemble before applying");
    storage_->matrix.apply(x, result);
}

LinearResult BdfLinearSystem::solve(const double* b, double* x, const LinearTolerance& tol) {
    auto& s = *storage_;
    const auto& w = s.work;
    if (!s.ready) throw std::logic_error("BDF: assemble before solving");
    s.matrix.partition().local([&] {
        if (s.count && (!b || !x || b == x))
            throw std::invalid_argument("BiCGSTAB: separate input and output required");
    });
    if (!(tol.relative >= 0. && tol.relative < 1. && tol.absolute >= 0.) || !std::isfinite(tol.absolute))
        throw std::invalid_argument("BiCGSTAB: invalid tolerance");
    const double rhs_norm = s.norm(b);
    if (!std::isfinite(rhs_norm)) throw std::domain_error("BiCGSTAB: nonfinite right-hand side");
    const double threshold = std::max(tol.absolute, tol.relative * rhs_norm);
    // В итерациях масштабируем невязку, чтобы при малой правой части скалярные
    // произведения не обнулялись. Допуск и итоговый отчёт остаются в исходных единицах.
    const double scale = rhs_norm > 0. ? rhs_norm : 1.;
    const auto update = [&](KrylovStep step, double a, double omega, const double* image) {
        s.matrix.partition().local([&] { s.update({w, b, x, image, scale, a, omega, step}); });
    };
    update(KrylovStep::Begin, 0., 0., nullptr);
    if (rhs_norm <= threshold) return {LinearStatus::Converged, 0, rhs_norm};
    const auto actual_residual = [&]() {
        s.matrix.apply(x, w.t);
        update(KrylovStep::TrueResidual, 0., 0., nullptr);
        return s.norm(w.r);
    };
    unsigned iteration = 0;
    LinearStatus reason = LinearStatus::IterationLimit;
    bool first = true;
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
        s.apply_preconditioned(w.p, w.v);
        const double projection = s.dot(w.shadow, w.v);
        if (!std::isfinite(projection) || projection == 0.) break;
        alpha = rho/projection;
        if (!std::isfinite(alpha)) break;
        update(KrylovStep::Advance, alpha, 0., w.v);
        double estimate = s.norm(w.r);
        if (!std::isfinite(estimate)) break;
        if (estimate > threshold/scale) {
            // Вторая поправка выбирает omega, минимизирующую евклидову норму r−omega*t.
            s.apply_preconditioned(w.r, w.t);
            const double tt = s.dot(w.t, w.t);
            if (!std::isfinite(tt) || tt == 0.) break;
            omega = s.dot(w.t, w.r)/tt;
            if (!std::isfinite(omega) || omega == 0.) break;
            update(KrylovStep::Advance, omega, 0., w.t);
            estimate = s.norm(w.r);
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
