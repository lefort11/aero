#include "scalar_topology.h"
#include "bdf1_linear_system.h"
#include "block_jacobi.h"
#include "spatial_jacobian.h"
#include "row_execution.h"
#include "krylov_operations.h"
#include "workspace.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace aero {
namespace implicit {
// Матрица, предобусловливатель и общая рабочая память BiCGSTAB.
struct BdfLinearSystem::Storage : KrylovOperations<BdfLinearSystem::Storage> {
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

    // Возвращает обмены и редукции данного блока для общей итерации.
    const Partition& partition() const { return matrix.partition(); }

    // Умножает газовую матрицу на проверяемый вектор.
    void apply(const double* x,double* y) const { matrix.apply(x,y); }
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
// Собственные векторы идут после заимствованных в порядке разметки конструктора.
KrylovScratch BdfLinearSystem::prediction_storage() const {
    const auto& s=*storage_;
    const auto& w=s.work;
    double* vectors[]={w.r,w.shadow,w.p,w.v,w.z,w.t};
    KrylovScratch free;
    for (unsigned i=s.matrix.partition().krylov_scratch.count;i<KrylovFieldCount;++i)
        free.field[free.count++]=vectors[i];
    return free;
}
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

// Векторы Крылова свободны до следующего линейного решения. После подстановки
// вызывающая сторона сразу потребляет z и заново вычисляет потоки пробного газа.
double* BdfLinearSystem::predict_in_workspace(const double* rhs) const {
    double* delta=storage_->work.z;
    predict(rhs,delta);
    return delta;
}

// Запускает BiCGSTAB с самостоятельным вычислением нормы правой части.
LinearResult BdfLinearSystem::solve(const double* b, double* x, const LinearTolerance& tol) {
    // Для самостоятельного вызова норма ещё неизвестна. Вычисляем её только
    // после общих проверок готовности матрицы и указателей.
    return solve_with_rhs_norm(b,x,tol,-1.);
}

// Решает систему BiCGSTAB, используя известную норму правой части для критерия останова.
LinearResult BdfLinearSystem::solve_with_rhs_norm(const double* b, double* x, const LinearTolerance& tol, double rhs_norm, double* shifted_rhs) {
    auto& s = *storage_;
    return solve_bicgstab(s,b,x,tol,rhs_norm,shifted_rhs);
}
// Переиспользует граф газовой матрицы для скалярной линейной системы.
ScalarTopology BdfLinearSystem::scalar_topology(bool colored) { return storage_->matrix.scalar_topology(colored); }

}
}
