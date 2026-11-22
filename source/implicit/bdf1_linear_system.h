#ifndef AERO_BDF1_LINEAR_SYSTEM_H
#define AERO_BDF1_LINEAR_SYSTEM_H
#include <memory>
#include "partition.h"
#include "implicit_settings.h"
#include "phase_timing.h"

namespace aero {
class SpatialResidual;
namespace residual_detail { struct Geometry; }
namespace implicit {
enum class LinearStatus { Converged, IterationLimit, Breakdown };

struct LinearTolerance {
    double relative = 1.e-8;       // Доля от максимальной по модулю компоненты правой части.
    double absolute = 0.;          // Допуск в единицах невязки.
    unsigned max_iterations = default_solver_max_iter; // Полные итерации BiCGSTAB, включая итерации после перезапуска.
};
struct LinearResult {
    LinearStatus status;
    unsigned iterations;          // Ноль, если начальное приближение уже удовлетворяет допуску.
    double residual_norm;         // Фактическая норма ||b−A*x||∞ для возвращённого x.
};

// Линейная система одной итерации Ньютона для BDF. Хранит матрицу, предобуславливатель
// и рабочие векторы BiCGSTAB на том же исполнителе, что и пространственная невязка.
class BdfLinearSystem {
public:
    explicit BdfLinearSystem(const SpatialResidual& residual);
    BdfLinearSystem(const residual_detail::Geometry& host, const residual_detail::Geometry& active, int device,
                     const Partition& partition = Partition());
    ~BdfLinearSystem();
    // volumes содержит по одному положительному объёму на узел. initial —
    // фиксированное поле для граничных условий. Поля читаются во время сборки.
    void assemble(const double* state, const double* initial, const double* volumes, double dt,
                  const MatrixSettings& settings = MatrixSettings(), bool cache_diagonal = false,
                  const PhaseTiming& timing = {});
    // Для нового шага BDF обновляет временную диагональ и её LU-разложение в BJ/CSGS.
    void update_time_diagonal(const double* volumes, double dt);
    // Сокращённая форма для существующих вызовов с якобианом LF и блочным Якоби.
    void assemble(const double* state, const double* initial, const double* volumes, double dt, bool simplified) {
        MatrixSettings settings;
        settings.simplified_jacobian = simplified;
        assemble(state, initial, volumes, dt, settings);
    }
    // Вычисляет (J + V/dt)*x. Вход и выход расположены в разных массивах.
    void apply(const double* x, double* result) const;
    // Одно применение выбранного предобусловливателя для пробного поля газа.
    void predict(const double* rhs,double* correction) const;
    // Максимум модуля по всем уравнениям и узлам на том же устройстве, что и матрица.
    double infinity_norm(const double* values) const;
    // Решает A*x=b с нулевой начальной поправки. b и x — отдельные массивы.
    // При исчерпании итераций или вырождении возвращает достигнутую поправку и её невязку.
    LinearResult solve(const double* b, double* x, const LinearTolerance& tolerance = LinearTolerance());
private:
    friend class BdfStep;
    // Ньютон уже измерил глобальную норму этой правой части. Передаём её в
    // BiCGSTAB, чтобы не делать вторую редукцию того же вектора на GPU и в MPI.
    LinearResult solve_with_rhs_norm(const double* b, double* x, const LinearTolerance& tolerance, double rhs_norm);
    struct Storage;
    std::unique_ptr<Storage> storage_;
};
using Bdf1LinearSystem = BdfLinearSystem;
}
}
#endif
