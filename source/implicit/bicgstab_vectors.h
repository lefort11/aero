#ifndef AERO_BICGSTAB_VECTORS_H
#define AERO_BICGSTAB_VECTORS_H
#include "../parallel/kernel_math.h"
#include "../mesh/mesh_index.h"
#include <Kokkos_Core.hpp>

namespace aero {
namespace implicit {
// Рабочие массивы одного решения. Правый предобусловливатель переводит p (или r)
// в поправку z; умножение A*z использует уже собранные блоки матрицы.
struct KrylovVectors {
    double* r;        // Текущая невязка, в итерациях делится на ||b||∞.
    double* shadow;   // Фиксированная копия начальной невязки цикла BiCGSTAB.
    double* p;        // Направление поиска.
    double* v;        // Образ A*M^{-1}*p.
    double* z;        // Временный результат применения блочного Якоби.
    double* t;        // Образ A*M^{-1}*r, также буфер для проверки A*x.
};
enum class KrylovStep { Begin, Restart, Direction, Advance, TrueResidual };
struct KrylovUpdate {
    KrylovVectors work;
    const double* rhs;      // Правая часть системы в исходных единицах.
    double* x;              // Поправка в физических единицах.
    const double* image;    // v или t для соответствующей половины итерации.
    double scale, a, omega; // Масштаб правой части и коэффициенты текущего обновления.
    KrylovStep step;

    AERO_HOST_DEVICE void operator()(MeshIndex i) const {
        switch (step) {
        case KrylovStep::Begin:
            x[i] = 0.;
            work.r[i] = rhs[i];
            // В начале и после уточнения фактической невязки строим новый базис.
            // fall through
        case KrylovStep::Restart:
            work.r[i] /= scale;
            work.shadow[i] = work.p[i] = work.r[i];
            break;
        case KrylovStep::Direction:
            work.p[i] = work.r[i] + a * (work.p[i] - omega * work.v[i]);
            break;
        case KrylovStep::Advance:
            // Совмещаем обновление решения и невязки в одном проходе по узлам.
            x[i] += scale * (a * work.z[i]);
            work.r[i] -= a * image[i];
            break;
        case KrylovStep::TrueResidual:
            work.r[i] = rhs[i] - work.t[i];
            break;
        }
    }
};

// Скалярное произведение и максимум модуля используют один и тот же исполнитель,
// что и матрица. На CPU и GPU компилируются одни и те же формулы редукций.
template<class Execution>
inline double krylov_reduce(const Execution& exec, MeshIndex size, const double* x, const double* y) {
    using Range = Kokkos::RangePolicy<Execution, Kokkos::IndexType<MeshIndex>>;
    const auto range = Kokkos::Experimental::require(Range(exec, 0, size),
                                  Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    double result = 0.;
    if (y) {
        Kokkos::parallel_reduce("Aero: BiCGSTAB dot", range, KOKKOS_LAMBDA(MeshIndex i, double& sum) {
            sum += x[i]*y[i];
        }, result);
    } else {
        Kokkos::parallel_reduce("Aero: BiCGSTAB infinity norm", range, KOKKOS_LAMBDA(MeshIndex i, double& largest) {
            const double value = ::fabs(x[i]);
            // NaN не должен превратиться в нулевую норму при сравнении.
            if (!(value <= max_finite_double)) largest = HUGE_VAL;
            else if (value > largest) largest = value;
        }, Kokkos::Max<double>(result));
    }
    return result;
}
}
}
#endif
