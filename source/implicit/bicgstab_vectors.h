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
enum class KrylovStep { Begin, Restart, Direction, Advance, TrueResidual, BeginShifted, AddImage };
// Параметры поэлементного обновления рабочих векторов BiCGSTAB.
struct KrylovUpdate {
    // Набор рабочих векторов итерации BiCGSTAB.
    KrylovVectors work;
    const double* rhs;      // Правая часть системы в исходных единицах.
    double* x;              // Поправка в физических единицах.
    const double* image;    // v или t для соответствующей половины итерации.
    double scale, a, omega; // Масштаб правой части и коэффициенты текущего обновления.
    // Выбирает формулу обновления векторов.
    KrylovStep step;

    // Выполняет выбранную поэлементную операцию итерации BiCGSTAB.
    AERO_HOST_DEVICE void operator()(MeshIndex i) const {
        switch (step) {
        case KrylovStep::AddImage:
            x[i] += image[i]; // Прибавляет A·x₀ к временной правой части фильтрованного Ньютона.
            break;
        case KrylovStep::Begin:
        case KrylovStep::BeginShifted:
            if (step == KrylovStep::Begin) x[i] = 0.;
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
            accumulate_infinity_norm(x[i], largest);
        }, Kokkos::Max<double>(result));
    }
    return result;
}
// Для omega нужны <t,t> и <t,r>. Каждый поток читает t один раз; обе суммы
// возвращаются с устройства вместе. Норму остановки по-прежнему считаем в L_inf.
struct KrylovProjection {
    // Одновременно накапливаемые квадрат нормы t и скалярное произведение t с r.
    struct value_type { double square, mixed; };
    // Векторы проекции для коэффициентов стабилизации.
    const double *t, *r;
    // Обнуляет обе компоненты редукции скалярных произведений.
    AERO_HOST_DEVICE void init(value_type& v) const {v={0.,0.};}
    // Суммирует частичные результаты двух скалярных произведений.
    AERO_HOST_DEVICE void join(volatile value_type& a,const volatile value_type& b) const {
        a.square+=b.square; a.mixed+=b.mixed;
    }
    // Добавляет вклад одной компоненты в два скалярных произведения.
    AERO_HOST_DEVICE void operator()(MeshIndex i,value_type& v) const {
        v.square+=t[i]*t[i]; v.mixed+=t[i]*r[i];
    }
};
// Вычисляет пару скалярных произведений для коэффициента omega одним проходом.
template<class Execution>
inline void krylov_project(const Execution& exec,MeshIndex count,const double* t,const double* r,double* pair) {
    using Range=Kokkos::RangePolicy<Execution,Kokkos::IndexType<MeshIndex>>;
    KrylovProjection::value_type sums{};
    Kokkos::parallel_reduce("Aero: BiCGSTAB omega",Range(exec,0,count),KrylovProjection{t,r},sums);
    pair[0]=sums.square; pair[1]=sums.mixed;
}
// Совмещает обновление векторов Крылова с редукцией нормы.
template<class Execution>
inline double krylov_update_norm(const Execution& exec,MeshIndex count,KrylovUpdate step) {
    using Range=Kokkos::RangePolicy<Execution,Kokkos::IndexType<MeshIndex>>;
    double norm=0.;
    if (count) Kokkos::parallel_reduce("Aero: BiCGSTAB correction and norm",Range(exec,0,count),
        KOKKOS_LAMBDA(MeshIndex i,double& largest) {
            step(i);
            accumulate_infinity_norm(step.work.r[i], largest);
        },Kokkos::Max<double>(norm));
    return norm;
}
}
}
#endif
