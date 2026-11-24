#pragma once
#include "../parallel/modules.h"
#include "../eulersolver/linear_euler.h"
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/sa.h"
#endif

namespace aero {
// Выбирает множитель следующего шага по реально принятому изменению в этом узле.
struct AdaptiveStepChange {
    // Выбирает множитель роста шага по относительным изменениям газа и рабочей вязкости SA.
    static AERO_HOST_DEVICE double growth(double gas, double sa, double ceiling) {
        // Ориентиры на один псевдошаг: 1% для газа и 5% для рабочей вязкости SA.
        const double error = ::fmax(gas / .01, sa / .05);
        return ::fmin(ceiling, ::fmax(.5, 1. / ::sqrt(::fmax(error, 1.e-30))));
    }
    const double *old_field, *field; // Начало и конец последнего принятого шага.
    // Фон для измерения изменения линеаризованного решения.
    LinearBackground background;
    // Размерность сравниваемых полей.
    unsigned dim;
    double gamma, fraction, ceiling; // Доля фактического шага и разрешённый множитель роста.
#ifdef AERO_HAS_TURBULENCE
    const double *old_sa, *sa; // Рабочая вязкость до и после того же шага.
    // Узловые коэффициенты турбулентной вязкости для оценки изменения.
    const turbulence::View::Material* material;
#endif
    // Возвращает предложение для следующей попытки. Старый dt мог быть ограничен сроком вывода.
    AERO_HOST_DEVICE double next(size_t row, size_t node, double previous_dt) const {
        // Старый газ хранится в компактных строках Ньютона, новый — в нумерации узлов сетки.
        const double* before = old_field + row*(dim+2);
        const double* after = field + node*(dim+2);
        double rho_change, pressure_change, sound, du[3];
        if (background.active()) {
            rho_change = ::fabs(after[0]-before[0])/background.rho;
            pressure_change = ::fabs(after[dim+1]-before[dim+1])/background.p;
            sound = background.sound;
            for (unsigned k=0; k<dim; ++k) du[k] = after[k+1]-before[k+1];
        } else {
            EulerState a{}, b{};
            if (euler_state(dim,before,gamma,&a)!=EulerStatus::Success ||
                euler_state(dim,after,gamma,&b)!=EulerStatus::Success) return -1.;
            rho_change = ::fabs(b.density-a.density)/a.density;
            pressure_change = ::fabs(b.pressure-a.pressure)/a.pressure;
            sound = a.sound_speed;
            for (unsigned k=0; k<dim; ++k) du[k] = b.velocity[k]-a.velocity[k];
        }
        const double velocity_change = ::sqrt(euler_detail::dot(dim,du,du))/sound;
        double sa_change = 0.;
#ifdef AERO_HAS_TURBULENCE
        // На стенке nu_tilde равно нулю. Масштаб там задаёт молекулярная вязкость.
        if (sa) sa_change = ::fabs(sa[node]-old_sa[node])/(material[node].nu+::fabs(old_sa[node]));
#endif
        return previous_dt*fraction*growth(::fmax(velocity_change,::fmax(rho_change,pressure_change)),sa_change,ceiling);
    }
};

}
