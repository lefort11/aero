#ifndef AERO_LINEAR_EULER_H
#define AERO_LINEAR_EULER_H
#include "euler.h"

namespace aero {
// Одно неподвижное во времени однородное фоновое состояние для всей сетки.
struct LinearBackground {
    double rho = 0.;          // Фоновая плотность; нуль оставляет режим Euler по умолчанию.
    double p = 0.;            // Фоновое давление.
    double velocity[3] = {};  // Фоновая скорость в координатах расчёта.
    double sound = 0.;        // sqrt(gamma*p/rho), вычисляется при чтении постановки.
    // Проверяет, задано ли фоновое состояние линейной системы.
    AERO_HOST_DEVICE bool active() const { return rho > 0.; }

    // Первая вариация полной энергии на единицу объёма в переменных (rho', u', p').
    AERO_HOST_DEVICE double energy(unsigned d, const double* q, double gamma) const {
        return q[d+1]/(gamma-1.) + .5*euler_detail::dot(d, velocity, velocity)*q[0]
                                  + rho*euler_detail::dot(d, velocity, q+1);
    }
    // Обе замены переменных линейны. Можно преобразовывать наклоны прямо в их буфере.
    AERO_HOST_DEVICE void to_conservative(unsigned d, double* q, double gamma) const {
        q[d+1] = energy(d, q, gamma);
        for (unsigned j=0; j<d; ++j) q[j+1] = rho*q[j+1] + velocity[j]*q[0];
    }
    // Преобразует консервативные пульсации в пульсации примитивных величин.
    AERO_HOST_DEVICE void to_primitive(unsigned d, double* q, double gamma) const {
        for (unsigned j=0; j<d; ++j) q[j+1] = (q[j+1] - velocity[j]*q[0])/rho;
        q[d+1] = (gamma-1.)*(q[d+1] - .5*euler_detail::dot(d,velocity,velocity)*q[0]
                                     - rho*euler_detail::dot(d,velocity,q+1));
    }
};

// Действие постоянной матрицы потока в единичной нормали. Поле хранится в порядке
// rho', компоненты u', p'. Фоновый поток здесь отсутствует, поэтому F(0) точно равен нулю.
struct LinearFace {
    // Стационарное состояние, относительно которого получен поток.
    const LinearBackground& gas;
    // Количество активных пространственных компонент.
    unsigned dim;
    // Направление нормали для проекции скорости.
    const double* normal;
    double transport; // Проекция фоновой скорости на внешнюю нормаль.
    // Подготавливает нормальную скорость и параметры фонового газа для линейного потока.
    AERO_HOST_DEVICE LinearFace(const LinearBackground& base, unsigned d, const double* n)
        : gas(base), dim(d), normal(n), transport(euler_detail::dot(d,base.velocity,n)) {}
    // Возвращает максимальную характеристическую скорость линейной системы в нормальном направлении.
    AERO_HOST_DEVICE double speed() const { return ::fabs(transport) + gas.sound; }
    // Вычисляет физический поток линейной системы по заданной нормали.
    AERO_HOST_DEVICE void physical(const double* q, double* flux) const {
        const double un = euler_detail::dot(dim,q+1,normal);
        flux[0] = transport*q[0] + gas.rho*un;
        for (unsigned j=0; j<dim; ++j)
            flux[j+1] = transport*q[j+1] + normal[j]*q[dim+1]/gas.rho;
        flux[dim+1] = transport*q[dim+1] + gas.rho*gas.sound*gas.sound*un;
    }
    // Две акустические компоненты выражены в единицах плотности.
    AERO_HOST_DEVICE void acoustic(const double* q, double& minus, double& plus) const {
        const double pressure = q[dim+1]/(gas.sound*gas.sound);
        const double velocity = gas.rho/gas.sound*euler_detail::dot(dim,normal,q+1);
        minus = .5*(pressure-velocity);
        plus = .5*(pressure+velocity);
    }
    // Добавляет акустическую характеристическую составляющую линейного потока.
    AERO_HOST_DEVICE void add_acoustic(double minus, double plus, double* q) const {
        q[0] += minus+plus;
        for (unsigned j=0; j<dim; ++j)
            q[j+1] += normal[j]*(gas.sound/gas.rho)*(plus-minus);
        q[dim+1] += gas.sound*gas.sound*(minus+plus);
    }
    // Центральная часть берётся из двух физических потоков. Для Роу транспортная
    // скорость общая у энтропийной и касательных компонент, поправки нужны только у акустических.
    AERO_HOST_DEVICE bool numerical(bool roe, double upwind, const double* left, const double* right,
                                    const double* fl, const double* fr, EulerFluxResult& out,double entropy_fix=0.) const {
        double jump[5], dissipation[5];
        for (unsigned k=0; k<dim+2; ++k) {
            jump[k] = right[k]-left[k];
            dissipation[k] = (roe ? ::fabs(transport) : speed())*jump[k];
        }
        if (roe) {
            double minus, plus;
            acoustic(jump, minus, plus);
            add_acoustic((euler_detail::acoustic_modulus(transport-gas.sound,entropy_fix*gas.sound)-::fabs(transport))*minus,
                         (euler_detail::acoustic_modulus(transport+gas.sound,entropy_fix*gas.sound)-::fabs(transport))*plus, dissipation);
        }
        out.components = dim+2;
        out.max_wave_speed = speed();
        for (unsigned k=0; k<dim+2; ++k) {
            out.flux[k] = .5*fl[k]+.5*fr[k]-(.5*upwind)*dissipation[k];
            if (!euler_detail::finite(out.flux[k])) return false;
        }
        return true;
    }
};
}
#endif
