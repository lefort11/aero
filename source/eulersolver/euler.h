#ifndef AERO_EULERSOLVER_EULER_H
#define AERO_EULERSOLVER_EULER_H

#include "../parallel/kernel_math.h"

namespace aero {
enum class EulerStatus {
    Success,
    InvalidDimension,
    NullPointer,
    InvalidGamma,
    InvalidDirection,
    InvalidDensity,
    InvalidPressure,
    NonFinite,
    InvalidRoeState
};

// Поток через единицу площади. За одну грань возвращаем массу, импульс и энергию.
struct EulerFluxResult {
    unsigned components;   // Число активных компонент равно размерности плюс два.
    double flux[5];        // Масса, компоненты импульса, полная энергия; остаток заполнен нулями.
    double max_wave_speed; // Самая быстрая волна вдоль нормали, нужна для CFL.
};

// Физические величины, восстановленные из консервативного состояния одного узла.
struct EulerState {
    double density;     // Масса единицы объёма.
    double velocity[3]; // Компоненты скорости вдоль координатных осей.
    double pressure;    // Термодинамическое давление идеального газа.
    double sound_speed; // Локальная скорость звука.
    double enthalpy;    // Полная энтальпия на единицу массы.
};
struct EulerRoeBasis {
    unsigned dimension;    // Число пространственных координат.
    double velocity[3];    // Скорость после осреднения по Роу.
    double enthalpy;       // Полная удельная энтальпия после осреднения.
    double sound_speed;    // Скорость звука в осреднённом состоянии.
    double right[25];      // Правые собственные векторы по столбцам, шаг строки равен пяти.
    double eigenvalues[5]; // Скорости волн в порядке: звук−, энтропия, сдвиговые волны, звук+.
};

namespace euler_detail {
AERO_HOST_DEVICE inline bool finite(double x) {
    return x == x && x <= max_finite_double && x >= -max_finite_double;
}
AERO_HOST_DEVICE inline double absolute(double x) {
    return x < 0 ? -x : x;
}
AERO_HOST_DEVICE inline double dot(unsigned d, const double* a, const double* b) {
    double value = 0;
    for (unsigned k = 0; k < d; ++k)
        value += a[k] * b[k];
    return value;
}
}

// Строит единичную нормаль. Площадь грани учитывается при сборке невязки.
AERO_HOST_DEVICE inline EulerStatus euler_unit_direction(unsigned dimension, const double* direction,
                                                         double* unit) {
    if (dimension < 1 || dimension > 3)
        return EulerStatus::InvalidDimension;
    if (!direction || !unit)
        return EulerStatus::NullPointer;
    // Сначала делим на наибольшую компоненту, чтобы квадрат длины не переполнился.
    double scale = 0, n[3] = {0, 0, 0};
    for (unsigned k = 0; k < dimension; ++k) {
        if (!euler_detail::finite(direction[k]))
            return EulerStatus::NonFinite;
        const double a = euler_detail::absolute(direction[k]);
        if (a > scale)
            scale = a;
    }
    if (!(scale > 0))
        return EulerStatus::InvalidDirection;
    for (unsigned k = 0; k < dimension; ++k)
        n[k] = direction[k] / scale;
    const double length = ::sqrt(euler_detail::dot(dimension, n, n));
    for (unsigned k = 0; k < dimension; ++k)
        unit[k] = n[k] / length;
    return EulerStatus::Success;
}

// Переводит вектор [ρ, ρu, E] в физические величины. Результат записывается после всех проверок.
AERO_HOST_DEVICE inline EulerStatus euler_state(unsigned dimension, const double* u, double gamma,
                                                EulerState* result) {
    if (dimension < 1 || dimension > 3)
        return EulerStatus::InvalidDimension;
    if (!u || !result)
        return EulerStatus::NullPointer;
    if (!euler_detail::finite(gamma) || !(gamma > 1))
        return EulerStatus::InvalidGamma;
    for (unsigned k = 0; k < dimension + 2; ++k)
        if (!euler_detail::finite(u[k]))
            return EulerStatus::NonFinite;
    if (!(u[0] > 0))
        return EulerStatus::InvalidDensity;
    EulerState state = {};
    state.density = u[0];
    double kinetic = 0;
    for (unsigned k = 0; k < dimension; ++k) {
        state.velocity[k] = u[k + 1] / u[0];
        kinetic += 0.5 * u[k + 1] * state.velocity[k];
        if (!euler_detail::finite(state.velocity[k]))
            return EulerStatus::NonFinite;
    }
    // Вычитаем кинетическую часть из полной энергии, чтобы получить внутреннюю.
    state.pressure = (gamma - 1) * (u[dimension + 1] - kinetic);
    if (!euler_detail::finite(state.pressure))
        return EulerStatus::NonFinite;
    if (!(state.pressure > 0))
        return EulerStatus::InvalidPressure;
    const double sound2 = gamma * (state.pressure / u[0]);
    state.sound_speed = ::sqrt(sound2);
    state.enthalpy = u[dimension + 1] / u[0] + state.pressure / u[0];
    if (!euler_detail::finite(state.sound_speed) || !euler_detail::finite(state.enthalpy))
        return EulerStatus::NonFinite;
    if (!(state.sound_speed > 0))
        return EulerStatus::InvalidPressure;
    *result = state;
    return EulerStatus::Success;
}

namespace euler_detail {
// Вычисляет один физический поток по готовым состоянию газа и единичной нормали.
AERO_HOST_DEVICE inline EulerStatus project_flux(unsigned dimension, const double* u, const EulerState& state,
                                                 const double* n, double* flux, double* wave_speed) {
    const double un = euler_detail::dot(dimension, state.velocity, n);
    // Массу переносит нормальный импульс. В потоки импульса и энергии входит также работа давления.
    double f[5] = {0, 0, 0, 0, 0};
    f[0] = euler_detail::dot(dimension, u + 1, n);
    for (unsigned k = 0; k < dimension; ++k)
        f[k + 1] = u[k + 1] * un + state.pressure * n[k];
    f[dimension + 1] = u[dimension + 1] * un + state.pressure * un;
    const double speed = euler_detail::absolute(un) + state.sound_speed;
    if (!euler_detail::finite(speed))
        return EulerStatus::NonFinite;
    for (unsigned k = 0; k < dimension + 2; ++k)
        if (!euler_detail::finite(f[k]))
            return EulerStatus::NonFinite;
    for (unsigned k = 0; k < dimension + 2; ++k)
        flux[k] = f[k];
    *wave_speed = speed;
    return EulerStatus::Success;
}
}

// Проецирует физический поток Эйлера на единичную нормаль и оценивает скорость |u·n| + c.
AERO_HOST_DEVICE inline EulerStatus euler_flux(unsigned dimension, const double* u, const double* direction,
                                               double gamma, double* flux, double* wave_speed) {
    if (!flux || !wave_speed)
        return EulerStatus::NullPointer;
    EulerState state;
    EulerStatus status = euler_state(dimension, u, gamma, &state);
    if (status != EulerStatus::Success)
        return status;
    double n[3] = {0, 0, 0};
    status = euler_unit_direction(dimension, direction, n);
    if (status != EulerStatus::Success)
        return status;
    return euler_detail::project_flux(dimension, u, state, n, flux, wave_speed);
}

namespace euler_detail {
// Общие данные двух сторон грани. Эти же плотности, скорости звука и нормаль нужны при осреднении Роу.
struct InterfaceGas {
    EulerState left, right; // Физические величины на двух сторонах.
    double normal[3];       // Единичная нормаль в направлении от левого узла к правому.
};

// Разбирает обе стороны до нормализации. Используется для готовых потоков и отдельного запроса базиса.
AERO_HOST_DEVICE inline EulerStatus read_interface_states(unsigned dimension, const double* left,
                                                          const double* right, const double* direction,
                                                          double gamma, EulerState& state_left,
                                                          EulerState& state_right, double* normal) {
    auto status = euler_state(dimension, left, gamma, &state_left);
    if (status != EulerStatus::Success)
        return status;
    status = euler_state(dimension, right, gamma, &state_right);
    if (status != EulerStatus::Success)
        return status;
    return euler_unit_direction(dimension, direction, normal);
}

// При реконструкции состояний вычисляем физические потоки и предел скорости волн.
// Завершаем проверку левой стороны перед правой, чтобы сообщать первую ошибку в порядке обхода.
AERO_HOST_DEVICE inline EulerStatus interface_fluxes(unsigned dimension, const double* left,
                                                     const double* right, const double* direction,
                                                     double gamma, double* flux_left, double* flux_right,
                                                     InterfaceGas& gas, double& max_speed) {
    auto status = euler_state(dimension, left, gamma, &gas.left);
    if (status != EulerStatus::Success)
        return status;
    status = euler_unit_direction(dimension, direction, gas.normal);
    if (status != EulerStatus::Success)
        return status;
    double speed_left, speed_right;
    status = project_flux(dimension, left, gas.left, gas.normal, flux_left, &speed_left);
    if (status != EulerStatus::Success)
        return status;
    status = euler_state(dimension, right, gamma, &gas.right);
    if (status != EulerStatus::Success)
        return status;
    status = project_flux(dimension, right, gas.right, gas.normal, flux_right, &speed_right);
    if (status != EulerStatus::Success)
        return status;
    max_speed = speed_left > speed_right ? speed_left : speed_right;
    return EulerStatus::Success;
}

// При реконструкции потоков они уже известны. Скорость диссипации определяем по состояниям.
AERO_HOST_DEVICE inline EulerStatus interface_speed(unsigned dimension, const double* left,
                                                    const double* right, const double* direction,
                                                    double gamma, InterfaceGas& gas, double& max_speed) {
    const auto status =
        read_interface_states(dimension, left, right, direction, gamma, gas.left, gas.right, gas.normal);
    if (status != EulerStatus::Success)
        return status;
    const double speed_left = absolute(dot(dimension, gas.left.velocity, gas.normal)) + gas.left.sound_speed;
    const double speed_right =
        absolute(dot(dimension, gas.right.velocity, gas.normal)) + gas.right.sound_speed;
    if (!finite(speed_left) || !finite(speed_right))
        return EulerStatus::NonFinite;
    max_speed = speed_left > speed_right ? speed_left : speed_right;
    return EulerStatus::Success;
}

// Вход численного потока на одном ребре. Порядок сторон задаёт направление положительного потока.
struct FaceStates {
    unsigned dimension;      // Размерность, определяет длину состояния dimension + 2.
    const double* left;      // Консервативные переменные с левой стороны.
    const double* right;     // Консервативные переменные с правой стороны.
    const double* direction; // Направление от левой стороны к правой до нормализации.
    double gamma;            // Показатель адиабаты.
};

// Общий путь для реконструкции состояний. Тип схемы известен компилятору и задаёт формулу диссипации.
template <class Scheme>
AERO_HOST_DEVICE inline EulerStatus flux_from_states(const FaceStates& face, EulerFluxResult* result) {
    if (!result)
        return EulerStatus::NullPointer;
    InterfaceGas gas;
    double left_flux[5], right_flux[5], speed;
    const auto status = interface_fluxes(face.dimension, face.left, face.right, face.direction, face.gamma,
                                         left_flux, right_flux, gas, speed);
    if (status != EulerStatus::Success)
        return status;
    return Scheme::combine(face, gas, left_flux, right_flux, speed, result);
}

// При восстановлении самих потоков их повторное вычисление не требуется. Оцениваем скорость по паре состояний.
template <class Scheme>
AERO_HOST_DEVICE inline EulerStatus
flux_from_reconstructed_fluxes(const FaceStates& face, const double* left_flux, const double* right_flux,
                               EulerFluxResult* result) {
    if (!result || !left_flux || !right_flux)
        return EulerStatus::NullPointer;
    InterfaceGas gas;
    double speed;
    const auto status =
        interface_speed(face.dimension, face.left, face.right, face.direction, face.gamma, gas, speed);
    if (status != EulerStatus::Success)
        return status;
    return Scheme::combine(face, gas, left_flux, right_flux, speed, result);
}

// Собирает матрицу из уже проверенных физических величин. Она используется и для потока, и для реконструкции.
AERO_HOST_DEVICE inline EulerStatus build_roe_basis(unsigned dimension, const EulerState& left,
                                                    const EulerState& right, const double* direction,
                                                    const double* n, double gamma, EulerRoeBasis* result) {
    // Веса Роу пропорциональны квадратным корням плотностей соседних состояний.
    const double root_left = ::sqrt(left.density), root_right = ::sqrt(right.density);
    const double a = root_left / (root_left + root_right), b = root_right / (root_left + root_right);
    EulerRoeBasis basis = {};
    basis.dimension = dimension;
    for (unsigned k = 0; k < dimension; ++k)
        basis.velocity[k] = a * left.velocity[k] + b * right.velocity[k];
    basis.enthalpy = a * left.enthalpy + b * right.enthalpy;
    const double kinetic = 0.5 * euler_detail::dot(dimension, basis.velocity, basis.velocity);
    const double sound2 = (gamma - 1) * (basis.enthalpy - kinetic);
    if (!euler_detail::finite(sound2) || !euler_detail::finite(basis.enthalpy))
        return EulerStatus::NonFinite;
    if (!(sound2 > 0))
        return EulerStatus::InvalidRoeState;
    basis.sound_speed = ::sqrt(sound2);
    const double un = euler_detail::dot(dimension, basis.velocity, n), c = basis.sound_speed;
    double tangent[2][3] = {{0, 0, 0}, {0, 0, 0}};
    if (dimension == 2) {
        tangent[0][0] = -n[1];
        tangent[0][1] = n[0];
    }
    if (dimension == 3) {
        unsigned axis = 0;
        // Сравниваем исходные компоненты, чтобы округление нормали на CPU и GPU не меняло выбор касательной.
        for (unsigned k = 1; k < 3; ++k)
            if (euler_detail::absolute(direction[k]) < euler_detail::absolute(direction[axis]))
                axis = k;
        for (unsigned k = 0; k < 3; ++k)
            tangent[0][k] = (k == axis ? 1.0 : 0.0) - n[axis] * n[k];
        const double length = ::sqrt(euler_detail::dot(3, tangent[0], tangent[0]));
        for (unsigned k = 0; k < 3; ++k)
            tangent[0][k] /= length;
        tangent[1][0] = n[1] * tangent[0][2] - n[2] * tangent[0][1];
        tangent[1][1] = n[2] * tangent[0][0] - n[0] * tangent[0][2];
        tangent[1][2] = n[0] * tangent[0][1] - n[1] * tangent[0][0];
    }
    // Крайние столбцы описывают акустические волны, между ними стоят энтропийная и сдвиговые.
    const unsigned energy = dimension + 1, plus = dimension + 1;
    basis.right[0] = 1;
    basis.right[1] = 1;
    basis.right[plus] = 1;
    for (unsigned k = 0; k < dimension; ++k) {
        basis.right[5 * (k + 1)] = basis.velocity[k] - c * n[k];
        basis.right[5 * (k + 1) + 1] = basis.velocity[k];
        basis.right[5 * (k + 1) + plus] = basis.velocity[k] + c * n[k];
    }
    basis.right[5 * energy] = basis.enthalpy - c * un;
    basis.right[5 * energy + 1] = kinetic;
    basis.right[5 * energy + plus] = basis.enthalpy + c * un;
    for (unsigned t = 0; t + 1 < dimension; ++t) {
        for (unsigned k = 0; k < dimension; ++k)
            basis.right[5 * (k + 1) + 2 + t] = tangent[t][k];
        basis.right[5 * energy + 2 + t] = euler_detail::dot(dimension, basis.velocity, tangent[t]);
    }
    for (unsigned k = 0; k < dimension + 2; ++k)
        basis.eigenvalues[k] = un;
    basis.eigenvalues[0] = un - c;
    basis.eigenvalues[plus] = un + c;
    for (unsigned k = 0; k < 25; ++k)
        if (!euler_detail::finite(basis.right[k]))
            return EulerStatus::NonFinite;
    for (unsigned k = 0; k < dimension + 2; ++k)
        if (!euler_detail::finite(basis.eigenvalues[k]))
            return EulerStatus::NonFinite;
    *result = basis;
    return EulerStatus::Success;
}
AERO_HOST_DEVICE inline EulerStatus roe_basis_with_normal(unsigned dimension, const double* original_left,
                                                          const double* original_right,
                                                          const double* direction, double gamma,
                                                          EulerRoeBasis* result, double* n) {
    if (!result)
        return EulerStatus::NullPointer;
    EulerState left, right;
    const auto status =
        read_interface_states(dimension, original_left, original_right, direction, gamma, left, right, n);
    if (status != EulerStatus::Success)
        return status;
    return build_roe_basis(dimension, left, right, direction, n, gamma, result);
}
} // namespace euler_detail
AERO_HOST_DEVICE inline EulerStatus euler_roe_basis(unsigned dimension, const double* original_left,
                                                    const double* original_right, const double* direction,
                                                    double gamma, EulerRoeBasis* result) {
    double normal[3];
    return euler_detail::roe_basis_with_normal(dimension, original_left, original_right, direction, gamma,
                                               result, normal);
}
} // namespace aero
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
#include <vector>
#include <stdexcept>
namespace aero {
const char* euler_status_message(EulerStatus status);
class EulerError : public std::runtime_error {
public:
    explicit EulerError(EulerStatus status)
        : std::runtime_error(euler_status_message(status)), status_(status) {}
    EulerStatus status() const { return status_; }

private:
    EulerStatus status_; // Исходный код ошибки вычислительного ядра.
};
namespace euler_detail {
// Обычные CPU-вызовы принимают векторы и бросают исключения; ядра CPU/CUDA работают с массивами.
using StateFluxKernel = EulerStatus (*)(unsigned, const double*, const double*, const double*, double,
                                        EulerFluxResult*);
using SuppliedFluxKernel = EulerStatus (*)(unsigned, const double*, const double*, const double*,
                                           const double*, const double*, double, EulerFluxResult*);
EulerFluxResult checked_state_flux(StateFluxKernel kernel, const std::vector<double>& left,
                                   const std::vector<double>& right, const std::vector<double>& direction,
                                   double gamma);
EulerFluxResult checked_supplied_flux(SuppliedFluxKernel kernel, const std::vector<double>& flux_left,
                                      const std::vector<double>& flux_right, const std::vector<double>& left,
                                      const std::vector<double>& right, const std::vector<double>& direction,
                                      double gamma);
}
} // namespace aero
#endif
#endif
