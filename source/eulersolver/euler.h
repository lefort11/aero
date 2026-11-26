#ifndef AERO_EULERSOLVER_EULER_H
#define AERO_EULERSOLVER_EULER_H

#include "../parallel/fixed_matrix.h"
#include "../parallel/kernel_math.h"
#include <cmath>

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
    double density_root; // Вес состояния при осреднении Роу. Для узла используется всеми смежными рёбрами.
};
// Осреднённые параметры Роу для перехода к характеристическому базису.
struct EulerRoeBasis {
    unsigned dimension;    // Число пространственных координат.
    double velocity[3];    // Скорость после осреднения по Роу.
    double enthalpy;       // Полная удельная энтальпия после осреднения.
    double sound_speed;    // Скорость звука в осреднённом состоянии.
    double right[25];      // Правые собственные векторы по столбцам, шаг строки равен пяти.
    // В столбце энтропийной волны энергетическая компонента равна |v_Roe|²/2.
    AERO_HOST_DEVICE double kinetic_energy() const { return right[5 * (dimension + 1) + 1]; }
    // Проверяет или возвращает коэффициент диссипативной части потока.
    AERO_HOST_DEVICE double coefficient(unsigned row, unsigned wave) const { return right[5 * row + wave]; }
    double eigenvalues[5]; // Скорости волн в порядке: звук−, энтропия, сдвиговые волны, звук+.
};

namespace euler_detail {
// Проверяет конечность всех проверяемых компонент.
AERO_HOST_DEVICE inline bool finite(double x) {
    // У бесконечности и NaN все биты порядка равны единице. Проверка целых битов
    // освобождает блоки FP64 для вычисления самих потоков и сохраняет субнормальные числа.
    return (static_cast<unsigned>(high_word(x)) & 0x7ff00000u) != 0x7ff00000u;
}
// Возвращает абсолютное значение в коде, общем для CPU и GPU.
AERO_HOST_DEVICE inline double absolute(double x) {
    return x < 0 ? -x : x;
}
// Предел скорости распространения для потока ЛФ и его линеаризации.
AERO_HOST_DEVICE inline double signal_speed(double normal_velocity, const EulerState& gas) {
    return absolute(normal_velocity) + gas.sound_speed;
}
// Размер вектора известен при компиляции. На GPU обе координаты плоской задачи
// остаются в регистрах даже внутри составного ядра реконструкции.
template <unsigned dimension>
AERO_HOST_DEVICE inline double dot_components(const double* a, const double* b) {
    double value = 0;
    for (unsigned k = 0; k < dimension; ++k)
        value += a[k] * b[k];
    return value;
}
// Вычисляет скалярное произведение по активным пространственным компонентам.
AERO_HOST_DEVICE inline double dot(unsigned dimension, const double* a, const double* b) {
    if (dimension == 2)
        return dot_components<2>(a, b);
    if (dimension == 3)
        return dot_components<3>(a, b);
    return dot_components<1>(a, b);
}

// Нормирует направление в размерности, которую уже проверила вызывающая функция.
template <unsigned dimension>
AERO_HOST_DEVICE inline EulerStatus normalized_coordinates(const double* direction, double* unit) {
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
    const double length = ::sqrt(dot_components<dimension>(n, n));
    for (unsigned k = 0; k < dimension; ++k)
        unit[k] = n[k] / length;
    return EulerStatus::Success;
}
}

// Строит единичную нормаль для потока и граничных условий. Площадь учитывается при сборке невязки.
AERO_HOST_DEVICE inline EulerStatus euler_unit_direction(unsigned dimension, const double* direction,
                                                         double* unit) {
    if (dimension < 1 || dimension > 3)
        return EulerStatus::InvalidDimension;
    if (!direction || !unit)
        return EulerStatus::NullPointer;
    if (dimension == 2)
        return euler_detail::normalized_coordinates<2>(direction, unit);
    if (dimension == 3)
        return euler_detail::normalized_coordinates<3>(direction, unit);
    return euler_detail::normalized_coordinates<1>(direction, unit);
}

namespace euler_detail {
// При неположительном давлении плотность, скорость и само давление доступны ограничителю.
KOKKOS_FORCEINLINE_FUNCTION EulerStatus decode_state(unsigned dimension, const double* u, double gamma,
                                                      EulerState& state) {
    if (dimension < 1 || dimension > 3)
        return EulerStatus::InvalidDimension;
    if (!u)
        return EulerStatus::NullPointer;
    if (!euler_detail::finite(gamma) || !(gamma > 1))
        return EulerStatus::InvalidGamma;
    for (unsigned k = 0; k < dimension + 2; ++k)
        if (!euler_detail::finite(u[k]))
            return EulerStatus::NonFinite;
    if (!(u[0] > 0))
        return EulerStatus::InvalidDensity;
    state.density = u[0];
    state.density_root = ::sqrt(u[0]);
    // Для скорости, удельной энергии и давления знаменатель один. В обычном диапазоне
    // плотности вычисляем его обратное значение один раз. У пределов double оставляем
    // прямое деление, чтобы промежуточное 1/ρ не переполнилось и не стало субнормальным.
    const bool common_denominator = u[0] > 1.e-150 && u[0] < 1.e150;
    const double inverse_density = common_denominator ? inverse_value(u[0]) : 0;
    // Нормирует консервативную величину на плотность, получая значение на единицу массы.
    const auto per_mass = [&](double amount) {
        return common_denominator ? amount * inverse_density : amount / u[0];
    };
    double kinetic = 0;
    for (unsigned k = 0; k < dimension; ++k) {
        state.velocity[k] = per_mass(u[k + 1]);
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
    const double pressure_per_mass = per_mass(state.pressure);
    const double sound2 = gamma * pressure_per_mass;
    state.sound_speed = ::sqrt(sound2);
    state.enthalpy = per_mass(u[dimension + 1]) + pressure_per_mass;
    if (!euler_detail::finite(state.sound_speed) || !euler_detail::finite(state.enthalpy))
        return EulerStatus::NonFinite;
    if (!(state.sound_speed > 0))
        return EulerStatus::InvalidPressure;
    return EulerStatus::Success;
}
}

// Переводит вектор [ρ, ρu, E] в физические величины. Результат записывается после всех проверок.
KOKKOS_FORCEINLINE_FUNCTION EulerStatus euler_state(unsigned dimension, const double* u, double gamma,
                                                   EulerState* result) {
    if (!result)
        return dimension < 1 || dimension > 3 ? EulerStatus::InvalidDimension : EulerStatus::NullPointer;
    EulerState state = {};
    const auto status = euler_detail::decode_state(dimension, u, gamma, state);
    if (status == EulerStatus::Success)
        *result = state;
    return status;
}

namespace euler_detail {
// Вычисляет один физический поток по готовым состоянию газа и единичной нормали.
KOKKOS_FORCEINLINE_FUNCTION EulerStatus project_flux(unsigned dimension, const double* u, const EulerState& state,
                                                 const double* n, double* flux, double* wave_speed) {
    const double un = euler_detail::dot(dimension, state.velocity, n);
    // Массу переносит нормальный импульс. В потоки импульса и энергии входит также работа давления.
    double f[5] = {0, 0, 0, 0, 0};
    f[0] = euler_detail::dot(dimension, u + 1, n);
    for (unsigned k = 0; k < dimension; ++k)
        f[k + 1] = u[k + 1] * un + state.pressure * n[k];
    f[dimension + 1] = u[dimension + 1] * un + state.pressure * un;
    const double speed = signal_speed(un, state);
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

// Звуковая и переносная скорости на каждой стороне уже известны ограничителю и решателю.
AERO_HOST_DEVICE inline EulerStatus interface_wave_speed(unsigned, const InterfaceGas&, double&);

// При реконструкции потоков они уже известны. Скорость диссипации определяем по состояниям.
AERO_HOST_DEVICE inline EulerStatus interface_speed(unsigned dimension, const double* left,
                                                    const double* right, const double* direction,
                                                    double gamma, InterfaceGas& gas, double& max_speed) {
    const auto status =
        read_interface_states(dimension, left, right, direction, gamma, gas.left, gas.right, gas.normal);
    if (status != EulerStatus::Success)
        return status;
    return interface_wave_speed(dimension, gas, max_speed);
}

// Звуковая и переносная скорости на каждой стороне уже известны ограничителю и решателю.
AERO_HOST_DEVICE inline EulerStatus interface_wave_speed(unsigned dimension, const InterfaceGas& gas, double& max_speed) {
    const double speed_left = signal_speed(dot(dimension, gas.left.velocity, gas.normal), gas.left);
    const double speed_right = signal_speed(dot(dimension, gas.right.velocity, gas.normal), gas.right);
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
    double upw_coeff;        // Доля противопоточной поправки.
};

// Квадратическая энтропийная поправка Хартена к модулю акустической скорости.
AERO_HOST_DEVICE inline double acoustic_modulus(double lambda,double width) {
    const double a = ::fabs(lambda);
    return a < width ? .5*(width+lambda*(lambda/width)) : a;
}

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

// Состояние на грани после осреднения Роу. Его достаточно для потока во всех трёх размерностях.
struct RoeAverage {
    double velocity[3];   // Скорость газа в декартовых координатах.
    double enthalpy;      // Полная удельная энтальпия.
    double sound_speed;   // Скорость звука c.
    double kinetic;       // Кинетическая энергия единицы массы.
    double normal_speed;  // Скорость переноса u·n по нормали ребра.

    // Компоненты энтропийной волны в порядке: масса, импульс, энергия.
    AERO_HOST_DEVICE double entropy_row(unsigned row, unsigned dimension) const {
        return row == 0 ? 1 : (row == dimension + 1 ? kinetic : velocity[row - 1]);
    }
    // Для этих пределов все компоненты собственных векторов конечны при единичной нормали.
    AERO_HOST_DEVICE bool bounded_basis() const {
        return kinetic <= 1.e100 && absolute(enthalpy) <= 1.e100 &&
               sound_speed <= 1.e50 && absolute(normal_speed) <= 1.e51;
    }
    // Компоненты двух акустических векторов в строке плотности, импульса или энергии.
    AERO_HOST_DEVICE void acoustic_row(unsigned r, unsigned dimension, const double* n,
                                       double& minus, double& plus) const {
        if (r == 0) {
            minus = plus = 1;
            return;
        }
        const bool energy = r == dimension + 1;
        const double center = energy ? enthalpy : velocity[r - 1];
        const double shift = sound_speed * (energy ? normal_speed : n[r - 1]);
        minus = center - shift;
        plus = center + shift;
    }
};

// Вычисляет осреднённые по Роу скорость и энтальпию двух состояний.
KOKKOS_FORCEINLINE_FUNCTION EulerStatus average_roe_state(unsigned dimension, const EulerState& left,
                                                       const EulerState& right, const double* n,
                                                       double gamma, RoeAverage& gas) {
    // Веса Роу пропорциональны квадратным корням плотностей соседних состояний.
    const double root_left = left.density_root, root_right = right.density_root;
    // Плотности проверены, а их корни лежат далеко от переполнения и потери порядка.
    const double inverse_sum = inverse_value(root_left + root_right);
    const double a = root_left * inverse_sum, b = root_right * inverse_sum;
    for (unsigned k = 0; k < dimension; ++k)
        gas.velocity[k] = a * left.velocity[k] + b * right.velocity[k];
    gas.enthalpy = a * left.enthalpy + b * right.enthalpy;
    gas.kinetic = 0.5 * dot(dimension, gas.velocity, gas.velocity);
    const double sound2 = (gamma - 1) * (gas.enthalpy - gas.kinetic);
    if (!finite(sound2) || !finite(gas.enthalpy))
        return EulerStatus::NonFinite;
    if (!(sound2 > 0))
        return EulerStatus::InvalidRoeState;
    gas.sound_speed = ::sqrt(sound2);
    gas.normal_speed = dot(dimension, gas.velocity, n);
    return EulerStatus::Success;
}

// Поворот плоской единичной нормали на 90° задаёт направление сдвиговой волны.
AERO_HOST_DEVICE inline double plane_tangent(const double* normal, unsigned axis) {
    return axis == 0 ? -normal[1] : normal[0];
}

// Единичные касательные к грани. Этот же выбор осей задаёт поперечные характеристики Роу
// и локальные компоненты импульса в гибриде MP6. В 1D касательных нет.
AERO_HOST_DEVICE inline void face_tangents(unsigned dimension, const double* direction, const double* n,
                                           FixedMatrix<double,2,3>& tangent) {
    if (dimension == 2) {
        tangent(0, 0) = plane_tangent(n, 0);
        tangent(0, 1) = plane_tangent(n, 1);
    }
    if (dimension == 3) {
        unsigned axis = 0;
        // Сравниваем исходные компоненты, чтобы округление нормали на CPU и GPU не меняло выбор касательной.
        for (unsigned k = 1; k < 3; ++k)
            if (euler_detail::absolute(direction[k]) < euler_detail::absolute(direction[axis]))
                axis = k;
        for (unsigned k = 0; k < 3; ++k)
            tangent(0, k) = (k == axis ? 1.0 : 0.0) - n[axis] * n[k];
        const double length = ::sqrt(euler_detail::dot(3, tangent.row(0), tangent.row(0)));
        for (unsigned k = 0; k < 3; ++k)
            tangent(0, k) /= length;
        tangent(1, 0) = n[1] * tangent(0, 2) - n[2] * tangent(0, 1);
        tangent(1, 1) = n[2] * tangent(0, 0) - n[0] * tangent(0, 2);
        tangent(1, 2) = n[0] * tangent(0, 1) - n[1] * tangent(0, 0);
    }
}

// Базис полного двумерного шаблона. Вместо хранения всей матрицы вычисляем запрошенную
// компоненту волны из среднего газа и нормали. Формулы акустических столбцов общие с потоком Роу.
struct PlaneWaveBasis : RoeAverage {
    static constexpr unsigned dimension = 2; // Плоский базис содержит четыре характеристики.
    double normal[2];     // Направление от первого конца ребра ко второму.
    double tangent_speed; // Скалярное произведение средней скорости на касательную.

    // Вычисляет кинетическую часть энергии газового состояния.
    AERO_HOST_DEVICE double kinetic_energy() const { return kinetic; }
    // Проверяет или возвращает коэффициент диссипативной части потока.
    AERO_HOST_DEVICE double coefficient(unsigned row, unsigned wave) const {
        if (wave == 1)
            return entropy_row(row, dimension);
        if (wave == 2)
            return row == 0 ? 0 : (row == 3 ? tangent_speed : plane_tangent(normal, row - 1));
        double minus, plus;
        acoustic_row(row, dimension, normal, minus, plus);
        return wave == 0 ? minus : plus;
    }

    // Подготавливает один базис на обе стороны ребра. В крайнем диапазоне double
    // проверяем элементы и волновые скорости так же, как в полном матричном представлении.
    AERO_HOST_DEVICE bool prepare(const InterfaceGas& gas, double gamma) {
        normal[0] = gas.normal[0];
        normal[1] = gas.normal[1];
        if (average_roe_state(dimension, gas.left, gas.right, normal, gamma, *this) != EulerStatus::Success)
            return false;
        const double tangent[] = {plane_tangent(normal, 0), plane_tangent(normal, 1)};
        tangent_speed = dot_components<2>(velocity, tangent);
        if (bounded_basis())
            return true;
        for (unsigned row = 0; row < 4; ++row)
            for (unsigned wave = 0; wave < 4; ++wave)
                if (!finite(coefficient(row, wave)))
                    return false;
        return finite(normal_speed - sound_speed) && finite(normal_speed + sound_speed);
    }
};

// Для перехода между характеристиками и состоянием готовим матрицу из осреднённого газа.
// Если вызывающей стороне нужна только проверка, она передаёт пустой указатель.
template <bool full_basis = true, unsigned fixed_dimension = 0>
AERO_HOST_DEVICE inline EulerStatus build_roe_basis(unsigned mesh_dimension, const EulerState& left,
                                                    const EulerState& right, const double* direction,
                                                    const double* n, double gamma, EulerRoeBasis* result) {
    const unsigned dimension = fixed_dimension ? fixed_dimension : mesh_dimension;
    RoeAverage gas;
    const auto status = average_roe_state(dimension, left, right, n, gamma, gas);
    if (status != EulerStatus::Success)
        return status;
    // Для линейного EBR прямое и обратное преобразования сокращаются. Осталось
    // проверить, что сам базис существует. Нормаль единичная, а скорость ограничена
    // кинетической энергией. При этих границах даже H ± c·uₙ заведомо помещается в double.
    // За пределами этого диапазона проверяем каждый элемент обычным способом ниже.
    if (!result && gas.bounded_basis())
        return EulerStatus::Success;
    EulerRoeBasis basis = {};
    basis.dimension = dimension;
    basis.sound_speed = gas.sound_speed;
    basis.enthalpy = gas.enthalpy;
    for (unsigned k = 0; k < dimension; ++k)
        basis.velocity[k] = gas.velocity[k];
    FixedMatrix<double,2,3> tangent = {};
    if (full_basis)
        face_tangents(dimension, direction, n, tangent);
    // Крайние столбцы описывают акустические волны, между ними стоят энтропийная и сдвиговые.
    const unsigned energy = dimension + 1, plus = dimension + 1;
    for (unsigned row = 0; row <= energy; ++row) {
        gas.acoustic_row(row, dimension, n, basis.right[5 * row], basis.right[5 * row + plus]);
        basis.right[5 * row + 1] = gas.entropy_row(row, dimension);
    }
    for (unsigned t = 0; full_basis && t + 1 < dimension; ++t) {
        for (unsigned k = 0; k < dimension; ++k)
            basis.right[5 * (k + 1) + 2 + t] = tangent(t, k);
        basis.right[5 * energy + 2 + t] = euler_detail::dot(dimension, basis.velocity, tangent.row(t));
    }
    for (unsigned k = 0; k < dimension + 2; ++k)
        basis.eigenvalues[k] = gas.normal_speed;
    basis.eigenvalues[0] = gas.normal_speed - gas.sound_speed;
    basis.eigenvalues[plus] = gas.normal_speed + gas.sound_speed;
    for (unsigned k = 0; k < 25; ++k)
        if (!euler_detail::finite(basis.right[k]))
            return EulerStatus::NonFinite;
    for (unsigned k = 0; k < dimension + 2; ++k)
        if (!euler_detail::finite(basis.eigenvalues[k]))
            return EulerStatus::NonFinite;
    if (result)
        *result = basis;
    return EulerStatus::Success;
}
// Строит характеристический базис Роу для заданной единичной нормали.
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
// Проверяет состояния и формирует базис Роу на площадке.
AERO_HOST_DEVICE inline EulerStatus euler_roe_basis(unsigned dimension, const double* original_left,
                                                    const double* original_right, const double* direction,
                                                    double gamma, EulerRoeBasis* result) {
    double normal[3];
    return euler_detail::roe_basis_with_normal(dimension, original_left, original_right, direction, gamma,
                                               result, normal);
}
} // namespace aero
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY)
#include <vector>
#include <stdexcept>
namespace aero {
// Возвращает диагностический текст кода газодинамического ядра.
const char* euler_status_message(EulerStatus status);
// Исключение газодинамического ядра с исходным кодом ошибки.
class EulerError : public std::runtime_error {
public:
    // Сохраняет код газодинамической ошибки и формирует её текст.
    explicit EulerError(EulerStatus status)
        : std::runtime_error(euler_status_message(status)), status_(status) {}
    // Возвращает исходный код ошибки газодинамического ядра.
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
// Вычисляет поток по состояниям и сообщает ошибку через исключение.
EulerFluxResult checked_state_flux(StateFluxKernel kernel, const std::vector<double>& left,
                                   const std::vector<double>& right, const std::vector<double>& direction,
                                   double gamma);
// Вычисляет численный поток с заданными физическими потоками и проверяет статус.
EulerFluxResult checked_supplied_flux(SuppliedFluxKernel kernel, const std::vector<double>& flux_left,
                                      const std::vector<double>& flux_right, const std::vector<double>& left,
                                      const std::vector<double>& right, const std::vector<double>& direction,
                                      double gamma);
}
} // namespace aero
#endif
#endif
