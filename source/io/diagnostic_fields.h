#pragma once
#include "../eulersolver/linear_euler.h"
#include <array>
#include <string>
#include <vector>

namespace aero {
namespace diagnostic {
// Номера столбцов фиксированы. В файле остаются только запрошенные имена и их порядок.
enum Column {
    Density,
    Ux,
    Uy,
    Uz,
    Pressure,
    Energy,
    WorkingNu,
    EddyNu,
    DensityPrime,
    UxPrime,
    UyPrime,
    UzPrime,
    PressurePrime,
    EnergyPrime,
    DpDt,
    Vorticity,
    Cp,
    Schlieren,
    Mach,
    ColumnCount
};
// Возвращает имя диагностического столбца.
inline const char *name(unsigned i) {
    static const char *const names[] = {"rho",      "ux",      "uy",        "uz",        "p",        "E",
                                        "nu_tilde", "nut",     "rho_prime", "ux_prime",  "uy_prime", "uz_prime",
                                        "p_prime",  "E_prime", "dp_dt",     "vorticity", "Cp", "schlieren", "mach"};
    return names[i];
}
// Находит диагностический столбец по текстовому имени.
inline unsigned column(const std::string &key) {
    if (key == "dp/dt")
        return DpDt;
    for (unsigned i = 0; i < ColumnCount; ++i)
        if (key == name(i))
            return i;
    return ColumnCount;
}
// Проверяет принадлежность столбца к временной статистике.
inline bool time_column(unsigned c) { return c >= DensityPrime && c <= EnergyPrime; }
// Определяет, нужно ли накопление моментов для выбранного столбца.
inline bool needs_time(const std::string &key) { return key == "fluctuations" || time_column(column(key)); }
// Определяет, зависит ли столбец от активной турбулентной модели.
inline bool turbulent(const std::string &key) {
    return key == "turbulence" || key == "nu_tilde" || key == "nut";
}

// Один узел на границе двух принятых шагов. Храним взвешенное по времени среднее,
// предыдущее мгновенное значение для интегрирования по следующему шагу.
struct Moments {
    // Средние, последний отсчёт и длительность накопления.
    double mean[6], last[6], duration;
    // В прежнем recovery слово 12 содержало разность давлений. Оставляем место в файле.
    static constexpr unsigned words = 14;
    // Читает компоненту накопленных моментов по её номеру.
    AERO_HOST_DEVICE double get(unsigned k) const {
        return k < 6 ? mean[k] : (k < 12 ? last[k - 6] : (k == 12 ? 0. : duration));
    }
    // Записывает компоненту накопленных моментов по её номеру.
    AERO_HOST_DEVICE void set(unsigned k, double value) {
        if (k < 6)
            mean[k] = value;
        else if (k < 12)
            last[k - 6] = value;
        else if (k == 13)
            duration = value;
    }
};
// Во всех файлах и средних есть три места под скорость, включая нулевые
// компоненты на линии и на плоскости. Порядок столбцов совпадает с Column.
AERO_HOST_DEVICE inline void store_primary(double rho, unsigned dim, const double* velocity,
                                           double pressure, double energy, double* q) {
    q[Density] = rho;
    for (unsigned k = 0; k < 3; ++k) q[Ux + k] = k < dim ? velocity[k] : 0.;
    q[Pressure] = pressure;
    q[Energy] = energy;
}

// В Linear состояние уже хранит пульсации примитивных величин. В Euler сначала
// восстанавливаем давление и скорость. Полная энергия уже есть в самом состоянии.
AERO_HOST_DEVICE inline bool primary_from_state(unsigned dim, const double* u, double gamma,
                                                const LinearBackground& background, double* q) {
    if (background.active()) {
        store_primary(u[0], dim, u + 1, u[dim + 1], background.energy(dim, u, gamma), q);
        return true;
    }
    EulerState gas;
    if (euler_state(dim, u, gamma, &gas) != EulerStatus::Success) return false;
    store_primary(u[0], dim, gas.velocity, gas.pressure, u[dim + 1], q);
    return true;
}

// Ядро накопления временных моментов физических величин в узлах.
struct Sample {
    // Число компонент скорости при накоплении статистики.
    unsigned dim;
    // Показатель адиабаты и длительность интервала усреднения.
    double gamma, dt;
    // Фон для восстановления физических величин из пульсаций.
    LinearBackground background;
    // Обновляемая узловая статистика.
    Moments *values;
    // Консервативное поле текущего отсчёта.
    const double *state;
    // Добавляет взвешенный по времени вклад состояния одного узла в его моменты.
    AERO_HOST_DEVICE void operator()(size_t i) const {
        const auto *u = state + i * (dim + 2);
        double q[6];
        if (!primary_from_state(dim, u, gamma, background, q)) {
            values[i].duration = -1.;
            return;
        }
        auto &history = values[i];
        if (dt == 0.) {
            history.duration = 0.;
            for (unsigned k = 0; k < 6; ++k)
                history.mean[k] = history.last[k] = q[k];
            return;
        }
        history.duration += dt;
        const double weight = dt / history.duration;
        for (unsigned k = 0; k < 6; ++k) {
            // Интеграл по принятому шагу берём трапецией. Переменный dt входит в вес.
            const double midpoint = .5 * history.last[k] + .5 * q[k];
            history.mean[k] += weight * (midpoint - history.mean[k]);
            history.last[k] = q[k];
        }
    }
};
// Представление дополнительных полей SA, временных моментов и производной давления.
struct Fields {
    const double *sa = nullptr; // Пары nu_tilde и nut по расчётным узлам.
    // Узловая история для вывода усреднённых величин.
    const Moments *moments = nullptr;
    const double *pressure_rate = nullptr; // Мгновенная производная давления в неподвижном узле.
    // Данные вращения для вычисляемой диагностики.
    const double *rotation = nullptr;
    const double *schlieren = nullptr; // Модуль пространственного градиента плотности.
};
// Теорема Гаусса: вклад в ротор равен векторному произведению площади
// площадки на разность между скоростью на площадке и скоростью в узле.
struct Curl {
    // Компоненты ротора скорости.
    double value[3];
    // Дополняет набор вывода доступными диагностическими величинами.
    AERO_HOST_DEVICE void add(const double *area, const double *jump, double factor) {
        for (unsigned k = 0; k < 3; ++k) {
            const unsigned a = (k + 1) % 3, b = (k + 2) % 3;
            value[k] += factor * (area[a] * jump[b] - area[b] * jump[a]);
        }
    }
    // Вычисляет модуль векторной диагностической величины.
    AERO_HOST_DEVICE double magnitude(double inverse_volume) const {
        return ::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]) * inverse_volume;
    }
};
}
}
