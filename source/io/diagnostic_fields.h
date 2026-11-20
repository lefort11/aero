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
    ColumnCount
};
inline const char *name(unsigned i) {
    static const char *const names[] = {"rho",      "ux",      "uy",        "uz",        "p",        "E",
                                        "nu_tilde", "nut",     "rho_prime", "ux_prime",  "uy_prime", "uz_prime",
                                        "p_prime",  "E_prime", "dp_dt",     "vorticity", "Cp"};
    return names[i];
}
inline unsigned column(const std::string &key) {
    if (key == "dp/dt")
        return DpDt;
    for (unsigned i = 0; i < ColumnCount; ++i)
        if (key == name(i))
            return i;
    return ColumnCount;
}
inline bool time_column(unsigned c) { return c >= DensityPrime && c <= EnergyPrime; }
inline bool needs_time(const std::string &key) { return key == "fluctuations" || time_column(column(key)); }
inline bool turbulent(const std::string &key) {
    return key == "turbulence" || key == "nu_tilde" || key == "nut";
}

// Один узел на границе двух принятых шагов. Храним взвешенное по времени среднее,
// предыдущее мгновенное значение для интегрирования по следующему шагу.
struct Moments {
    double mean[6], last[6], duration;
    // В прежнем recovery слово 12 содержало разность давлений. Оставляем место в файле.
    static constexpr unsigned words = 14;
    AERO_HOST_DEVICE double get(unsigned k) const {
        return k < 6 ? mean[k] : (k < 12 ? last[k - 6] : (k == 12 ? 0. : duration));
    }
    AERO_HOST_DEVICE void set(unsigned k, double value) {
        if (k < 6)
            mean[k] = value;
        else if (k < 12)
            last[k - 6] = value;
        else if (k == 13)
            duration = value;
    }
};
struct Sample {
    unsigned dim;
    double gamma, dt;
    LinearBackground background;
    Moments *values;
    const double *state;
    AERO_HOST_DEVICE void operator()(size_t i) const {
        const auto *u = state + i * (dim + 2);
        double q[6] = {u[0], 0., 0., 0., u[dim + 1], u[dim + 1]};
        if (background.active()) {
            for (unsigned k = 0; k < dim; ++k)
                q[k + 1] = u[k + 1];
            q[Energy] = background.energy(dim, u, gamma);
        } else {
            EulerState gas;
            if (euler_state(dim, u, gamma, &gas) != EulerStatus::Success) {
                values[i].duration = -1.;
                return;
            }
            for (unsigned k = 0; k < dim; ++k)
                q[k + 1] = gas.velocity[k];
            q[Pressure] = gas.pressure;
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
struct Fields {
    const double *sa = nullptr; // Пары nu_tilde и nut по расчётным узлам.
    const Moments *moments = nullptr;
    const double *pressure_rate = nullptr; // Мгновенная производная давления в неподвижном узле.
    const double *rotation = nullptr;
};
// Теорема Гаусса: вклад в ротор равен векторному произведению площади
// площадки на разность между скоростью на площадке и скоростью в узле.
struct Curl {
    double value[3];
    AERO_HOST_DEVICE void add(const double *area, const double *jump, double factor) {
        for (unsigned k = 0; k < 3; ++k) {
            const unsigned a = (k + 1) % 3, b = (k + 2) % 3;
            value[k] += factor * (area[a] * jump[b] - area[b] * jump[a]);
        }
    }
    AERO_HOST_DEVICE double magnitude(double inverse_volume) const {
        return ::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]) * inverse_volume;
    }
};
}
}
