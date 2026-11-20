#ifndef AERO_BOUNDARY_JACOBIAN_H
#define AERO_BOUNDARY_JACOBIAN_H
#include "edge_jacobian.h"
#include "../bc/bc.h"

namespace aero {
namespace implicit {
namespace boundary_detail {
// Один столбец производных примитивных переменных по состоянию узла.
struct Variation {
    double rho, p, velocity[3];
};

// Изменение граничного состояния при фиксированных начальных данных и заданном
// правиле. На пороге переключения дифференцируем ветвь, выбранную явным потоком.
AERO_HOST_DEVICE inline void apply_rule(unsigned d, bc_detail::FreeRegime regime, const FreeBoundaryRules& rules,
        const EulerState& inside, const EulerState& boundary, const double* normal, double gamma, Variation& v) {
    using bc_detail::FreeRegime;
    if (regime == FreeRegime::SuperOut)
        return;
    if (regime == FreeRegime::SuperIn) {
        // Заданные и начальные компоненты постоянны относительно текущего поля.
        const auto& in = rules.superin;
        if (in.use_initial || (in.mask & 1u)) v.rho = 0.;
        if (in.use_initial || (in.mask & 2u)) v.p = 0.;
        for (unsigned k = 0; k < d; ++k)
            if (in.use_initial || (in.mask & (4u << k))) v.velocity[k] = 0.;
        return;
    }
    const auto& rule = regime == FreeRegime::In ? rules.inflow : rules.outflow;
    if (rule.source == BoundarySource::Current)
        return;
    const double dr = v.rho / inside.density, dp = v.p / inside.pressure;
    const double dc = .5 * inside.sound_speed * (dp - dr);
    const double cb = ::sqrt(gamma * (boundary.pressure / boundary.density));
    const double dun = euler_detail::dot(d, v.velocity, normal);
    double normal_change;
    if (rule.mode == BoundaryMode::NormalVelocity) {
        // Заданная нормальная скорость постоянна. Инвариант J+ определяет c_b,
        // а изэнтропа даёт относительные изменения граничных плотности и давления.
        const double dcb = dc + .5 * (gamma - 1.) * dun;
        const double expansion = 2. / (gamma - 1.) * (dcb / cb - dc / inside.sound_speed);
        v.rho = boundary.density * (dr + expansion);
        v.p = boundary.pressure * (dp + gamma * expansion);
        normal_change = -dun;
    } else {
        // Фиксируем заданную величину, вторую находим из p/rho^gamma.
        const bool pressure_fixed = rule.mode == BoundaryMode::Pressure;
        v.rho = pressure_fixed ? boundary.density * (dr - dp / gamma) : 0.;
        v.p = pressure_fixed ? 0. : boundary.pressure * (dp - gamma * dr);
        const double dcb = .5 * cb * (v.p / boundary.pressure - v.rho / boundary.density);
        normal_change = 2. / (gamma - 1.) * (dc - dcb);
    }
    // Касательная скорость остаётся внутренней во всех дозвуковых режимах.
    for (unsigned k = 0; k < d; ++k)
        v.velocity[k] += normal_change * normal[k];
}

// Граничная площадка даёт один диагональный блок. Площадь уже содержится в area.
AERO_HOST_DEVICE inline bool block(unsigned d, BoundaryConfig patch, const FreeBoundaryRules* table,
        const double* current, const double* initial, const double* area, double gamma, double* matrix) {
    double normal[3];
    EulerState gas, boundary;
    if (euler_unit_direction(d, area, normal) != EulerStatus::Success ||
        euler_state(d, current, gamma, &gas) != EulerStatus::Success)
        return false;
    const double un = euler_detail::dot(d, gas.velocity, normal);
    const auto regime = bc_detail::free_regime(un, gas.sound_speed);
    const bool wall = patch.type() == BoundaryType::Wall;
    const auto* rules = patch.rules(table);
    if (!wall && (!rules || bc_detail::free_boundary_state(d, *rules, gas, normal, un, gamma, initial,
                                                            boundary) != BoundaryStatus::Success))
        return false;

    const unsigned width = d + 2;
    const double kinetic = .5 * euler_detail::dot(d, gas.velocity, gas.velocity);
    const double volume_flux = wall ? 0. : euler_detail::dot(d, boundary.velocity, area);
    const double boundary_kinetic = wall ? 0. : .5 * euler_detail::dot(d, boundary.velocity, boundary.velocity);
    for (unsigned col = 0; col < width; ++col) {
        Variation v = {};
        v.rho = col == 0 ? 1. : 0.;
        v.p = edge_detail::pressure_column(col, d + 1, gamma, kinetic, gas);
        if (wall) {
            // Через стенку передаётся только давление по нормали.
            matrix[col] = matrix[(d + 1) * width + col] = 0.;
            for (unsigned k = 0; k < d; ++k)
                matrix[(k + 1) * width + col] = area[k] * v.p;
        } else {
            for (unsigned k = 0; k < d; ++k)
                v.velocity[k] = edge_detail::velocity_column(col,k,gas);
            apply_rule(d, regime, *rules, gas, boundary, normal, gamma, v);
            // Правило произведения для расхода массы, импульса и потока полной энтальпии.
            const double dq = euler_detail::dot(d, v.velocity, area);
            const double dm = v.rho * volume_flux + boundary.density * dq;
            matrix[col] = dm;
            for (unsigned k = 0; k < d; ++k)
                matrix[(k + 1) * width + col] = dm * boundary.velocity[k] +
                    boundary.density * volume_flux * v.velocity[k] + v.p * area[k];
            const double h = gamma / (gamma - 1.) * boundary.pressure + boundary.density * boundary_kinetic;
            const double dh = gamma / (gamma - 1.) * v.p + v.rho * boundary_kinetic +
                boundary.density * euler_detail::dot(d, boundary.velocity, v.velocity);
            matrix[(d + 1) * width + col] = h * dq + dh * volume_flux;
        }
    }
    for (unsigned i = 0; i < width * width; ++i)
        if (!euler_detail::finite(matrix[i])) return false;
    return true;
}
} // namespace boundary_detail
} // namespace implicit
} // namespace aero
#endif
