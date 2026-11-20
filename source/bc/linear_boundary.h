#ifndef AERO_LINEAR_BOUNDARY_H
#define AERO_LINEAR_BOUNDARY_H
#include "bc.h"
#include "../eulersolver/linear_euler.h"
namespace aero {
namespace bc_detail {
// На сверхзвуковом входе правило может назначить каждую компоненту независимо.
// В матрице производная константы или начального поля по текущему полю равна нулю.
AERO_HOST_DEVICE inline bool linear_inlet(unsigned dim, const SupersonicInflow& rule,
                                          const double* initial, bool homogeneous, double* q) {
    if (rule.use_initial && !homogeneous && !initial)
        return false;
    for (unsigned k = 0; k < dim + 2; ++k) {
        // В конфигурации давление идёт сразу за плотностью, в поле узла оно последнее.
        const unsigned slot = k == 0 ? 0 : (k == dim + 1 ? 1 : k + 1);
        if (rule.mask & (1u << slot))
            q[k] = homogeneous ? 0. : rule.primitive[slot];
        else if (rule.use_initial)
            q[k] = homogeneous ? 0. : initial[k];
    }
    return true;
}

// Заданная величина заменяет одну входящую акустическую характеристику.
// Касательная скорость, энтропия и исходящее сочетание u'_n + p'/(rho0*c0) сохраняются.
AERO_HOST_DEVICE inline bool linear_subsonic(const LinearFace& face, const BoundaryRule& rule,
                                             const double* initial, bool homogeneous, double* q) {
    if (rule.source == BoundarySource::Current)
        return true;
    const unsigned d = face.dim;
    const unsigned slot = rule.mode == BoundaryMode::Pressure ? d + 1 : 0;
    const bool velocity = rule.mode == BoundaryMode::NormalVelocity;
    double target = homogeneous ? 0. : rule.value;
    if (!homogeneous && rule.source == BoundarySource::Initial) {
        if (!initial) return false;
        target = velocity ? euler_detail::dot(d, initial + 1, face.normal) : initial[slot];
    }
    // Любое из трёх правил сводится к одной поправке давления. Из неё находим
    // согласованные изменения плотности и нормальной компоненты скорости.
    const double sound_squared = face.gas.sound * face.gas.sound;
    const double impedance = face.gas.rho * face.gas.sound;
    const double un = euler_detail::dot(d, q + 1, face.normal);
    const double dp = velocity ? impedance * (un - target)
                               : (target - q[slot]) * (slot == 0 ? sound_squared : 1.);
    q[0] += dp / sound_squared;
    q[d + 1] += dp;
    for (unsigned j = 0; j < d; ++j)
        q[j + 1] -= face.normal[j] * dp / impedance;
    return true;
}
}

// Возвращает поток на единицу площади. Для якобиана используем те же правила
// с нулевыми заданными данными, оставляя зависимость потока от внутреннего узла.
AERO_HOST_DEVICE inline bool linear_boundary_flux(const LinearFace& face, BoundaryConfig patch,
        const FreeBoundaryRules* table, const double* inside, const double* initial,
        bool homogeneous, EulerFluxResult& out) {
    const unsigned d = face.dim;
    if (patch.type() == BoundaryType::Wall) {
        // Неподвижная стенка исключает нормальное движение. Остаётся сила давления.
        for (unsigned k = 0; k < d + 2; ++k) out.flux[k] = 0.;
        for (unsigned j = 0; j < d; ++j) out.flux[j + 1] = inside[d + 1] * face.normal[j] / face.gas.rho;
    } else {
        const auto* rules = patch.rules(table);
        if (!rules) return false;
        double q[5];
        for (unsigned k = 0; k < d + 2; ++k) q[k] = inside[k];
        // Знак фоновой нормальной скорости и число Маха выбирают режим один раз
        // для данного направления. На сверхзвуковом выходе берём все значения изнутри.
        const auto regime = bc_detail::free_regime(face.transport, face.gas.sound);
        if (regime == bc_detail::FreeRegime::SuperIn) {
            if (!bc_detail::linear_inlet(d, rules->superin, initial, homogeneous, q)) return false;
        } else if (regime != bc_detail::FreeRegime::SuperOut) {
            const auto& rule = regime == bc_detail::FreeRegime::In ? rules->inflow : rules->outflow;
            if (!bc_detail::linear_subsonic(face, rule, initial, homogeneous, q)) return false;
        }
        face.physical(q, out.flux);
    }
    out.components = d + 2;
    out.max_wave_speed = face.speed();
    return true;
}
}
#endif
