#ifndef AERO_FLOW_VALUES_H
#define AERO_FLOW_VALUES_H
#include "fields.h"
#include "../eulersolver/linear_euler.h"
#include <array>

namespace aero {
// Собирает столбцы отчёта из плотности, скоростей и давления. Для Linear энергия — пульсация.
inline std::array<double, 6> primitive_output_values(unsigned d, const double* q, double gamma,
                                                      const LinearBackground& background) {
    const double energy = background.active() ? background.energy(d, q, gamma)
        : q[d+1]/(gamma-1.) + .5*q[0]*euler_detail::dot(d, q+1, q+1);
    std::array<double, 6> values = {{q[0], 0., 0., 0., q[d+1], energy}};
    for (unsigned j = 1; j <= d; ++j) values[j] = q[j];
    return values;
}

// Один порядок колонок для локальных снимков, MPI-вывода и отчёта.
// Для Linear возвращаем пульсации, включая первую вариацию полной энергии.
inline std::array<double, 6> output_values(unsigned dimension, const double* u, double gamma, std::uint64_t node, const LinearBackground& background = {}) {
    if (background.active()) {
        const auto values = primitive_output_values(dimension, u, gamma, background);
        for (double value : values)
            if (!std::isfinite(value)) throw FieldError("output: nonfinite perturbation at node " + std::to_string(node));
        return values;
    }
    EulerState primitive;
    if (euler_state(dimension, u, gamma, &primitive) != EulerStatus::Success)
        throw FieldError("output: invalid Euler state at node " + std::to_string(node));
    return {{u[0], primitive.velocity[0], primitive.velocity[1], primitive.velocity[2], primitive.pressure,
             u[dimension + 1]}};
}
}
#endif
