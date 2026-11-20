#ifndef AERO_FLOW_VALUES_H
#define AERO_FLOW_VALUES_H
#include "fields.h"
#include "../eulersolver/linear_euler.h"
#include <array>

namespace aero {
// Один порядок колонок для локальных снимков, MPI-вывода и отчёта.
// Для Linear возвращаем пульсации, включая первую вариацию полной энергии.
inline std::array<double, 6> output_values(unsigned dimension, const double* u, double gamma, std::uint64_t node, const LinearBackground& background = {}) {
    if (background.active()) {
        std::array<double,6> values = {{u[0],0.,0.,0.,u[dimension+1],background.energy(dimension,u,gamma)}};
        for (unsigned j=0; j<dimension; ++j) values[j+1] = u[j+1];
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
