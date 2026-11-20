#ifndef AERO_VISCOUS_TRANSPORT_H
#define AERO_VISCOUS_TRANSPORT_H
#include "grad.h"

namespace aero {
namespace viscosity {
struct Gas {
    double velocity[3], pressure_over_density, density, mu;
};
struct Material {
    double mu, mu_velocity[3]; // Средние по вершинам исходного элемента, формула (48).
    // Напряжения дают поток импульса, а их работа и теплопроводность — поток энергии.
    AERO_HOST_DEVICE void add_flux(unsigned dim, const Gradient& gradient, double* flux) const {
        flux[dim+1] += mu * gradient.heat;
        for (unsigned m = 0; m < dim; ++m) {
            flux[1+m] += mu * gradient.stress[m];
            flux[dim+1] += mu_velocity[m] * gradient.stress[m];
        }
    }
};
struct Row {
    double residual[5], cfl_rate;
};
struct Law {
    double gamma, temperature_scale, inverse_reynolds, prandtl, sutherland_ratio;
    // Масштабы: rho_inf = U_inf = 1. При T = T_inf получаем mu = 1/Re для обоих законов.
    AERO_HOST_DEVICE double dynamic_viscosity(double p_over_rho) const {
        const double theta = temperature_scale * p_over_rho;
        const double normal = ::sqrt(theta) * inverse_reynolds;
        return sutherland_ratio == 0. ? normal : normal * (theta / (theta + sutherland_ratio)) * (1. + sutherland_ratio);
    }
};
// Адреса принадлежат одному исполнителю. Те же три операции выполняются на CPU и в Kokkos на GPU.
struct Operator {
    unsigned dim;
    Law law;
    MeshIndex node_count, cell_count, row_count, vertex_count, neighbour_count;
    const Cell* cells;
    GradientStencil stencil; // Геометрические данные принадлежат тому же исполнителю, что и поле.
    const MeshIndex *offsets, *incidence;
    Gas* gas;
    Material* material;
    Row* rows;

    // При наличии готового газа используем давление и скорости, уже вычисленные для
    // конвективного потока. В экономичной раскладке на GPU этого буфера может не быть.
    AERO_HOST_DEVICE void prepare_gas(MeshIndex node, const double* state, const EulerState* cached) const {
        EulerState decoded{};
        if (!cached && euler_state(dim,state+node*(dim+2),law.gamma,&decoded) != EulerStatus::Success)
            decoded.density = 0.;
        const EulerState& primitive = cached ? cached[node] : decoded;
        auto& value = gas[node];
        if (!(primitive.density > 0.)) {
            value = Gas{};
            value.mu = HUGE_VAL;
            return;
        }
        value.density = primitive.density;
        value.pressure_over_density = primitive.pressure / primitive.density;
        value.mu = law.dynamic_viscosity(value.pressure_over_density);
        for (unsigned k = 0; k < dim; ++k) value.velocity[k] = primitive.velocity[k];
    }
    // Для работы напряжений в уравнении энергии усредняем именно произведение mu*u.
    AERO_HOST_DEVICE void average_material(MeshIndex index) const {
        const auto& cell = cells[index];
        Material average{};
        for (unsigned k = 0; k < cell.count; ++k) {
            const auto& point = gas[cell.node[k]];
            average.mu += point.mu;
            for (unsigned m = 0; m < dim; ++m) average.mu_velocity[m] += point.mu * point.velocity[m];
        }
        average.mu *= cell.inverse_count;
        for (unsigned m = 0; m < dim; ++m) average.mu_velocity[m] *= cell.inverse_count;
        material[index] = average;
    }
    // Каждую строку записывает только владелец узла. Номер вхождения хранит
    // номер элемента и местный номер его вершины (не больше семи).
    template<unsigned d>
    AERO_HOST_DEVICE void assemble_row(MeshIndex node) const {
        Row result{};
        for (MeshIndex k = offsets[node]; k < offsets[node+1]; ++k) {
            const auto entry = incidence[k];
            const auto e = entry / 8;
            const auto& average = material[e];
            const unsigned i = static_cast<unsigned>(entry % 8);
            const auto gradient = stencil.vertex_gradient<d>(cells[e], i, gas);
            average.add_flux(d, gradient, result.residual);
            result.cfl_rate += average.mu * cells[e].diffusion[i];
        }
        result.cfl_rate *= 2. / gas[node].density;
        rows[node] = result;
    }
    AERO_HOST_DEVICE void assemble(MeshIndex node) const {
        if (dim == 3) assemble_row<3>(node);
        else if (dim == 2) assemble_row<2>(node);
        else assemble_row<1>(node);
    }
};
}
}
#endif
