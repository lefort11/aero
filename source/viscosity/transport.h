#ifndef AERO_VISCOUS_TRANSPORT_H
#define AERO_VISCOUS_TRANSPORT_H
#include "grad.h"
#include "../parallel/modules.h"
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/sa.h"
#endif

namespace aero {
namespace viscosity {
// Внутри шага BDF скаляр SA уточняется при фиксированном газе. Этап указывает,
// какие данные устарели и нужно ли уже собирать газовую невязку.
enum class Evaluation { Full, BeforeSaIterations, AfterSaUpdate, RowsOnly };
struct Gas {
    double velocity[3], pressure_over_density, density, mu;
};
struct Material {
    double mu, mu_velocity[3]; // Средние по вершинам исходного элемента, формула (48).
#ifdef AERO_HAS_TURBULENCE
    double extra_heat; // Разность коэффициентов тепла и импульса при разных Pr и Pr_t.
#endif
    // Напряжения дают поток импульса, а их работа и теплопроводность — поток энергии.
    AERO_HOST_DEVICE void add_flux(unsigned dim, const Gradient& gradient, double* flux) const {
        double heat = mu;
#ifdef AERO_HAS_TURBULENCE
        heat += extra_heat;
#endif
        flux[dim+1] += heat * gradient.heat;
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
#ifdef AERO_HAS_TURBULENCE
    turbulence::View sa;
#endif
    MeshIndex node_count, cell_count, row_count, vertex_count, neighbour_count;
    const Cell* cells;
    GradientStencil stencil; // Геометрические данные принадлежат тому же исполнителю, что и поле.
    const MeshIndex *offsets, *incidence;
    Gas* gas;
    Material* material;
    Row* rows;

#ifdef AERO_HAS_TURBULENCE
    // На вершину приходится несколько элементов. Функцию демпфирования вычисляем
    // один раз по текущему полю SA. Материал всех соседних элементов читает результат.
    AERO_HOST_DEVICE void prepare_sa_material(MeshIndex node) const {
        const auto& g=gas[node];
        const double rho_w=g.density*sa.w[node];
        sa.nodal_material[node]={g.mu/g.density,rho_w*turbulence::fv1(rho_w/g.mu)};
    }
#endif
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
#ifdef AERO_HAS_TURBULENCE
        if (sa.active) prepare_sa_material(node);
#endif
    }
    // Для работы напряжений в уравнении энергии усредняем именно произведение mu*u.
    AERO_HOST_DEVICE void average_material(MeshIndex index) const {
        const auto& cell = cells[index];
        Material average{};
#ifdef AERO_HAS_TURBULENCE
        double molecular_sum=0., working_sum=0.;
#endif
        for (unsigned k = 0; k < cell.count; ++k) {
            const auto& point = gas[cell.node[k]];
            double mu = point.mu;
#ifdef AERO_HAS_TURBULENCE
            if (sa.active) {
                const double working=sa.w[cell.node[k]];
                const auto properties=sa.nodal_material[cell.node[k]];
                molecular_sum+=properties.nu;
                working_sum+=working;
                const double turbulent_mu=properties.eddy_mu;
                mu += turbulent_mu;
                average.extra_heat += turbulent_mu*(sa.pr_ratio-1.);
            }
#endif
            average.mu += mu;
            for (unsigned m = 0; m < dim; ++m) average.mu_velocity[m] += mu * point.velocity[m];
        }
#ifdef AERO_HAS_TURBULENCE
        if (sa.active) {
            // Газ и SA используют одни вершины. Средние для диффузии SA готовим
            // при том же чтении узлов, что и напряжения и тепловой поток газа.
            sa.diffusion.molecular_mean[index]=molecular_sum*cell.inverse_count;
            sa.diffusion.element_mean[index]=sa.diffusion.molecular_mean[index]+working_sum*cell.inverse_count;
        }
        average.extra_heat *= cell.inverse_count;
#endif
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
            double diffusion_mu=average.mu;
#ifdef AERO_HAS_TURBULENCE
            diffusion_mu+=::fmax(0.,average.extra_heat);
#endif
            result.cfl_rate += diffusion_mu * cells[e].diffusion[i];
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
