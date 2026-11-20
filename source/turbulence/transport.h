#pragma once
#include "../timestep/node_step.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#include "../io/diagnostic_fields.h"

namespace aero {
namespace turbulence {
// Все индексы и указатели относятся к одной подобласти и одному исполнителю.
struct Transport {
    View sa; // Только массивы, нужные уравнению SA на выбранном исполнителе.
    const viscosity::Gas *gas_nodes;
    const viscosity::Cell *cells;
    MeshIndex cell_count, edges;
    unsigned dim;
    const EulerFluxResult *flux;
    const double *volume;
    const MeshIndex *owned;
    MeshIndex count;

    AERO_HOST_DEVICE MeshIndex node(MeshIndex k) const { return owned ? owned[k] : k; }
    // В уравнении переноса разность с соседним значением нужна только при втекании.
    AERO_HOST_DEVICE double incoming_speed(residual_detail::OrientedFace side, double inverse_density) const {
        const double inward = -side.outward_flux(flux[side.face()].flux[0]) * inverse_density;
        return inward > 0. ? inward : 0.;
    }
    // Значение вычисляется один раз на элемент и используется во всех прилегающих строках.
    AERO_HOST_DEVICE void refresh_diffusivity(MeshIndex e) const {
        const auto &cell = cells[e];
        double sum = 0.;
        for (unsigned j = 0; j < cell.count; ++j) {
            const auto n = cell.node[j];
            sum += sa.w[n];
        }
        sa.diffusion.element_mean[e] = sa.diffusion.molecular_mean[e] + sum * cell.inverse_count;
    }
    // Схема первого порядка по потоку массы сохраняет постоянное nu_tilde и при
    // переменной плотности. На втекании используем набегающий поток, на вытекании — значение узла.
    template <unsigned d, bool reuse_gas, bool host = false>
    AERO_HOST_DEVICE void residual_row(MeshIndex k, bool save_rotation) const {
        const auto i = node(k);
        if (sa.inverse_distance[i] < 0.) {
            sa.rate[i] = sa.diagonal[i] = 0.;
            return;
        }
        const auto &gas = gas_nodes[i];
        const double w = sa.w[i], nu = sa.nodal_material[i].nu;
        double advection = 0., diffusion = 0., diagonal = 0., dw[3]{};
        diagnostic::Curl curl{};
        const double inverse_density = 1. / gas.density;
        const auto faces = sa.face_rows[i];
        for (auto pos = faces.first; pos < faces.end; pos += (host ? 1 : sa.face_stride)) {
            const auto &face = sa.face_terms[pos];
            const residual_detail::OrientedFace orientation{face.encoded};
            const bool edge = orientation.face() < edges;
            const auto j = face.neighbour;
            // В серии скалярных итераций поток газа на ребре один и тот же.
            const double incoming = reuse_gas ? sa.incoming[pos] : incoming_speed(orientation, inverse_density);
            if (!reuse_gas && save_rotation)
                sa.incoming[pos] = incoming;
            const double jump = edge ? sa.w[j] - w : (incoming > 0. ? sa.ambient - w : 0.);
            advection += incoming * jump;
            diagonal += incoming;
            for (unsigned m = 0; m < d; ++m)
                dw[m] += face.area[m] * jump;
            if (!reuse_gas && edge) {
                double jump_u[3]{};
                for (unsigned n = 0; n < d; ++n)
                    jump_u[n] = gas_nodes[j].velocity[n] - gas.velocity[n];
                curl.add(face.area, jump_u, 1.);
            }
        }
        const double inv_volume = 1. / volume[i];
        double grad2 = 0.;
        for (unsigned m = 0; m < d; ++m) {
            const double derivative = dw[m] * inv_volume;
            grad2 += derivative * derivative;
        }
        // Каждая ненулевая пара даёт приращение от разности рабочих вязкостей.
        // Коэффициенты могут иметь разный знак на косой сетке, поэтому в оценке
        // шага берём модуль каждого вклада до суммирования по элементам.
        const auto &stencil = sa.diffusion;
        const auto row = stencil.rows[i];
        const auto base_element = stencil.anchor ? stencil.anchor[i] : 0;
        const auto base_node = stencil.anchor ? i : 0;
        for (auto pos = row.first; pos < row.end; pos += (host ? 1 : stencil.stride)) {
            const auto term = stencil.terms[pos];
            const double coupling = stencil.element_mean[base_element + term.element] * term.geometry;
            diffusion += coupling * (sa.w[base_node + term.neighbour] - w);
            diagonal += ::fabs(coupling);
        }
        // Пока Ньютон уточняет только рабочую вязкость, поле скорости неизменно.
        const double rotation = reuse_gas ? sa.rotation[i] : curl.magnitude(inv_volume);
        if (!reuse_gas && save_rotation)
            sa.rotation[i] = rotation;
        const double source = source_rate(w, nu, rotation, sa.inverse_distance[i]);
        sa.rate[i] = (advection + diffusion) * inv_volume + cb2 / sigma * grad2 + source * w;
        sa.diagonal[i] = diagonal * inv_volume + ::fabs(source);
    }
    template <bool reuse_gas> AERO_HOST_DEVICE void residual(MeshIndex k, bool save_rotation) const {
        if (dim == 3)
            residual_row<3, reuse_gas>(k, save_rotation);
        else if (dim == 2)
            residual_row<2, reuse_gas>(k, save_rotation);
        else
            residual_row<1, reuse_gas>(k, save_rotation);
    }
};
enum class Operation {
    Residual,
    RepeatResidual,
    Explicit,
    SaveStart,
    SaveBackup,
    RestoreBackup,
    CommitHistory,
    Relax
};
enum class Norm { Rate, Diagonal, Bdf };
struct Step {
    Transport transport;
    explicit_detail::Stage stage;
    double dt, leading, history;
    NodeSteps local_steps;
    bool keep_gas; // Давление, плотность и скорость фиксированы между итерациями скалярного уравнения.
    // Погрешность дискретного уравнения на новом слое BDF в единицах рабочей вязкости.
    AERO_HOST_DEVICE double defect(MeshIndex i) const {
        const auto &s = transport.sa;
        const double previous = history ? history * (s.old[i] - s.previous[i]) : 0.;
        return leading * (s.w[i] - s.old[i]) - previous - local_steps.at(i, dt) * s.rate[i];
    }
    AERO_HOST_DEVICE void apply(Operation op, MeshIndex k) const {
        const auto i = transport.node(k);
        const auto &s = transport.sa;
        switch (op) {
        case Operation::Residual:
            transport.residual<false>(k, keep_gas);
            break;
        case Operation::RepeatResidual:
            transport.residual<true>(k, false);
            break;
        case Operation::SaveStart:
            s.old[i] = s.w[i];
            break;
        case Operation::SaveBackup:
            s.backup[i] = s.w[i];
            break;
        case Operation::RestoreBackup:
            s.w[i] = s.backup[i];
            break;
        case Operation::CommitHistory:
            s.previous[i] = s.old[i];
            break;
        case Operation::Explicit:
        case Operation::Relax: {
            // Положительная ветвь SA. На стенке задано нулевое значение на всех стадиях.
            const double next =
                op == Operation::Relax
                    ? s.w[i] - defect(i) / (leading + local_steps.at(i, dt) * s.diagonal[i])
                    : stage.combine(stage.uses_step_start() ? s.old[i] : s.w[i], s.w[i] + dt * s.rate[i]);
            s.w[i] = s.inverse_distance[i] < 0. ? 0. : (next < 0. ? 0. : next);
            break;
        }
        }
    }
    AERO_HOST_DEVICE double magnitude(Norm norm, MeshIndex k) const {
        const auto i = transport.node(k);
        const auto &s = transport.sa;
        double value = norm == Norm::Bdf ? defect(i) : (norm == Norm::Rate ? s.rate[i] : s.diagonal[i]);
        // При нулевом значении рабочей вязкости допускается только неотрицательное обновление.
        if (norm == Norm::Bdf && s.w[i] == 0. && value > 0.)
            value = 0.;
        return ::fabs(value) <= max_finite_double ? ::fabs(value) : HUGE_VAL;
    }
};
}
}
