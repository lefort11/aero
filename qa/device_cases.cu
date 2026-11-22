// Каждая проверка исполняет ту же формулу, что и CPU-тест, на данных карты.
// Очередь передаёт вызывающий тест, чтобы сохранить порядок внутри стадии.
#define AERO_BUILD_DEVICE_WORK
#include "device_cases.h"
#include "../source/implicit/roe_edge_jacobian.h"
#include "../source/implicit/lf_edge_jacobian.h"
#include "../source/parallel/device_execution.h"
#include "../source/eulersolver/lax_friedrichs.h"
#include "../source/eulersolver/roe.h"
#include "../source/eulersolver/characteristic.h"
namespace aero {
namespace device_tests {
// Входим через ту же функцию, которую будет вызывать сборщик матрицы на GPU.
const char* edge_blocks(Case input, double area, double upwind, implicit::EdgeFluxJacobian* output, bool simplified, bool roe) {
    return device_work::on_device(nullptr, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: LF edge Jacobian", 1, KOKKOS_LAMBDA(MeshIndex) {
            const euler_detail::FaceStates face = {input.dimension, input.left, input.right,
                                                   input.direction, input.gamma, upwind};
            if (!(roe ? implicit::roe_edge_jacobian(face, area, *output, simplified)
                      : implicit::lf_edge_jacobian(face, area, *output, simplified)))
                output->width = 0;
        });
    });
}
const char* periodic_rows(void* stream, ReconstructionStencilView stencil, const PeriodicEdge* edges, const double* field, double* output) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: periodic_rows", 2 * stencil.edge_count, KOKKOS_LAMBDA(MeshIndex row) {
            const PeriodicEdge& edge = edges[row / 2];
            // Чётная строка отвечает левой стороне ребра, нечётная — правой.
            const bool right = (row % 2) != 0;
            ReconstructionFieldView q = {field, 1, 1};
            auto status =
                reconstruct_increment2(stencil, row / 2, right ? ReconstructionSide::Right : ReconstructionSide::Left,
                                       field + (right ? edge.end : edge.begin), 1, q, output + row);
            if (status != ReconstructionStatus::Success)
                output[row] = 1.e100;
        });
    });
}
const char* test_flux(void* stream, const BcTestCase* cases, size_t count, const FreeBoundaryRules* rules, int* failed) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: test_flux", count, KOKKOS_LAMBDA(MeshIndex i) {
            const BcTestCase& t = cases[i];
            EulerFluxResult result = {};
            // Контрольная метка покажет, перезаписала ли ошибочная операция выходное значение.
            result.flux[0] = 123;
            BoundaryStatus status =
                boundary_flux(t.dimension, t.config, rules, t.current, t.initial_available ? t.initial : nullptr,
                              t.area, t.gamma, &result, t.profile_data());
            bool ok = status == t.expected_status;
            if (status == BoundaryStatus::Success) {
                ok = ok && result.components == t.dimension + 2 &&
                     fabs(result.max_wave_speed - t.expected.max_wave_speed) < 1e-11;
                for (unsigned k = 0; k < t.dimension + 2; ++k)
                    ok = ok && fabs(result.flux[k] - t.expected.flux[k]) < 1e-11;
            } else
                ok = ok && result.flux[0] == 123;
            if (!ok)
                Kokkos::atomic_fetch_min(failed, static_cast<int>(i));
        });
    });
}
const char* inspect(void* stream, aero::CvView v, double* result) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: inspect", 1, KOKKOS_LAMBDA(MeshIndex) {
            // Сравниваем три независимые геометрические суммы с эталонами для этой сетки.
            double volume = 0, outer = 0, edge_projection = 0;
            for (size_t i = 4; i < 12; ++i)
                volume += v.volumes[i];
            for (size_t i = 0; i < v.boundary_count; ++i) {
                const aero::CvBoundary b = v.boundary[i];
                if (b.patch == 1 && (b.node == 5 || b.node == 6 || b.node == 9 || b.node == 10))
                    outer += b.area.x;
            }
            for (size_t e = 0; e < v.edge_count; ++e)
                edge_projection += fabs(v.edge_areas[e].x);
            result[0] = volume;
            result[1] = outer;
            result[2] = edge_projection;
        });
    });
}
const char* one_interface(void* stream, const Case* input, Output* output) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: one_interface", 1, KOKKOS_LAMBDA(MeshIndex) {
            const Case& c = *input;
            aero::EulerStatus status;
            if (c.mode == 0)
                status = aero::lax_friedrichs_conservative(c.dimension, c.left, c.right, c.direction, c.gamma,
                                                           &output->result);
            else if (c.mode == 3)
                status = aero::roe_conservative(c.dimension, c.left, c.right, c.direction, c.gamma, &output->result);
            else if (c.mode == 1 || c.mode == 4) {
                // Перед выбором LF или Roe вызывающий код восстанавливает U из W.
                aero::EulerRoeBasis basis;
                double ul[5], ur[5];
                status = aero::euler_roe_basis(c.dimension, c.aux_left, c.aux_right, c.direction, c.gamma, &basis);
                if (status == aero::EulerStatus::Success)
                    status = aero::characteristic_to_conservative(basis, c.left, ul);
                if (status == aero::EulerStatus::Success)
                    status = aero::characteristic_to_conservative(basis, c.right, ur);
                if (status == aero::EulerStatus::Success)
                    status = c.mode == 1
                                 ? aero::lax_friedrichs_conservative(c.dimension, ul, ur, c.direction, c.gamma,
                                                                     &output->result)
                                 : aero::roe_conservative(c.dimension, ul, ur, c.direction, c.gamma, &output->result);
            } else if (c.mode == 2)
                status = aero::lax_friedrichs_fluxes(c.dimension, c.left, c.right, c.aux_left, c.aux_right,
                                                     c.direction, c.gamma, &output->result);
            else
                status = aero::roe_fluxes(c.dimension, c.left, c.right, c.aux_left, c.aux_right, c.direction, c.gamma,
                                          &output->result);
            output->status = static_cast<int>(status);
        });
    });
}
const char* evaluate(void* stream, TestProblem problem, double time, aero::ExplicitField field, double* residual) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: evaluate", field.nodes * field.components, KOKKOS_LAMBDA(MeshIndex k) {
            residual[k] = test_residual(problem, time, field, k);
        });
    });
}
const char* reconstruct_kernel(void* stream, aero::ReconstructionStencilView s, const double* values, ReconstructionIndex components, double* left, double* right) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: reconstruct_kernel", 2 * s.edge_count, KOKKOS_LAMBDA(MeshIndex row) {
            const aero::ReconstructionFieldView input = {values, 1, s.node_count};
            const ReconstructionIndex edge = row / 2, owner = s.row_owner[row];
            aero::reconstruct_increment2(
                s, edge, row % 2 ? aero::ReconstructionSide::Right : aero::ReconstructionSide::Left,
                values + owner, components, input, (row % 2 ? right : left) + edge, s.node_count, s.edge_count);
        });
    });
}
const char* one_side_kernel(void* stream, aero::ReconstructionStencilView s, ReconstructionIndex edge, unsigned side, const double* own, ReconstructionIndex components, const double* values, double* output) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: one_side_kernel", 1, KOKKOS_LAMBDA(MeshIndex) {
            const aero::ReconstructionFieldView input = {values, 1, s.node_count};
            // Ядро CUDA получает ровно одну сторону и один вектор состояния на конце ребра.
            aero::reconstruct_increment2(s, edge, static_cast<aero::ReconstructionSide>(side), own, components, input,
                                         output);
        });
    });
}
const char* speed_kernel(void* stream, const double* u, double* speed) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: speed_kernel", 1, KOKKOS_LAMBDA(MeshIndex) {
            const double n = 1;
            EulerFluxResult f = {};
            const auto status = lax_friedrichs_conservative(1, u, u, &n, 1.4, &f);
            *speed = status == EulerStatus::Success ? f.max_wave_speed : -1;
        });
    });
}
const char* residual_kernel(void* stream, double* r) {
    return device_work::on_device(stream, [&](const device_work::Execution& exec) {
        device_work::for_indices(exec, "Aero test: residual_kernel", 1, KOKKOS_LAMBDA(MeshIndex) {
            r[0] = 0;
            r[1] = 0;
            r[2] = -5;
        });
    });
}
}
}
