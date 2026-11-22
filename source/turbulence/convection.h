#pragma once
#include "../residual/residual_detail.h"

namespace aero {
namespace turbulence {
// Все указатели относятся к одному исполнителю. Номера и коэффициенты шаблона
// принадлежат газовому солверу. Изменяем только поле, по которому вычисляются наклоны.
struct SaEdgeConvection {
    ReconstructionStencilView stencil;
    const residual_detail::Surface* surfaces;
    const MeshIndex* computed; // В MPI здесь только площадки, назначенные данному процессу.
    const EulerFluxResult* gas_flux;
    const double* working;     // Текущее nu_tilde, включая удалённые узлы шаблона.
    double* correction;        // Поправка к потоку первого порядка в направлении begin -> end.
    double upwind, hybrid;

    AERO_HOST_DEVICE double increment(MeshIndex edge, bool right) const {
        const auto side = right ? ReconstructionSide::Right : ReconstructionSide::Left;
        const auto owner = stencil.row_owner[2*edge + right];
        const ReconstructionFieldView field{working,1,1};
        double value = 0.;
        reconstruct_increment2(stencil, edge, side, working+owner, 1, field, &value);
        return value;
    }

    // Возвращаем сдвиг относительно узла выше по потоку. В обычном случае
    // нужна лишь одна сторона. Обе стороны нужны только для центральной примеси.
    AERO_HOST_DEVICE double face_increment(MeshIndex edge, bool right) const {
        const auto& face = surfaces[edge];
        if (hybrid > 0. && (stencil.mp5.enabled || stencil.mp5.weno) &&
            stencil.weno_order[2*edge] == 5 && stencil.weno_order[2*edge+1] == 5) {
            const ReconstructionFieldView field{working,1,1};
            double slopes[5][1];
            for (unsigned k=0; k<5; ++k) {
                const auto row = 8*edge + residual_detail::edge_slope_offset(k,5);
                slopes[k][0] = evaluate_reconstruction_sparse(stencil.slope_offsets, stencil.slope_nodes,
                    stencil.slope_weights, row, row/4, 0, field);
            }
            double left = 0., other = 0.;
            if (stencil.mp5.weno) mp6::reconstruct<true>(slopes,1,stencil.mp5,hybrid,&left,&other);
            else mp6::reconstruct<false>(slopes,1,stencil.mp5,hybrid,&left,&other);
            return right ? -.5*other : .5*left;
        }
        if (!stencil.weno_order && upwind < 1.) {
            residual_detail::EdgeTraces values;
            values.left[0] = working[face.begin] + .5*increment(edge,false);
            values.right[0] = working[face.end] - .5*increment(edge,true);
            residual_detail::blend_ebr_traces(values,1,upwind);
            return right ? values.right[0]-working[face.end] : values.left[0]-working[face.begin];
        }
        return (right ? -.5 : .5)*increment(edge,right);
    }

    AERO_HOST_DEVICE void operator()(MeshIndex k) const {
        const auto edge = computed ? computed[k] : k;
        const auto& face = surfaces[edge];
        const double mass = gas_flux[edge].flux[0];
        const bool right = mass < 0.;
        const unsigned mask = face.use_nodal_traces;
        double flux = 0.;
        if (mass != 0. && stencil.row_owner && mask != 1 && !(mask & (right ? 4u : 2u))) {
            const double w = working[right ? face.end : face.begin];
            // Ограничиваем добавку к значению до [-w,w]. Тогда уносимая величина
            // лежит в [0,2w]; из узла с w=0 поправка ничего вывести не может.
            const double delta = face_increment(edge,right);
            flux = mass * ::fmax(-w,::fmin(w,delta));
        }
        correction[edge] = flux;
    }
};
}
}
