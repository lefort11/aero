#pragma once
#include "../residual/residual_detail.h"

namespace aero {
namespace turbulence {
// Все указатели относятся к одному исполнителю. Номера и коэффициенты шаблона
// принадлежат газовому солверу. Изменяем только поле, по которому вычисляются наклоны.
struct SaEdgeConvection {
    // Шаблоны конвективной реконструкции рабочей вязкости.
    ReconstructionStencilView stencil;
    // Геометрия и ориентация площадок.
    residual_detail::SurfaceTable surfaces;
    const MeshIndex* computed; // В MPI здесь только площадки, назначенные данному процессу.
    // Газовый массовый поток, определяющий направление переноса.
    const EulerFluxResult* gas_flux;
    const double* working;     // Текущее nu_tilde, включая удалённые узлы шаблона.
    double* correction;        // Поправка к потоку первого порядка в направлении begin -> end.
    // Веса противопоточности и гибридной схемы.
    double upwind, hybrid;

    // На полном шаблоне читает одну скалярную строку из той же плитки, что и газ.
    template<unsigned row,unsigned order>
    AERO_HOST_DEVICE double scalar_slope(const residual_detail::WenoEdgeTile<order>& tile,unsigned lane) const {
        constexpr unsigned first=residual_detail::first_slope_term(row);
        const double origin=working[tile.nodes(first,lane)];
        double value=tile.weights(first-row,lane)*(working[tile.nodes(first+1,lane)]-origin);
        if (!tile.order[lane].paired())
            for (unsigned j=2; j<residual_detail::slope_terms(row); ++j)
                value+=tile.weights(first-row+j-1,lane)*(working[tile.nodes(first+j,lane)]-origin);
        return value;
    }
    // Для конвекции SA нужна одна сторона по знаку потока массы. У неполных
    // шаблонов и отмеченных сторон оставляем обычное правило реконструкции.
    template<unsigned order, bool mp>
    AERO_HOST_DEVICE void packed(MeshIndex k,const residual_detail::WenoEdgeTile<order>* tiles) const {
        using namespace residual_detail;
        const auto edge=computed ? computed[k] : k;
        // При нулевом массовом потоке обе стороны дают нулевую поправку.
        // В этом случае плитку, флаги реконструкции и значения SA читать не требуется.
        const double mass = gas_flux[edge].flux[0];
        if (mass == 0.) { correction[edge] = 0.; return; }
        const auto& tile=tiles[edge_tile::index(edge)];
        const unsigned lane=edge_tile::lane(edge);
        if (!tile.order[lane].complete() || surfaces[edge].use_nodal_traces) { (*this)(k); return; }
        const bool right=mass<0.;
        const double center=scalar_slope<CentralSlope>(tile,lane);
        const double left=scalar_slope<NearLeftSlope>(tile,lane), other=scalar_slope<NearRightSlope>(tile,lane);
        const double far=order==5 ? (right ? scalar_slope<FarRightSlope>(tile,lane) : scalar_slope<FarLeftSlope>(tile,lane)) : 0.;
        const double slopes[]={center,right ? other : left,right ? left : other,far};
        const double value=mp ? mp5::increment(far,slopes[1],center,slopes[2],stencil.mp5.alpha) :
                                weno_increment2<order>(order,slopes,stencil.epsilon(2*edge+right));
        const auto& face=surfaces[edge];
        const double w=working[right ? face.end : face.begin];
        correction[edge]=mass*::fmax(-w,::fmin(w,(right ? -.5 : .5)*value));
    }

    // Восстанавливает приращение рабочей вязкости на выбранной стороне ребра.
    AERO_HOST_DEVICE double increment(MeshIndex edge, bool right) const {
        const auto side = right ? ReconstructionSide::Right : ReconstructionSide::Left;
        const auto owner = stencil.owner(2*edge + right);
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
            FixedMatrix<double,5,1> slopes;
            for (unsigned k=0; k<5; ++k) {
                const auto row = 8*edge + residual_detail::edge_slope_offset(k,5);
                slopes(k, 0) = evaluate_reconstruction_row(stencil.slope_row(row),row/4,0,field);
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

    // Вычисляет противопоточный перенос SA по направлению газового потока массы.
    AERO_HOST_DEVICE void operator()(MeshIndex k) const {
        const auto edge = computed ? computed[k] : k;
        const auto& face = surfaces[edge];
        const double mass = gas_flux[edge].flux[0];
        const bool right = mass < 0.;
        const unsigned mask = face.use_nodal_traces;
        double flux = 0.;
        if (mass != 0. && stencil.has_owners() && mask != 1 && !(mask & (right ? 4u : 2u))) {
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
