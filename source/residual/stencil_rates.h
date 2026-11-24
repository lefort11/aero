#pragma once
#include "residual_detail.h"
#include "../timestep/local_rate_limit.h"

namespace aero {
namespace residual_detail {
// Владелец проверяет невязку своего узла относительно его собственного среднего.
// Если хотя бы один узел действующей строки отмечен, отмечается и её контрольный объём.
struct StencilRates {
    // Шаблоны газа для определения области влияния невязки.
    ReconstructionStencilView stencil;
    // Геометрия площадок и номера концов.
    SurfaceTable faces;
    // Связь узла с ориентированными площадками.
    FaceIncidence incidence;
    // Границы строк инцидентности по узлам.
    IndexArray<Index> offsets;
    // Списки собственных и вычисляемых узлов.
    const MeshIndex *owned, *computed;
    // Количество внутренних рёбер перед граничными площадками.
    Index edge_count;
    // Параметры ограничения шага по росту невязки.
    LocalRateLimit feedback;
    // Текущие газовые потоки на площадках.
    const EulerFluxResult* flux;
    double *nodes, *edges; // Пары [газ, SA] в локальной нумерации узлов и рёбер.
    // Признаки первого порядка газа и высокого порядка переноса SA.
    bool first_order, high_order_sa;
    // Веса противопоточности и гибридной реконструкции.
    double upwind, hybrid;
    // Стадия распространения оценки от узлов через рёбра к объёмам.
    enum Pass { NodeGrowth, Edges, Volumes } pass;
    // Ненулевое число процессов включает номера источников для диагностики.
    unsigned ranks, rank;

    // Обычно передаём только две оценки. При наблюдении за узлом добавляем
    // номера источников: по одному на газ и SA, независимо от нумерации сетки.
    AERO_HOST_DEVICE unsigned stride() const { return ranks ? 4u : 2u; }

    // Учитывает ограничение одной опоры при поиске максимальной скорости изменения.
    AERO_HOST_DEVICE void include(const double* input, double* result, unsigned k) const {
        if (input[k] > result[k] || (ranks && input[k] == result[k] && input[k+2] > result[k+2])) {
            result[k] = input[k];
            if (ranks) result[k+2] = input[k+2];
        }
    }

    // Обходит опоры разреженной строки для ограничения локального шага.
    AERO_HOST_DEVICE void include_row(StencilRow row, bool gas, bool sa, double* maximum) const {
        for (ReconstructionIndex j=0; j<row.count; ++j) {
            const double* input = nodes + stride()*row.node(j);
            if (gas) include(input,maximum,0);
            if (sa) include(input,maximum,1);
        }
    }
    // Возвращает наибольшее ограничение среди опор двух сторон ребра.
    AERO_HOST_DEVICE void edge_maximum(Index edge) const {
        const auto& face = faces[edge];
        double* maximum = edges + stride()*edge;
        // Концы ребра входят даже в полностью узловой поток первого порядка.
        for (unsigned k=0; k<stride(); ++k) maximum[k] = nodes[stride()*face.begin+k];
        for (unsigned k=0; k<2; ++k) include(nodes+stride()*face.end,maximum,k);
        const unsigned mask = face.use_nodal_traces;
        if (first_order || mask == 1 || mask == 6 || !stencil.has_owners()) return;
        const Index left = 2*edge;
        const bool coupled = (!stencil.weno_order && upwind < 1.) ||
            (hybrid > 0. && stencil.weno_order && (stencil.mp5.enabled || stencil.mp5.weno) &&
             stencil.weno_order[left] == 5 && stencil.weno_order[left+1] == 5);
        const double mass = high_order_sa ? flux[edge].flux[0] : 0.;
        const unsigned upstream = mass < 0. ? 1 : 0;
        const bool reconstruct_sa = mass != 0. && !(mask & (2u << upstream));
        for (unsigned side=0; side<2; ++side) {
            // При общей центральной поправке даже одна оставшаяся сторона зависит от обеих строк.
            const bool gas = coupled || !(mask & (2u << side));
            const bool sa = reconstruct_sa && (coupled || side == upstream);
            if (!gas && !sa) continue;
            const Index row = left+side;
            const unsigned order = stencil.weno_order ? stencil.weno_order[row] : 0;
            if (!order) include_row(stencil.linear_row(row),gas,sa,maximum);
            else for (unsigned k=0; k<weno_slope_count(order); ++k)
                include_row(stencil.slope_row(4*row+k),gas,sa,maximum);
        }
    }
    // Выполняет проход ограничений по рёбрам или собирает их в узловые значения.
    AERO_HOST_DEVICE void operator()(Index index) const {
        if (pass == Edges) {
            edge_maximum(computed ? computed[index] : index);
            return;
        }
        const Index node = owned ? owned[index] : index;
        double* value = nodes + stride()*node;
        if (pass == NodeGrowth) {
            const double* mean = feedback.history + LocalRateLimit::Width*index;
            value[0] = LocalRateLimit::rising(feedback.gas_rate[index],mean[LocalRateLimit::GasMean],feedback.gas_max);
            value[1] = LocalRateLimit::rising(feedback.sa_magnitude(node),mean[LocalRateLimit::SaMean],feedback.sa_max);
            if (ranks) {
                const double mark = static_cast<double>(1 + index*ranks + rank);
                for (unsigned k=0; k<2; ++k) value[k+2] = value[k] > 0. ? mark : 0.;
            }
            return;
        }
        // Для физической границы достаточно значения самого узла. После сбора
        // рёбер сюда же можно записать результат: соседние узлы больше не читаются.
        for (Index j=offsets[node]; j<offsets[node+1]; ++j) {
            const Index face = incidence[j].face();
            if (face >= edge_count) continue;
            for (unsigned k=0; k<2; ++k) include(edges+stride()*face,value,k);
        }
    }
};
}
}
