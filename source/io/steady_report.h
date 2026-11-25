#pragma once
#include "diagnostic_fields.h"
#include "../mesh/mesh_index.h"
#include <algorithm>

namespace aero {
// Частичная сводка стационарного шага. Полосы читают соседние узлы в соседних
// потоках GPU. Результаты занимают освободившуюся правую часть Ньютона.
struct SteadyReport {
    enum { Columns = 21 };       // Шесть минимумов, максимумов, интегралов и три величины для RMS.
    // Буфер сводки помещается и в правую часть газа, и в компактное поле хоста.
    static MeshIndex partial_size(MeshIndex values) {
        return std::min<MeshIndex>(2048,values/Columns)*Columns;
    }
    MeshIndex nodes, strips;     // Число узлов и независимых полос накопления.
    unsigned dim;                // Размерность газа и ширина консервативного поля минус два.
    double gamma;                // Показатель адиабаты для давления.
    const double *start, *state; // Принятые консервативные поля до и после псевдошага.
    const double* volume;        // Объём контрольной области узла.
    double* partial;             // Столбцы частичных сумм с шагом strips.

    // Сохраняет все показатели полосы. Неконечный масштаб означает, что для
    // предельных значений интеграла требуется расширенная точность CPU.
    AERO_HOST_DEVICE void operator()(MeshIndex lane) const {
        double low[6], high[6], sum[6]{};
        for (unsigned k=0; k<6; ++k) { low[k]=max_finite_double; high[k]=-max_finite_double; }
        double scale=0., squares=0., weight=0.;
        const unsigned width=dim+2;
        for (MeshIndex i=lane; i<nodes; i+=strips) {
            double sampled[6];
            if (!diagnostic::primary_from_state(dim,state+i*width,gamma,{},sampled)) { scale=HUGE_VAL; break; }
            const double v=volume[i];
            weight+=v;
            for (unsigned k=0; k<=width; ++k) {
                const double value=sampled[k<=dim ? k : k==width ? 5 : 4];
                low[k]=::fmin(low[k],value);
                high[k]=::fmax(high[k],value);
                sum[k]+=v*value;
            }
            const double delta=::fabs(sampled[5]-start[i*width+width-1]);
            if (delta>scale) {
                const double ratio=scale/delta;
                squares=v+squares*ratio*ratio;
                scale=delta;
            } else if (delta>0.) {
                const double ratio=delta/scale;
                squares+=v*ratio*ratio;
            }
        }
        for (unsigned k=0; k<=width; ++k) {
            partial[k*strips+lane]=low[k];
            partial[(6+k)*strips+lane]=high[k];
            partial[(12+k)*strips+lane]=sum[k];
        }
        partial[18*strips+lane]=scale;
        partial[19*strips+lane]=squares;
        partial[20*strips+lane]=weight;
    }
};
}
