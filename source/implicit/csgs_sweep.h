#ifndef AERO_IMPLICIT_CSGS_SWEEP_H
#define AERO_IMPLICIT_CSGS_SWEEP_H
#include "block_jacobi_detail.h"

namespace aero {
namespace implicit {
// Номера записей строки в прямом и обратном проходах. На GPU соседние строки
// читают одноимённые коэффициенты подряд, на CPU вся строка лежит непрерывно.
struct CsgsRow {
    MeshIndex first[2], count[2];
};
struct CsgsPacking {
    const MeshIndex* source; // Адрес исходного блока CSR для каждой заполненной позиции.
    double* values;
    MeshIndex count;
    unsigned width, lanes;

    // Перекладываем только значения после сборки новой пространственной матрицы.
    // Раскраска, соответствие соседей и разбиение по проходам от коэффициентов не зависят.
    AERO_HOST_DEVICE void copy(MeshIndex pos, const double* matrix) const {
        if (source[pos]==static_cast<MeshIndex>(-1)) return;
        const unsigned size=width*width;
        const auto first=(pos/lanes)*size*lanes+pos%lanes;
        for (unsigned k=0;k<size;++k) values[first+k*lanes]=matrix[source[pos]*size+k];
    }
};
struct CsgsSweep {
    JacobiBlocks diagonal;
    const CsgsRow* rows;
    const MeshIndex *sequence, *neighbours;
    const double *blocks, *rhs, *forward_with_halo;
    double *output, *middle_rhs;
    bool backward, external;
    unsigned lanes;

    template<unsigned width,unsigned stride>
    AERO_HOST_DEVICE void solve_row(MeshIndex index) const {
        const MeshIndex row=sequence[index], start=row*width;
        // После прямого прохода уже известна правая часть b-L*y. Сохраняем её
        // отдельно от y, чтобы в обратном проходе повторно не читать блоки L.
        const double* original=backward ? middle_rhs : rhs;
        double balance[width];
        for (unsigned k=0;k<width;++k) balance[k]=original[start+k];

        const auto range=rows[index];
        auto pos=range.first[backward];
        for (MeshIndex j=0;j<range.count[backward];++j,pos+=stride) {
            const auto neighbour=neighbours[pos];
            // Для чужой подобласти берём результат прямого прохода после обмена MPI.
            // Младший бит хранит этот выбор, остальные — номер строки или узла ореола.
            const double* field=(neighbour&1) ? forward_with_halo : output;
            const double* x=field+(neighbour/2)*width;
            const double* a=blocks+(pos/stride)*stride*width*width+pos%stride;
            for (unsigned i=0;i<width;++i)
                for (unsigned k=0;k<width;++k) balance[i]-=a[(i*width+k)*stride]*x[k];
        }
        if (!backward)
            for (unsigned k=0;k<width;++k) middle_rhs[start+k]=balance[k];
        diagonal.solve_fixed<width>(row,balance,output+start);
    }
    // После прямой подстановки D*y = b−L*y, после обратной (D+U)*z = D*y−E*y.
    // Поэтому A*z = b+(L+E)*(z−y). E связывает участки разных потоков CPU.
    // Полная CSR-матрица остаётся эталоном для проверки линейной невязки.
    template<unsigned width,unsigned stride>
    AERO_HOST_DEVICE void image_row(MeshIndex index,double* image) const {
        const auto row=sequence[index], start=row*width;
        double sum[width];
        for (unsigned k=0;k<width;++k) sum[k]=rhs[start+k];
        const auto range=rows[index];
        for (unsigned pass=0;pass<(external ? 2u : 1u);++pass) {
            auto pos=range.first[pass];
            for (MeshIndex j=0;j<range.count[pass];++j,pos+=stride) {
                const auto neighbour=neighbours[pos];
                if (pass && !(neighbour&1)) continue;
                const auto n=neighbour/2*width;
                const double* a=blocks+(pos/stride)*stride*width*width+pos%stride;
                for (unsigned k=0;k<width;++k) {
                    const double difference=output[n+k]-forward_with_halo[n+k];
                    for (unsigned i=0;i<width;++i) sum[i]+=a[(i*width+k)*stride]*difference;
                }
            }
        }
        for (unsigned k=0;k<width;++k) image[start+k]=sum[k];
    }
    template<unsigned stride>
    AERO_HOST_DEVICE void image(MeshIndex index,double* result) const {
        switch (diagonal.width) {
        case 3: return image_row<3,stride>(index,result);
        case 4: return image_row<4,stride>(index,result);
        default: return image_row<5,stride>(index,result);
        }
    }
    template<unsigned stride>
    AERO_HOST_DEVICE void solve(MeshIndex index) const {
        switch (diagonal.width) {
        case 3: return solve_row<3,stride>(index);
        case 4: return solve_row<4,stride>(index);
        default: return solve_row<5,stride>(index);
        }
    }
    AERO_HOST_DEVICE void operator()(MeshIndex index) const {
        if (lanes==32) solve<32>(index);
        else solve<1>(index);
    }
};
}
}
#endif
