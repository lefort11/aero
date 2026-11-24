#ifndef AERO_IMPLICIT_CSGS_SWEEP_H
#define AERO_IMPLICIT_CSGS_SWEEP_H
#include "block_jacobi_detail.h"
#include "block_layout.h"

namespace aero {
namespace implicit {
// Номера записей строки в прямом и обратном проходах. На GPU соседние строки
// читают одноимённые коэффициенты подряд, на CPU вся строка лежит непрерывно.
template<class Index> struct CsgsRowOf { Index first[2], count[2]; };
using CsgsRow = CsgsRowOf<MeshIndex>;
using CsgsRow32 = CsgsRowOf<std::uint32_t>;
// Схема переноса блоков CSR в упакованные строки CSGS.
struct CsgsPacking {
    const MeshIndex* source; // Адрес исходного блока CSR для каждой заполненной позиции.
    double* values;
    MeshIndex count;
    unsigned width, lanes;
    BlockLayout input;        // Раскладка матрицы до перестановки коэффициентов.
    LinearCoefficient* neighbour_values; // Точность внедиагональных блоков задаётся при сборке.

    // Один адрес используют как при сборке непосредственно в порядке CSGS,
    // так и при перекладке уже готовой матрицы.
    AERO_HOST_DEVICE MeshIndex first_coefficient(MeshIndex pos) const {
        return (pos / lanes) * width * width * lanes + pos % lanes;
    }
    // Перекладываем только значения после сборки новой пространственной матрицы.
    // Раскраска, соответствие соседей и разбиение по проходам от коэффициентов не зависят.
    AERO_HOST_DEVICE void copy(MeshIndex pos, const double* matrix) const {
        if (source[pos]==static_cast<MeshIndex>(-1)) return;
        const unsigned size=width*width;
        const auto first=first_coefficient(pos);
        // В этом массиве находятся только соседи. Для mixed обе раскладки
        // имеют одинаковую точность, при перестановке коэффициент не меняется.
        const auto slot=source[pos];
        const auto offset=input.first ? input.first[slot] : slot*size;
        for (unsigned k=0;k<size;++k) {
            if (neighbour_values) neighbour_values[first+k*lanes]=input.neighbours[offset+k];
            else values[first+k*lanes]=matrix[offset+k];
        }
    }
};
// Представление диагонали и соседних блоков для проходов CSGS.
struct CsgsSweep {
    JacobiBlocks diagonal;
    const CsgsRow* rows;
    const CsgsRow32* short_rows; // Вмещающие весь граф 32-разрядные смещения на GPU.
    IndexList sequence, neighbours;
    const double *blocks, *rhs, *forward_with_halo;
    double *output, *middle_rhs;
    bool backward, external;
    unsigned lanes;
    const LinearCoefficient* neighbour_values; // Адрес отдельно размещённых соседних блоков.

    // Разворачиваем только адреса обрабатываемой строки. Оба прохода и умножение
    // используют одну нумерацию независимо от ширины хранимых индексов.
    AERO_HOST_DEVICE CsgsRow row_range(MeshIndex index) const {
        if (!short_rows) return rows[index];
        const auto r = short_rows[index];
        return {{r.first[0], r.first[1]}, {r.count[0], r.count[1]}};
    }
    // Обновляет одну строку CSGS по уже доступным соседним значениям.
    template<unsigned width,unsigned stride,class Coefficient>
    AERO_HOST_DEVICE void solve_row(MeshIndex index,const Coefficient* coefficients) const {
        const MeshIndex row=sequence[index], start=row*width;
        // После прямого прохода уже известна правая часть b-L*y. Сохраняем её
        // отдельно от y, чтобы в обратном проходе повторно не читать блоки L.
        const double* original=backward ? middle_rhs : rhs;
        double balance[width];
        for (unsigned k=0;k<width;++k) balance[k]=original[start+k];

        const auto range=row_range(index);
        auto pos=range.first[backward];
        for (MeshIndex j=0;j<range.count[backward];++j,pos+=stride) {
            const auto neighbour=neighbours[pos];
            // Для чужой подобласти берём результат прямого прохода после обмена MPI.
            // Младший бит хранит этот выбор, остальные — номер строки или узла ореола.
            const double* field=(neighbour&1) ? forward_with_halo : output;
            const double* x=field+(neighbour/2)*width;
            const auto* a=coefficients+(pos/stride)*stride*width*width+pos%stride;
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
    template<unsigned width,unsigned stride,class Coefficient>
    AERO_HOST_DEVICE void image_row(MeshIndex index,double* image,const Coefficient* coefficients) const {
        const auto row=sequence[index], start=row*width;
        double sum[width];
        for (unsigned k=0;k<width;++k) sum[k]=rhs[start+k];
        const auto range=row_range(index);
        for (unsigned pass=0;pass<(external ? 2u : 1u);++pass) {
            auto pos=range.first[pass];
            for (MeshIndex j=0;j<range.count[pass];++j,pos+=stride) {
                const auto neighbour=neighbours[pos];
                if (pass && !(neighbour&1)) continue;
                const auto n=neighbour/2*width;
                const auto* a=coefficients+(pos/stride)*stride*width*width+pos%stride;
                for (unsigned k=0;k<width;++k) {
                    const double difference=output[n+k]-forward_with_halo[n+k];
                    for (unsigned i=0;i<width;++i) sum[i]+=a[(i*width+k)*stride]*difference;
                }
            }
        }
        for (unsigned k=0;k<width;++k) image[start+k]=sum[k];
    }
    // Вычисляет образ вектора для выбранного формата строк CSGS.
    template<unsigned stride>
    AERO_HOST_DEVICE void image(MeshIndex index,double* result) const {
        if (neighbour_values) image_typed<stride>(index,result,neighbour_values);
        else image_typed<stride>(index,result,blocks);
    }
    // Вычисляет образ вектора для выбранного формата строк CSGS.
    template<unsigned stride,class Coefficient>
    AERO_HOST_DEVICE void image_typed(MeshIndex index,double* result,const Coefficient* a) const {
        switch (diagonal.width) {
        case 3: return image_row<3,stride>(index,result,a);
        case 4: return image_row<4,stride>(index,result,a);
        default: return image_row<5,stride>(index,result,a);
        }
    }
    // Запускает проход CSGS с подходящим форматом индексов строк.
    template<unsigned stride>
    AERO_HOST_DEVICE void solve(MeshIndex index) const {
        if (neighbour_values) solve_typed<stride>(index,neighbour_values);
        else solve_typed<stride>(index,blocks);
    }
    // Выполняет проход CSGS с выбранной разрядностью индексов.
    template<unsigned stride,class Coefficient>
    AERO_HOST_DEVICE void solve_typed(MeshIndex index,const Coefficient* a) const {
        switch (diagonal.width) {
        case 3: return solve_row<3,stride>(index,a);
        case 4: return solve_row<4,stride>(index,a);
        default: return solve_row<5,stride>(index,a);
        }
    }
    // Копирует коэффициенты одного блока в упакованную раскладку CSGS.
    AERO_HOST_DEVICE void operator()(MeshIndex index) const {
        if (lanes==32) solve<32>(index);
        else solve<1>(index);
    }
};
}
}
#endif
