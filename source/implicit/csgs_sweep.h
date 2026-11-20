#ifndef AERO_IMPLICIT_CSGS_SWEEP_H
#define AERO_IMPLICIT_CSGS_SWEEP_H
#include "block_jacobi_detail.h"

namespace aero {
namespace implicit {
// Адреса коэффициентов и порядок обхода для одного MPI-процесса.
// Строка матрицы компактна, а столбец обозначает узел полного поля с соседями.
struct CsgsSweep {
    JacobiBlocks diagonal;
    const MeshIndex *rows, *columns;                             // Начала строк CSR и номера узлов столбцов.
    const MeshIndex *sequence, *row_of_node, *position, *domain; // Таблицы CsgsOrder.
    const double *blocks, *rhs;      // Коэффициенты плотных блоков и исходная правая часть.
    const double* forward_with_halo; // Неизменный результат прямой подстановки после обмена.
    double* output;
    bool backward;
    AERO_HOST_DEVICE void operator()(MeshIndex index) const {
        const MeshIndex row = sequence[index];
        const unsigned width = diagonal.width;
        double balance[5];
        for (unsigned k = 0; k < width; ++k)
            balance[k] = rhs[row * width + k];
        // На старте BiCGSTAB просит M^{-1}b. Начальное приближение равно нулю.
        // Прямой проход использует только уже обработанных соседей своей подобласти.
        for (MeshIndex p = rows[row] + 1; p < rows[row + 1]; ++p) {
            const MeshIndex node = columns[p], other = row_of_node[node];
            const bool local = other < diagonal.nodes && (!domain || domain[row] == domain[other]);
            const bool updated = local && (backward ? position[other] > index : position[other] < index);
            if (!updated && !backward) continue;
            // В обратном проходе для соседей за границей подобласти и ещё не пройденных
            // строк берём неизменную копию результата прямого прохода.
            const double* x = updated ? output + other * width : forward_with_halo + node * width;
            const double* block = blocks + p * width * width;
            for (unsigned i = 0; i < width; ++i)
                for (unsigned j = 0; j < width; ++j)
                    balance[i] -= block[i * width + j] * x[j];
        }
        diagonal.solve_block(row, balance, output + row * width);
    }
};
}
}
#endif
