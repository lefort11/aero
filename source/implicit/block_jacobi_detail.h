#ifndef AERO_IMPLICIT_BLOCK_JACOBI_DETAIL_H
#define AERO_IMPLICIT_BLOCK_JACOBI_DETAIL_H
#include "../mesh/mesh_index.h"
#include "../parallel/kernel_math.h"

namespace aero {
namespace implicit {
// Вид рабочих массивов, одинаковый в цикле CPU и в ядре Kokkos на GPU.
struct JacobiBlocks {
    MeshIndex nodes;        // Число собственных узлов процесса.
    unsigned width;         // Число уравнений Эйлера в одном узле: от трёх до пяти.
    double* factors;        // Нижний и верхний треугольники LU хранятся в одном плотном блоке.
    unsigned char* pivots;  // С какой строкой поменяли местами строку k на шаге исключения k.

    // Разлагает один диагональный блок с частичным выбором главного элемента.
    // Слева от диагонали записываем множители L, на диагонали и справа — U.
    AERO_HOST_DEVICE bool factor_node(MeshIndex node, const double* diagonal, const MeshIndex* row_starts) const {
        const MeshIndex offset = node * width * width;
        double* lu = factors + offset;
        unsigned char* swaps = pivots + node * width;
        const double* source = diagonal + (row_starts ? row_starts[node] * width * width : offset);
        for (unsigned i = 0; i < width * width; ++i) {
            const double value = source[i];
            if (!(::fabs(value) <= max_finite_double))
                return false;
            lu[i] = value;
        }
        for (unsigned k = 0; k < width; ++k) {
            // Ведущая строка выбирается среди ещё не обработанных. Нулевая
            // диагональ исходного блока вполне допустима, если ниже есть ненулевой элемент.
            unsigned pivot = k;
            for (unsigned row = k + 1; row < width; ++row)
                if (::fabs(lu[row * width + k]) > ::fabs(lu[pivot * width + k]))
                    pivot = row;
            if (lu[pivot * width + k] == 0)
                return false;
            swaps[k] = static_cast<unsigned char>(pivot);
            if (pivot != k) {
                // Меняем и уже сохранённую часть L, иначе разложение переставленного блока неверно.
                for (unsigned col = 0; col < width; ++col) {
                    const double tmp = lu[k * width + col];
                    lu[k * width + col] = lu[pivot * width + col];
                    lu[pivot * width + col] = tmp;
                }
            }
            for (unsigned row = k + 1; row < width; ++row) {
                double* below = lu + row * width;
                below[k] /= lu[k * width + k];
                for (unsigned col = k + 1; col < width; ++col) {
                    below[col] -= below[k] * lu[k * width + col];
                    if (!(::fabs(below[col]) <= max_finite_double))
                        return false;
                }
            }
        }
        return true;
    }

    // Применяет две треугольные подстановки одного узла. Другие узлы
    // решают свои системы независимо, поэтому обмен через MPI здесь не нужен.
    AERO_HOST_DEVICE void solve_node(MeshIndex node, const double* residual, double* correction) const {
        solve_block(node, residual + node*width, correction + node*width);
    }
    // Решает систему для одной строки, в том числе с правой частью прямого или обратного прохода.
    AERO_HOST_DEVICE void solve_block(MeshIndex node, const double* rhs, double* answer) const {
        switch (width) {
        case 3: return solve_fixed<3>(node,rhs,answer);
        case 4: return solve_fixed<4>(node,rhs,answer);
        default: return solve_fixed<5>(node,rhs,answer);
        }
    }
    // Размер известен из размерности задачи. Промежуточные компоненты одного
    // узла помещаются в регистры при обеих подстановках.
    template<unsigned size>
    AERO_HOST_DEVICE void solve_fixed(MeshIndex node, const double* rhs, double* answer) const {
        const MeshIndex first = node * size;
        const double* lu = factors + first * size;
        const unsigned char* swaps = pivots + first;
        double x[5];
        for (unsigned row = 0; row < size; ++row)
            x[row] = rhs[row];
        // Применяем к правой части те же перестановки, что и к матрице.
        for (unsigned k = 0; k < size; ++k) {
            const double tmp = x[k];
            x[k] = x[swaps[k]];
            x[swaps[k]] = tmp;
        }
        // Прямой ход L y = P r. Единицы на диагонали L в памяти не храним.
        for (unsigned row = 1; row < size; ++row)
            for (unsigned col = 0; col < row; ++col)
                x[row] -= lu[row * size + col] * x[col];
        // Обратный ход U z = y. Справа от диагонали все компоненты z уже известны.
        for (unsigned end = size; end > 0; --end) {
            const unsigned row = end - 1;
            for (unsigned col = row + 1; col < size; ++col)
                x[row] -= lu[row * size + col] * x[col];
            x[row] /= lu[row * size + row];
            answer[row] = x[row];
        }
    }
};
}
}
#endif
