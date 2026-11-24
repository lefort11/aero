#include "affine_hex.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aero {
namespace viscosity {
namespace {
// Определитель трёх рёбер равен ориентированному шестикратному объёму тетраэдра.
double determinant(const Coordinate &a, const Coordinate &b, const Coordinate &c) {
    return dot3(a,cross3(b,c));
}
// Максимальный по объёму опорный тетраэдр уменьшает ошибки при расчёте весов на вытянутых элементах.
void choose_nodes(const Element &cell, AffineHex &g) {
    double largest = 0.;
    for (unsigned i = 0; i < 5; ++i)
        for (unsigned j = i + 1; j < 6; ++j)
            for (unsigned k = j + 1; k < 7; ++k)
                for (unsigned l = k + 1; l < 8; ++l) {
                    const auto origin = cell.position[i];
                    const double v = std::abs(determinant(cell.position[j] - origin, cell.position[k] - origin,
                                                          cell.position[l] - origin));
                    if (v <= largest)
                        continue;
                    largest = v;
                    g.node[0] = static_cast<unsigned char>(i);
                    g.node[1] = static_cast<unsigned char>(j);
                    g.node[2] = static_cast<unsigned char>(k);
                    g.node[3] = static_cast<unsigned char>(l);
                }
    if (!(largest > 0.))
        throw std::runtime_error("Singular affine AES geometry");
    unsigned next = 4;
    for (unsigned i = 0; i < 8; ++i)
        if (std::find(g.node, g.node + 4, i) == g.node + 4)
            g.node[next++] = static_cast<unsigned char>(i);
}
// Вычисляет коэффициенты интерполяции для аффинного гексаэдра.
void prepare_interpolation(const Element &e, AffineHex &g) {
    const auto origin = e.position[g.node[0]];
    const auto a = e.position[g.node[1]] - origin, b = e.position[g.node[2]] - origin,
               c = e.position[g.node[3]] - origin;
    const double volume = determinant(a, b, c);
    for (unsigned r = 0; r < 4; ++r) {
        const auto x = e.position[g.node[4 + r]] - origin;
        double* w = g.interpolation.row(r);
        w[1] = determinant(x, b, c) / volume;
        w[2] = determinant(a, x, c) / volume;
        w[3] = determinant(a, b, x) / volume;
        w[0] = 1. - w[1] - w[2] - w[3];
    }
}
// Штатный оператор хранит коэффициенты пар. Блок самого узла получается из нулевой суммы строки.
PairMetric block(const Cell &c, const std::vector<PairMetric> &table, unsigned i, unsigned j) {
    PairMetric result{};
    for (unsigned k = 0; k < c.count; ++k)
        if (k != i && (i == j || k == j)) {
            const auto &pair = table[c.pair_index(i, k)];
            const double sign = i == j ? -1. : 1.;
            result.heat += sign * pair.heat;
            for (unsigned m = 0; m < 3; ++m)
                for (unsigned d = 0; d < 3; ++d)
                    result.stress(m, d) += sign * (i > k ? pair.stress(d, m) : pair.stress(m, d));
        }
    return result;
}
}
// Выделяет из штатной геометрии AES часть оператора для отклонений от линейного поля.
AffineHex make_affine_hex(const Element &e, const Cell &c, const MeanGradient &mean,
                          const std::vector<PairMetric> &table) {
    AffineHex g{};
    choose_nodes(e, g);
    prepare_interpolation(e, g);
    g.volume = mean.volume;
    for (unsigned i = 0; i < 8; ++i)
        g.ordered[g.node[i]] = static_cast<unsigned char>(i);
    for (unsigned i = 0; i < 8; ++i)
        for (unsigned d = 0; d < 3; ++d)
            g.basis(d, i) = mean.basis[g.node[i]][d];
    // Отделяем постоянную по элементу часть производной. Её вклад посчитаем через один средний градиент.
    for (unsigned r = 0; r < 4; ++r)
        for (unsigned s = r; s < 4; ++s) {
            const unsigned i = g.node[4 + r], j = g.node[4 + s];
            const auto pair = block(c, table, i, j);
            const auto &a = mean.basis[i];
            const auto &b = mean.basis[j];
            const double trace = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            for (unsigned m = 0; m < 3; ++m)
                for (unsigned d = 0; d < 3; ++d) {
                    if (r == s && m > d)
                        continue;
                    const double average = (m == d ? trace : 0.) + a[d] * b[m] - (2. / 3.) * a[m] * b[d];
                    g.stress[AffineHex::upper_index(3 * r + m, 3 * s + d, 12)] =
                        pair.stress(m, d) - mean.volume * average;
                }
        }
    return g;
}
}
}
