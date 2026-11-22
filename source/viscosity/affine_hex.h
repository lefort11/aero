#pragma once
#include "grad.h"

namespace aero {
namespace viscosity {
// Локальные номера в этой записи начинаются с четырёх опорных узлов.
// Последующие четыре узла задают отклонения от линейного поля.
struct AffineHex {
    unsigned char node[8], ordered[8]; // Опорный порядок и обратное отображение из порядка сетки.
    FixedMatrix<double,4,4> interpolation;        // Значение линейного поля в добавочном узле по четырём опорным.
    FixedMatrix<double,3,8> basis;                // Средние по объёму производные функций восьми узлов.
    double volume;
    double stress[78]; // Верхний треугольник симметричной матрицы 12 на 12.

    // Храним каждую связь один раз. Для обратного направления читаем ту же запись.
    AERO_HOST_DEVICE static unsigned upper_index(unsigned i, unsigned j, unsigned n) {
        const unsigned row = i < j ? i : j;
        const unsigned column = i < j ? j : i;
        return row * (2 * n - 1 - row) / 2 + column;
    }
    // Строка оператора, вычитающего значение линейного поля из значения в добавочном узле.
    // В своём добавочном узле коэффициент равен единице. В опорных узлах это веса с обратным знаком.
    AERO_HOST_DEVICE double departure_weight(unsigned r, unsigned i) const {
        return i < 4 ? -interpolation(r, i) : double(i == 4 + r);
    }
    // В неявном решателе нужен отдельный блок по паре исходных вершин.
    // Сначала берём постоянный по элементу градиент, затем добавляем вклад отклонений.
    AERO_HOST_DEVICE PairMetric metric(unsigned first, unsigned second, double pair_heat) const {
        const unsigned i = ordered[first], j = ordered[second];
        double trace = 0.;
        for (unsigned d = 0; d < 3; ++d)
            trace += basis(d, i) * basis(d, j);
        PairMetric pair{};
        pair.heat = pair_heat;
        for (unsigned m = 0; m < 3; ++m)
            for (unsigned k = 0; k < 3; ++k) {
                double value = volume * ((m == k ? trace : 0.) + basis(k, i) * basis(m, j) -
                                         (2. / 3.) * basis(m, i) * basis(k, j));
                const unsigned r0 = i < 4 ? 0 : i - 4, r1 = i < 4 ? 4 : i - 3;
                const unsigned s0 = j < 4 ? 0 : j - 4, s1 = j < 4 ? 4 : j - 3;
                for (unsigned r = r0; r < r1; ++r)
                    for (unsigned s = s0; s < s1; ++s)
                        value += departure_weight(r, i) * stress[upper_index(3 * r + m, 3 * s + k, 12)] *
                                 departure_weight(s, j);
                pair.stress(m, k) = value;
            }
        return pair;
    }
};
// Выделяет из штатной геометрии AES часть оператора для отклонений от линейного поля.
AffineHex make_affine_hex(const Element &, const Cell &, const MeanGradient &, const std::vector<PairMetric> &);

}
}
