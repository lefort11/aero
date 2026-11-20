#ifndef AERO_WENO_H
#define AERO_WENO_H
#include "../parallel/kernel_math.h"
#include <string.h>

// Скалярная формула WENO3/WENO5 общая для CPU и CUDA и пригодна для вызова внутри ядра.

namespace aero {
// Деление на три с округлением binary64. На GPU заранее известны обе части числа 1/3.
AERO_HOST_DEVICE inline double weno_third(double x) {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    // Для нормальных x/3 расстояние до середины между соседними double не меньше 1/6 шага.
    // Ошибка младшей части значительно меньше этого расстояния. FMA даёт то же округление,
    // что и деление. Для очень малых x произведение с младшей частью могло бы потерять точность.
    const unsigned exponent = (static_cast<unsigned>(high_word(x)) >> 20) & 2047u;
    if (exponent >= 80 && exponent < 2047) {
        constexpr double high = 1.0 / 3.0;
        constexpr double low = high / 18014398509481984.0; // (1/3 − high), округлённое к double.
        return ::fma(x, high, x * low);
    }
#endif
    return x / 3.0;
}

// Масштабируем все наклоны и корень ε одной степенью двойки. Схема JS однородна:
// после вычисления весов возвращаем размерность приращения обратным масштабированием.
struct WenoScale {
    double unit;       // Множитель для возвращения к размерному приращению.
    double inverse;    // Обратная степень двойки для всех наклонов одной характеристики.

    AERO_HOST_DEVICE explicit WenoScale(double largest) {
        // В binary64 старшие биты положительного числа содержат порядок со смещением 1023.
        // Выбираем 2^e и 2^(-e) так, чтобы обе величины оставались нормальными даже на краях диапазона.
        unsigned long long bits;
        static_assert(sizeof(bits) == sizeof(double), "WENO requires binary64");
        memcpy(&bits, &largest, sizeof(bits));
        unsigned exponent = static_cast<unsigned>(bits >> 52);
        if (exponent < 1)
            exponent = 1;
        if (exponent > 2045)
            exponent = 2045;
        const unsigned long long unit_bits = static_cast<unsigned long long>(exponent) << 52;
        const unsigned long long inverse_bits = static_cast<unsigned long long>(2046 - exponent) << 52;
        memcpy(&unit, &unit_bits, sizeof(unit));
        memcpy(&inverse, &inverse_bits, sizeof(inverse));
    }
    // Переводит и наклоны, и корень ε в одни и те же безразмерные единицы.
    AERO_HOST_DEVICE double normalize(double value) const {
        return value * inverse;
    }
};

// Число используемых строк у нелинейной стороны. Для третьего порядка
// нужны центральный и свой ближний наклоны. Пятый добавляет противоположный и дальний.
// Нулевой порядок вызывающая сторона обрабатывает по отдельной линейной строке.
AERO_HOST_DEVICE constexpr unsigned weno_slope_count(unsigned order) { return order == 5 ? 4u : 2u; }

// На ребре два индикатора используют одну пару наклонов. Если d = near − center,
// разложение квадратов Jiang–Shu даёт общую часть center² + 4d²/3 и поправку ±center·d.
struct WenoIndicators {
    double outer, inner; // Кандидат, смещённый наружу, и кандидат, содержащий оба соседних узла.
    AERO_HOST_DEVICE WenoIndicators(double center, double near) {
        const double difference = near - center;
        const double base = (4.0 / 3.0) * difference * difference + center * center;
        const double cross = center * difference;
        outer = base - cross;
        inner = base + cross;
    }
};

// Выражаем обратный знаменатель через наименьший в этой тройке (или паре).
// Если минимум нулевой, вклад сохраняют только кандидаты с таким же нулевым знаменателем.
KOKKOS_FORCEINLINE_FUNCTION double relative_weno_weight(double minimum, double denominator) {
    return minimum == 0 ? double(denominator == 0) : minimum / denominator;
}

// Возвращает ненормированные веса для двух или трёх кандидатов. Делитель у всех один.
// После сокращения произведения знаменателей остаются только умножения и одно обращение суммы.
template <unsigned order, bool safe_products>
KOKKOS_FORCEINLINE_FUNCTION void weno_weights(double a, double b, double c, double* weights) {
    double first = b, second = a, third = 0;
    const double minimum = order == 5 ? ::fmin(a, ::fmin(b, c)) : ::fmin(a, b);
    if (safe_products || minimum >= 1.e-18) {
        if (order == 5) {
            // Общий множитель a·b·c превращает три обратных знаменателя в b·c, a·c, a·b.
            third = a * b;
            first *= c;
            second *= c;
        }
    } else {
        // На границах диапазона double нормируем отношения к самому малому знаменателю.
        // Если он равен нулю, вес получают только кандидаты с нулевым индикатором.
        first = relative_weno_weight(minimum, a);
        second = relative_weno_weight(minimum, b);
        if (order == 5)
            third = relative_weno_weight(minimum, c);
    }
    // Линейные веса пропорциональны (2,1) для WENO3 и (3,6,1) для WENO5.
    weights[0] = (order == 5 ? 3 : 2) * first * first;
    weights[1] = (order == 5 ? 6 : 1) * second * second;
    weights[2] = third * third;
}

template <bool safe_products>
KOKKOS_FORCEINLINE_FUNCTION double weno_three(double center, double near, double epsilon) {
    double w[3];
    weno_weights<3, safe_products>(epsilon + center * center, epsilon + near * near, 0, w);
    const double inverse = 1 / (w[0] + w[1]);
    if (safe_products)
        return center + (w[1] * inverse) * (near - center);
    return (w[0] * inverse) * center + (w[1] * inverse) * near;
}

// Смешивает три полинома формулы (9). Веса нормируются до умножения на наклоны,
// чтобы малое размерное приращение не исчезло при умножении на малый общий множитель.
template <bool safe_products>
KOKKOS_FORCEINLINE_FUNCTION double weno_five(double center, double near, double opposite, double far,
                                                      double a, double b, double c) {
    double w[3];
    weno_weights<5, safe_products>(a, b, c, w);
    const double inverse = 1 / (w[0] + w[1] + w[2]);
    if (!safe_products) {
        // После масштабирования крайних величин возможны сильно различающиеся
        // кандидаты. Смешиваем их напрямую, чтобы сохранить слабый вклад.
        return weno_third((w[0] * inverse) * (4 * center - opposite) +
                          (w[1] * inverse) * (2 * center + near) +
                          (w[2] * inverse) * (5 * near - 2 * far));
    }
    // У всех трёх кандидатов выделяем центральный наклон. Веса суммируются в
    // единицу, поэтому смешиваем только отклонения, а общий наклон прибавляем один раз.
    const double change = -(w[0] * inverse) * (opposite - center) +
                           ((w[1] + 3 * w[2]) * inverse) * (near - center) -
                           (2 * w[2] * inverse) * (far - near);
    return center + weno_third(change);
}

// У парной реконструкции показатели двух ближних наклонов уже вычислены для обеих сторон.
// Дальний показатель относится только к текущей стороне. Порядок сложений сохраняем и здесь.
template <bool safe_products>
KOKKOS_FORCEINLINE_FUNCTION double weno_five_side(double center, double near, double opposite, double far,
                                                 double epsilon, const WenoIndicators& near_indicator,
                                                 const WenoIndicators& opposite_indicator) {
    return weno_five<safe_products>(center, near, opposite, far, epsilon + opposite_indicator.outer,
                                    epsilon + near_indicator.inner, epsilon + WenoIndicators(near, far).outer);
}

template <unsigned order, bool safe_products>
KOKKOS_FORCEINLINE_FUNCTION double weighted_weno_increment(const double* d, double epsilon) {
    if (order == 3)
        return weno_three<safe_products>(d[0], d[1], epsilon);
    return weno_five_side<safe_products>(d[0], d[1], d[2], d[3], epsilon,
                                         WenoIndicators(d[0], d[1]), WenoIndicators(d[0], d[2]));
}

// Вне обычного диапазона выбираем единицы отдельно для каждой характеристики.
// Этот путь нужен при очень большой амплитуде или малом ε, в том числе при ε = 0.
template <unsigned order>
AERO_HOST_DEVICE inline double scaled_weno_increment(const double* slopes, double epsilon_scale, double scale) {
    if (scale == 0)
        return 0;
    const WenoScale units(scale);
    double normalized[4] = {};
    for (unsigned k = 0; k < weno_slope_count(order); ++k)
        normalized[k] = units.normalize(slopes[k]);
    const double epsilon = units.normalize(epsilon_scale);
    return units.unit * weighted_weno_increment<order, false>(normalized, epsilon * epsilon);
}

// При h·√ε ≥ 1e−34 квадрат произведения двух знаменателей ≥ 1e−272.
// Верхняя граница защищает те же произведения от переполнения.
AERO_HOST_DEVICE inline bool regular_weno_scale(double maximum, double epsilon) {
    return epsilon >= 1.e-34 && epsilon <= 32 && maximum <= 32;
}

template <unsigned order>
AERO_HOST_DEVICE inline double side_weno_increment(const double* slopes, double epsilon) {
    double scale = epsilon;
    for (unsigned k = 0; k < weno_slope_count(order); ++k)
        scale = ::fmax(scale, ::fabs(slopes[k]));
    if (regular_weno_scale(scale, epsilon))
        return weighted_weno_increment<order, true>(slopes, epsilon * epsilon);
    return scaled_weno_increment<order>(slopes, epsilon, scale);
}

// Порядок разреженной строки может понижаться на одной из сторон у границы.
template <unsigned fixed_order = 0>
AERO_HOST_DEVICE inline double weno_increment2(unsigned row_order, const double* slopes, double epsilon_scale) {
    return (fixed_order ? fixed_order : row_order) == 5 ? side_weno_increment<5>(slopes, epsilon_scale)
                                                       : side_weno_increment<3>(slopes, epsilon_scale);
}

// Восстанавливает вектор на одной подготовленной стороне ребра. Веса Jiang–Shu зависят от текущего решения.
// Формула одинакова на CPU и CUDA.
template <class SideInput, class Formula>
AERO_HOST_DEVICE inline void reconstruct_slope_rows(const SideInput& input, unsigned long long components,
                                                    double* output, unsigned long long output_stride,
                                                    const Formula& formula) {
    const unsigned order = input.order();
    for (unsigned long long c = 0; c < components; ++c) {
        if (!order) {
            output[c * output_stride] = input.linear(c);
            continue;
        }
        double slopes[4] = {0, 0, 0, 0};
        for (unsigned k = 0; k < weno_slope_count(order); ++k)
            slopes[k] = input.slope(k, c);
        output[c * output_stride] = formula(order, slopes, input.epsilon_scale());
    }
}

struct WenoFormula {
    AERO_HOST_DEVICE double operator()(unsigned order, const double* slopes, double epsilon) const {
        return weno_increment2(order, slopes, epsilon);
    }
};

// Сохраняем прямой интерфейс WENO для библиотечных вызовов.
template <class SideInput>
AERO_HOST_DEVICE inline void reconstruct_weno_increment2(const SideInput& input, unsigned long long components,
                                                         double* output, unsigned long long output_stride = 1) {
    reconstruct_slope_rows(input, components, output, output_stride, WenoFormula{});
}

#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY)
class Mesh;
class ReconstructionStencil;
struct ReconstructionOptions;

// Готовит геометрию при options.scheme равном Weno3 или Weno5. Общие типы настроек и шаблона объявлены в reconstruction.h.
ReconstructionStencil prepare_weno(const Mesh& mesh, const ReconstructionOptions& options);
#endif
} // namespace aero
#endif
