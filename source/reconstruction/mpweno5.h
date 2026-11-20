#ifndef AERO_MPWENO5_H
#define AERO_MPWENO5_H
#include "mp5.h"
#include "weno.h"

namespace aero {
namespace mpweno5 {
// Веса можно привести к произведениям, пока наибольший индикатор меньше этого порога.
constexpr double product_limit = 1.e4;
// При одиночном вызове проверяем диапазон внутри формулы. Для очереди GPU
// он уже известен и компилятор оставляет только нужный способ нормировки.
enum class WeightRange { Automatic, ScaledProducts, DividedWeights };

// В обычном диапазоне полей масштаб 2^56 удерживает произведения весов в double
// даже при beta=0. Один общий знаменатель заменяет шесть частных трёх кандидатов.
struct ProductWeight {
    double a;  // Множитель другого кандидата в общем знаменателе.
    double n;  // Числитель собственного нелинейного веса.
    double dq; // Поправка к малой константе после приведения к общему знаменателю.
    AERO_HOST_DEVICE ProductWeight(double beta, double b, double maximum, double tau, double delta, double h2,
                                   double boost) {
        const double r = b + delta;
        dq = b * r;
        a = (0.2 * maximum * b + tau * (maximum - .8 * b)) * dq;
        n = h2 * r * (b * b + tau * tau) + b * b * boost * beta;
    }
};
// Каждому из трёх полиномов соответствует индикатор гладкости beta.
// Делим исходные веса на общий Dq1*Dq2*Dq3: добавка в знаменателе тогда ровно eps.
struct WeightTerm {
    double numerator, factor;
    AERO_HOST_DEVICE WeightTerm(double beta, double maximum, double tau, double delta, double h2,
                                double boost) {
        constexpr double eps = 1.e-40;
        const double b = beta + eps;
        factor = 0.2 * maximum * b + tau * (maximum - 0.8 * b);
        numerator = h2 * (b + tau * (tau * inverse_value(b))) + boost * b * (beta * inverse_value(b + delta));
    }
};

// Индикаторы гладкости относятся к трём полиномам одной стороны ребра.
// При смене стороны ближние строки меняются ролями, а дальняя строка становится другой.
struct Indicators {
    double first, middle, last;

    template <bool regular> AERO_HOST_DEVICE double maximum() const {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
        if (regular) {
            // Эти три индикатора проверены перед обработкой очереди. Для неотрицательных
            // конечных чисел наибольшему double соответствует наибольшее двоичное слово.
            const mp5::OrderKey a = double_bits(first), b = double_bits(middle);
            const mp5::OrderKey c = double_bits(last), ab = a > b ? a : b;
            return from_double_bits(ab > c ? ab : c);
        }
#endif
        return ::fmax(first, ::fmax(middle, last));
    }

    // Для неотрицательного индикатора граница 10^4 имеет нулевое младшее слово.
    // Сравнение старшего слова точно выбирает способ нормировки.
    AERO_HOST_DEVICE bool regular() const {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
        const unsigned limit = high_word(product_limit);
        return static_cast<unsigned>(high_word(first)) < limit &&
               static_cast<unsigned>(high_word(middle)) < limit &&
               static_cast<unsigned>(high_word(last)) < limit;
#else
        return ::fmax(first, ::fmax(middle, last)) < product_limit;
#endif
    }

    template <bool packed = false> AERO_HOST_DEVICE bool nonlinear() const {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
        if (packed) {
            const mp5::OrderKey a = double_bits(first), b = double_bits(middle);
            const mp5::OrderKey c = double_bits(last);
            // |β₁−β₃| ≥ min(β₁,β₂,β₃) достаточно выполнить хотя бы для одного β.
            // Для положительных конечных значений сравниваем точные двоичные представления.
            // Если общая маска заходит в область особых значений, действует обычный датчик.
            if ((a | b | c) < 0x7ff0000000000000ull) {
                const mp5::OrderKey difference = double_bits(::fabs(first - last));
                return difference >= a || difference >= b || difference >= c;
            }
        }
#endif
        return ::fabs(first - last) >= ::fmin(first, ::fmin(middle, last));
    }
    // На постоянном поле датчик формально выбирает WENO (0 >= 0), но сам след равен нулю.
    template <bool packed = false> AERO_HOST_DEVICE bool all_zero() const {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
        if (packed) {
            const mp5::OrderKey bits = double_bits(first) | double_bits(middle) | double_bits(last);
            return (bits & ~mp5::sign_bit) == 0;
        }
#endif
        return first == 0 && middle == 0 && last == 0;
    }
};

// Значения трёх полиномов до взвешивания. В таком виде GPU может передать работу
// другому потоку: ему нужны только эти три числа и три индикатора, все в double.
struct Polynomials {
    double first, middle, last;

    AERO_HOST_DEVICE static Polynomials from_slopes(double outer, double inner, double center, double opposite) {
        return {inner + (2. / 3.) * (inner - outer),
                inner + (2. / 3.) * (center - inner),
                center - (1. / 3.) * (opposite - center)};
    }
};

// Общая нелинейная формула для обычного вызова и для группы работ на GPU.
// Датчик уже выбрал WENO, и хотя бы один из индикаторов отличен от нуля.
template <WeightRange range = WeightRange::Automatic>
KOKKOS_FORCEINLINE_FUNCTION double weighted_value(Indicators indicators, Polynomials polynomial,
                                                            bool correction) {
    constexpr bool packed = range != WeightRange::Automatic;
    constexpr bool regular = range == WeightRange::ScaledProducts;
    double b1 = indicators.first, b2 = indicators.middle, b3 = indicators.last;
    constexpr double eps = 1.e-40;
    // beta имеет размерность квадрата наклона. После деления весов на Dq1*Dq2*Dq3
    // они однородны степени семь по beta, поэтому добавка к их сумме меняет масштаб как 2^(56*7).
    constexpr double scale = 72057594037927936.; // 2^56
    const double peak = indicators.maximum<packed && regular>();

    const bool products = packed ? regular : peak < product_limit;
    const double unit = products ? scale : 1.;
    b1 *= unit;
    b2 *= unit;
    b3 *= unit;
    const double x = b1 - b2, y = b2 - b3, z = x + y;
    const double delta = ::sqrt(x * x + y * y + z * z);
    const double tau = ::fabs(0.5 * (b1 + b3) - b2);
    const double h = (0.1 * b1 + 0.6 * b2 + 0.3 * b3) + delta + eps * unit;
    // Общий знаменатель chi*(2-chi) уже включён в h² для всех трёх весов.
    const double boost = correction ? delta * (2 * h - delta) : 0;
    const double maximum = (peak + eps) * unit;
    double w1, w2, w3, regularizer = eps;
    if (products) {
        const ProductWeight a(b1, b1 + eps * scale, maximum, tau, delta, h * h, boost);
        const ProductWeight b(b2, b2 + eps * scale, maximum, tau, delta, h * h, boost);
        const ProductWeight c(b3, b3 + eps * scale, maximum, tau, delta, h * h, boost);
        w1 = .1 * a.n * b.a * c.a;
        w2 = .6 * b.n * a.a * c.a;
        w3 = .3 * c.n * a.a * b.a;
        regularizer = (eps * scale * scale * scale * scale * scale * scale * scale) * a.dq * b.dq * c.dq;
    } else {
        const WeightTerm a(b1, maximum, tau, delta, h * h, boost), b(b2, maximum, tau, delta, h * h, boost),
            c(b3, maximum, tau, delta, h * h, boost);
        w1 = .1 * a.numerator * b.factor * c.factor;
        w2 = .6 * b.numerator * a.factor * c.factor;
        w3 = .3 * c.numerator * a.factor * b.factor;
    }
    // Регуляризатор входит только в знаменатель, как в исходной формуле MPWENO5.
    // При β < 10^4 нижнюю границу суммы задаёт eps·2^(56·7)·(eps·2^56)^6.
    // Верхняя граница произведений меньше 2^960. Сумма и 1/сумму помещаются в нормальный double.
    return (w1 * polynomial.first + w2 * polynomial.middle + w3 * polynomial.last)
         * inverse_value(w1 + w2 + w3 + regularizer);
}

// Данные одной стороны после проекции в характеристику. И CPU, и GPU принимают
// одно и то же решение: фиксированный след, линейный полином или нелинейное взвешивание.
struct Side {
    double outer, inner, center, opposite; // Наклоны перечислены от дальней точки к противоположной стороне.
    Indicators indicators;                // Гладкость трёх полиномов именно с этой стороны.
    mp5::Bounds bounds;                   // Допустимый MP-интервал содержит ноль.

    template <bool packed = false> AERO_HOST_DEVICE bool needs_weights() const {
        return !bounds.fixed<packed>() && indicators.nonlinear<packed>() && !indicators.all_zero<packed>();
    }
    AERO_HOST_DEVICE Polynomials polynomials() const {
        return Polynomials::from_slopes(outer, inner, center, opposite);
    }
    // Вызываем, когда нелинейные веса не нужны. Если интервал схлопнулся или поле постоянно,
    // обе ветви ограничителя возвращают ноль. Иначе датчик разрешил линейную схему.
    template <bool packed = false> AERO_HOST_DEVICE double direct_trace() const {
        if (bounds.fixed<packed>() || indicators.all_zero<packed>())
            return bounds.clamp<packed>(0.);
        return bounds.clamp<packed>(mp5::linear(outer, inner, center, opposite));
    }
    AERO_HOST_DEVICE double trace(bool correction) const {
        if (!needs_weights())
            return direct_trace();
        return bounds.clamp(weighted_value(indicators, polynomials(), correction));
    }
};

// Собираем одну сторону из её дальнего наклона и уже известных ближних
// индикаторов. У правой стороны вызывающий код меняет направление и знак кривизны.
template <bool packed = false>
AERO_HOST_DEVICE inline Side prepare_side(double outer, double inner, double center, double opposite,
                                           double alpha, double curvature, WenoIndicators near,
                                           WenoIndicators across) {
    return {outer, inner, center, opposite, {WenoIndicators(inner, outer).outer, near.inner, across.outer},
            mp5::bounds<packed>(outer, inner, center, alpha, curvature)};
}

// Для обеих сторон исходные пять наклонов одни и те же. Сохраняем пару ближних
// индикаторов и центральную кривизну перед тем, как изменить направление шаблона.
template <bool packed = false>
AERO_HOST_DEVICE inline void prepare_pair(double fl, double nl, double c, double nr, double fr, double alpha,
                                          Side& left, Side& right) {
    const WenoIndicators l(c, nl), r(c, nr);
    const double curvature = mp5::curvature<packed>(c - nl, nr - c);
    left = prepare_side<packed>(fl, nl, c, nr, alpha, curvature, l, r);
    right = prepare_side<packed>(fr, nr, c, nl, alpha, -curvature, r, l);
}

// Неограниченный кандидат отдельной стороны; флаг датчика нужен и при нулевом интервале.
AERO_HOST_DEVICE inline double candidate(double fl, double nl, double c, double nr, double b1, double b2,
                                          double b3, bool correction, bool* selected = nullptr,
                                          bool zero_interval = false) {
    const Indicators beta = {b1, b2, b3};
    const bool nonlinear = beta.nonlinear();
    if (selected)
        *selected = nonlinear;
    if (zero_interval || beta.all_zero())
        return 0.;
    if (!nonlinear)
        return mp5::linear(fl, nl, c, nr);
    return weighted_value(beta, Polynomials::from_slopes(fl, nl, c, nr), correction);
}

// Гибрид MP6 использует и ограниченный след, и выбор датчика до ограничения.
template <bool anchored>
AERO_HOST_DEVICE inline double bounded_candidate(double fl, double nl, double c, double nr, double b1,
                                                 double b2, double b3, bool correction, mp5::Bounds bounds,
                                                 bool& selected, double linear_value, double anchor) {
    const Side side = {fl, nl, c, nr, {b1, b2, b3}, bounds};
    // MP6 использует решение датчика при расчёте общей поправки даже тогда,
    // когда нулевой интервал уже определил след. Поэтому проверяем датчик первым.
    selected = side.indicators.nonlinear<true>();
    // При восстановлении по осям сетки исходное значение на этой стороне уже известно.
    // В осях грани нужен только датчик активности для разностной моды MP6.
    if (anchored)
        return anchor;
    if (bounds.fixed<true>() || side.indicators.all_zero<true>())
        return bounds.clamp(0.);
    // Линейный полином уже нужен вызывающей стороне для разностной моды.
    // Дорогие нелинейные веса считаем только для оставшихся сторон.
    if (!selected)
        return bounds.clamp(linear_value);
    return bounds.clamp(weighted_value(side.indicators, side.polynomials(), correction));
}

// Для разреженного шаблона солвер может запросить только одну сторону.
AERO_HOST_DEVICE inline double increment(double fl, double nl, double c, double nr, double alpha,
                                         bool correction) {
    const double curvature = mp5::curvature(c - nl, nr - c);
    return prepare_side(fl, nl, c, nr, alpha, curvature, WenoIndicators(c, nl), WenoIndicators(c, nr))
        .trace(correction);
}

AERO_HOST_DEVICE inline void pair(double fl, double nl, double c, double nr, double fr, double alpha,
                                  bool correction, double& left, double& right) {
    Side l, r;
    prepare_pair(fl, nl, c, nr, fr, alpha, l, r);
    left = l.trace(correction);
    right = r.trace(correction);
}
} // namespace mpweno5
} // namespace aero
#endif
