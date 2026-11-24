#ifndef AERO_MP5_H
#define AERO_MP5_H
#include "../parallel/kernel_math.h"

namespace aero {
// Эйлер и SSPRK3 имеют C_ssp = 1. Вычисляем alpha один раз по заданному CFL.
double mp5_alpha_from_cfl(double cfl);
// Для неявного шага MP5 и MPWENO5 берут alpha = 4 при любом допустимом CFL.
// Решатель выбирает коэффициент один раз перед построением шаблонов.
double mp5_alpha_for_time_scheme(double cfl, bool implicit_time);
// Самостоятельный вызов реконструкции может передать alpha напрямую.
void validate_mp5_alpha(double alpha);

// У пустого представления enabled равен false. Нулевой alpha допустим для включённой MP5.
struct Mp5Settings {
    bool enabled;
    bool weno;       // Выбирает кандидата MPWENO5 перед общим ограничителем.
    bool correction; // Включает антидиссипативную поправку MPWENO5.
    double alpha;
};

namespace mp5 {
// Оптимальный линейный кандидат и короткая граница общие для обеих MP-схем.
AERO_HOST_DEVICE inline double linear(double outer_slope, double inner_slope, double center, double opposite) {
    return -0.1 * opposite + 0.8 * center + (11.0 / 30.0) * inner_slope - (1.0 / 15.0) * outer_slope;
}
// Ограничивает значение заданным допустимым интервалом.
AERO_HOST_DEVICE inline double bound(double inner_slope, double center, double alpha) {
    const double upwind = alpha * inner_slope;
    return center > 0 ? ::fmax(0., ::fmin(center, upwind)) : ::fmin(0., ::fmax(center, upwind));
}

// Проверяем кандидата пятого порядка по дешёвому MP-критерию из исходной формулы.
// Все наклоны имеют размерность удвоенного приращения от узла до середины ребра.
struct Candidate {
    double value;    // Удвоенное приращение по линейному полиному пятого порядка.
    bool admissible; // Можно ли принять этот полином без расчёта полного MP-интервала.
    // Формирует высокопорядковый кандидат и допустимые приращения ограничителя MP5.
    AERO_HOST_DEVICE Candidate(double outer_slope, double inner_slope, double center, double opposite,
                               double alpha) {
        value = linear(outer_slope, inner_slope, center, opposite);
        admissible = value * (value - bound(inner_slope, center, alpha)) < 1.e-10;
    }
};

#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
// Код конечного double, монотонный по его численному значению. Для положительных
// чисел меняем знаковый бит; для отрицательных обращаем все биты (порядок модулей обратный).
// На GPU сравнения границ тогда не занимают арифметические устройства double.
using OrderKey = unsigned long long;
constexpr OrderKey sign_bit = 1ull << 63;
// Преобразует вещественное значение в упорядочиваемый целочисленный ключ.
KOKKOS_INLINE_FUNCTION OrderKey order_key(double value) {
    const OrderKey bits = double_bits(value);
    return bits ^ ((bits & sign_bit) ? ~0ull : sign_bit);
}
// Восстанавливает вещественное значение из упорядоченного ключа.
KOKKOS_INLINE_FUNCTION double ordered_value(OrderKey key) {
    return from_double_bits(key ^ ((key & sign_bit) ? sign_bit : ~0ull));
}
#endif

// Minmod четырёх кривизн. При общем знаке только 4*малая-большая
// может дополнительно сузить модуль результата.
template <bool packed = false>
AERO_HOST_DEVICE inline double curvature(double a, double b) {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    if (packed) {
        const OrderKey bits_a = double_bits(a), bits_b = double_bits(b);
        const OrderKey abs_a = bits_a & ~sign_bit, abs_b = bits_b & ~sign_bit;
        // У конечных модулей порядок битов совпадает с порядком величин. При NaN
        // или бесконечности используем стандартные правила fmin/fmax ниже.
        constexpr OrderKey infinity = 0x7ff0000000000000ull;
        if (abs_a < infinity && abs_b < infinity) {
            if (((bits_a ^ bits_b) & sign_bit) || !abs_a || !abs_b)
                return 0.;
            const OrderKey small = abs_a < abs_b ? abs_a : abs_b;
            const OrderKey large = abs_a < abs_b ? abs_b : abs_a;
            // Саму кривизну 4·small − large по-прежнему считаем в double.
            const OrderKey cap = double_bits(4 * from_double_bits(small) - from_double_bits(large));
            const OrderKey magnitude = (cap & sign_bit) ? 0 : (small < cap ? small : cap);
            return from_double_bits((bits_a & sign_bit) | magnitude);
        }
    }
#endif
    if ((a > 0 && b > 0) || (a < 0 && b < 0)) {
        const double small = ::fmin(::fabs(a), ::fabs(b));
        const double large = ::fmax(::fabs(a), ::fabs(b));
        return ::copysign(::fmax(0., ::fmin(small, 4 * small - large)), a);
    }
    return 0;
}

// Пересекаем два допустимых интервала и ограничиваем удвоенное приращение.
struct Bounds {
    double low, high;
    // Включает принятое приращение своей стороны в интервал для следующей поправки.
    AERO_HOST_DEVICE void include(double accepted) {
        low = ::fmin(low, accepted);
        high = ::fmax(high, accepted);
    }
    // Оба конца интервала включают ноль. Если оба равны нулю, значение фиксировано.
    template <bool packed = false> AERO_HOST_DEVICE bool fixed() const {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
        if (packed) {
            const OrderKey bits = double_bits(low) | double_bits(high);
            return (bits & ~sign_bit) == 0;
        }
#endif
        return low == high;
    }
    // Ограничивает реконструированное значение допустимыми границами MP.
    template <bool packed = false> AERO_HOST_DEVICE double clamp(double value) const {
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
        if (packed) {
            const OrderKey bits = double_bits(value), magnitude = bits & ~sign_bit;
            // Для конечного значения достаточно одной границы: нижней для
            // отрицательного следа и верхней для положительного. Сравнение точное.
            if (magnitude < 0x7ff0000000000000ull && magnitude != 0) {
                const double limit = (bits & sign_bit) ? low : high;
                const OrderKey bound = double_bits(limit) & ~sign_bit;
                return magnitude <= bound ? value : limit;
            }
        }
#endif
        return ::fmin(high, ::fmax(low, value));
    }
};
// Вычисляет интервал монотонно допустимых значений на стороне ребра.
template <bool packed = false, bool packed_curvature = packed>
AERO_HOST_DEVICE inline Bounds bounds(double outer_slope, double inner_slope, double center, double alpha,
                                      double central_curvature) {
    const double gc = 2 * center, ul = 2 * alpha * inner_slope;
    const double md = center - central_curvature;
    // Кривизну для LC и пересечение готовых границ можно вычислять независимо.
    // Гибрид сохраняет арифметику боковой кривизны при ускоренном сравнении границ.
    const double lc = inner_slope + (8.0 / 3.0) * curvature<packed_curvature>(
        inner_slope - outer_slope, center - inner_slope);
#ifndef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    if (packed) {
        // В обычном диапазоне все четыре границы конечны (alpha не превышает 4).
        // Вне него оставляем fmin/fmax, в том числе при переполнении промежуточной разности.
        const auto moderate = [](double value) {
            return (static_cast<unsigned>(high_word(value)) & 0x7fffffffu) < 0x5cd00000u; // |x| < 2^462
        };
        if (moderate(outer_slope) && moderate(inner_slope) && moderate(center) && moderate(central_curvature)) {
            const OrderKey x = order_key(gc), y = order_key(md), u = order_key(ul), v = order_key(lc);
            const OrderKey min1 = x < y ? x : y, min2 = u < v ? u : v;
            const OrderKey max1 = x > y ? x : y, max2 = u > v ? u : v;
            // Оба исходных интервала содержат ноль. В пересечении выбираем
            // большую нижнюю границу и меньшую верхнюю, затем включаем ноль.
            const OrderKey lo = min1 > min2 ? min1 : min2, hi = max1 < max2 ? max1 : max2;
            const OrderKey zero = order_key(0.);
            return {ordered_value(lo < zero ? lo : zero), ordered_value(hi > zero ? hi : zero)};
        }
    }
#endif
    const double low = ::fmax(::fmin(0., ::fmin(gc, md)), ::fmin(0., ::fmin(ul, lc)));
    const double high = ::fmin(::fmax(0., ::fmax(gc, md)), ::fmax(0., ::fmax(ul, lc)));
    return {low, high};
}
// Применяет ограничитель монотонности к высокопорядковому кандидату.
AERO_HOST_DEVICE inline double limit(double value, double outer_slope, double inner_slope, double center,
                                     double alpha, double central_curvature) {
    return bounds(outer_slope, inner_slope, center, alpha, central_curvature).clamp(value);
}

// Вычисляет ограниченное приращение MP5 относительно узлового значения.
AERO_HOST_DEVICE inline double increment(double outer_slope, double inner_slope, double center, double opposite,
                                         double alpha) {
    const Candidate candidate(outer_slope, inner_slope, center, opposite, alpha);
    if (candidate.admissible)
        return candidate.value;
    return limit(candidate.value, outer_slope, inner_slope, center, alpha,
                 curvature(center - inner_slope, opposite - center));
}

// В полном двумерном шаблоне наклоны уже в регистрах. Левой и правой стороне
// нужна одна центральная кривизна с противоположными знаками.
AERO_HOST_DEVICE inline void pair(double fl, double nl, double center, double nr, double fr, double alpha,
                                  double &left, double &right) {
    const Candidate l(fl, nl, center, nr, alpha), r(fr, nr, center, nl, alpha);
    left = l.value;
    right = r.value;
    // На гладком ребре оба полинома обычно проходят короткую проверку.
    // Только отклонённому следу нужны кривизна и полный интервал MP.
    if (l.admissible && r.admissible)
        return;
    const double shared = curvature(center - nl, nr - center);
    if (!l.admissible)
        left = limit(left, fl, nl, center, alpha, shared);
    if (!r.admissible)
        right = limit(right, fr, nr, center, alpha, -shared);
}
} // namespace mp5
} // namespace aero
#endif
