#ifndef AERO_MP6_HYBRID_H
#define AERO_MP6_HYBRID_H
#include "mpweno5.h"

namespace aero {
namespace mp6 {
// В базисе Роу первыми идут левая акустическая и энтропийная характеристики,
// последней — правая акустическая. Между ними расположены d−1 поперечных характеристик.
// В локальном векторе состояния или потока те же позиции занимают
// касательные компоненты импульса, а последнюю позицию занимает энергия.
enum class MeanLimit { Shared, Transverse };
AERO_HOST_DEVICE inline MeanLimit face_mean_limit(unsigned component, unsigned width) {
    return component >= 2 && component + 1 < width ? MeanLimit::Transverse : MeanLimit::Shared;
}

// Ниже этого масштаба шум одной характеристики должен лишь постепенно влиять
// на поправку к остальным компонентам. Порог задан в тех же единицах, что и dS.
constexpr double mean_tolerance = 1.e-8;

// Какую долю среднего сдвига допускает эта характеристика при выборе общего веса ребра.
// Для значимых сдвигов сохраняем исходное отношение. Вблизи нуля нормируем
// только запрещённую часть на фиксированный масштаб, поэтому деления на малый dS нет.
AERO_HOST_DEVICE inline double mean_weight(double ds, double low, double high) {
    if (ds == 0. || (ds >= low && ds <= high))
        return 1.;
    const double limit = ds < low ? low : high;
    const double magnitude = ::fabs(ds);
    if (magnitude >= mean_tolerance)
        return ::fmax(0., ::fmin(1., limit * inverse_value(ds)));

    // Ограничение может уменьшать сдвиг только в сторону нуля.
    const double allowed = ::fmax(0., ::fmin(magnitude, ds > 0. ? limit : -limit));
    return 1. - (magnitude - allowed) / mean_tolerance;
}
// Первый проход сохраняет допустимые интервалы двух связанных мод.
// Второй проход уже знает общий для ребра коэффициент ограничения среднего наклона.
struct Correction {
    double ds, dd;       // Требуемые поправки к среднему и полуразности приращений.
    double active;       // Доля сторон (0, 1/2 или 1), участвующих в первой поправке.
    double smin, smax;   // Допустимое пересечение для среднего с учётом обеих сторон.
    double min1, max1;   // Границы для s + 2d.
    double min2, max2;   // Границы для s - 2d.

    // Наибольший положительный сдвиг среднего, совместимый с ограничениями
    // на s+2d и s−2d. Для нижней границы меняем знаки всех трёх аргументов.
    AERO_HOST_DEVICE static double upper_mean(double first, double second, double difference) {
        if (difference == 0.)
            return ::fmin(first, second);
        return ::fmin(.5*(first+second), ::fmin(first - 2.*::fmin(0., difference),
                                                second + 2.*::fmax(0., difference)));
    }

    // Рассчитывает цель и вклад данной компоненты в общий ограничитель.
    template <bool bounded = false>
    AERO_HOST_DEVICE double prepare(double left, double right, double raw_l, double raw_r, bool skip_l,
                                    bool skip_r, double center, double h, mp5::Bounds l, mp5::Bounds r) {
        active = 1 - .5 * (double(skip_l) + double(skip_r));
        ds = h * (1 + (1 - h) * active) * (center - .5 * (left + right));
        dd = .5 * h * active * ((raw_l - left) - (raw_r - right));
        const double lm = left - l.low, lp = l.high - left, rm = right - r.low, rp = r.high - right;
        min1 = -(1.5 * lm + .5 * rp);
        max1 = 1.5 * lp + .5 * rm;
        min2 = -(1.5 * rm + .5 * lp);
        max2 = 1.5 * rp + .5 * lm;
        if (bounded) {
            // Оба исходных значения уже входят в допустимые интервалы. Нулевой
            // сдвиг разрешён, поэтому вычисляем лишь границу в направлении ds.
            const bool negative = ds < 0.;
            const double bound = negative ? upper_mean(-min1, -min2, -dd) : upper_mean(max1, max2, dd);
            const double allowed = ::fmin(::fabs(ds), bound);
            smin = negative ? -allowed : 0.;
            smax = negative ? 0. : allowed;
        } else {
            // MP5 иногда принимает линейное значение по своей первой проверке.
            // Для него ноль может лежать вне интервала, поэтому нужны обе границы.
            smin = ::fmax(::fmin(0., ds), -upper_mean(-min1, -min2, -dd));
            smax = ::fmin(::fmax(0., ds), upper_mean(max1, max2, dd));
        }
        return mean_weight(ds, smin, smax);
    }

    // Средний сдвиг s уже выбран. Пересекаем ограничения для s+2d и s−2d
    // с отрезком между нулём и целью dd. Ближайшая к dd точка даёт разностную поправку.
    KOKKOS_FORCEINLINE_FUNCTION double limited_difference(double s) const {
        const double dlo = ::fmax(::fmin(0., dd), ::fmax(.5 * (min1 - s), .5 * (s - max2)));
        const double dhi = ::fmin(::fmax(0., dd), ::fmin(.5 * (max1 - s), .5 * (s - min2)));
        return ::fmax(dlo, ::fmin(dhi, dd));
    }

    // Поперечную компоненту ограничиваем по её собственному интервалу. Обе акустические
    // характеристики и энтропийная используют минимальный вес по всем компонентам ребра.
    AERO_HOST_DEVICE void apply(MeanLimit limit, double weight, double &left, double &right) const {
        const bool independent = limit == MeanLimit::Transverse;
        const double proposed = independent ? ds : weight * ds;
        // Общий вес смягчён только для малых сдвигов. Каждая такая компонента по-прежнему
        // обязана выдержать местное ограничение, прежде чем мы выберем разностную моду.
        const double s = independent || ::fabs(ds) < mean_tolerance
                       ? ::fmax(smin, ::fmin(smax, proposed)) : proposed;
        if (dd == 0 && ds == s && s >= smin && s <= smax) {
            left += s;
            right += s;
            return;
        }
        // При выбранном s обе границы дают разрешённый интервал для d. Берём ближайшее к цели dd.
        const double d = limited_difference(s);
        const double delta = ds * ds + dd * dd;
        // При существенном ограничении уменьшаем вклад гибрида в пользу исходного значения на грани.
        double keep = 1;
        if (delta >= 1.e-16 && active < 1) {
            const double es = ds - s, ed = dd - d;
            keep -= (1 - active) * ::fmin(1., (es * es + ed * ed) * inverse_value(delta));
        }
        left += keep * (s + d);
        right += keep * (s - d);
    }
};

// Одна компонента в локальных осях или в базисе Роу. Вычисляем оба ограничения до выбора общего веса по компонентам.
template <bool weno, bool anchored>
KOKKOS_FORCEINLINE_FUNCTION double prepare_wave(double fl, double l, double c, double r, double fr,
                                                        Mp5Settings settings, double h, Correction &correction,
                                                        double &left, double &right) {
    const double alpha = settings.alpha;
    const mp5::Candidate raw_l(fl, l, c, r, alpha), raw_r(fr, r, c, l, alpha);
    bool skip_l = !raw_l.admissible, skip_r = !raw_r.admissible;
    // В центре обе стороны видят одну кривизну с противоположным знаком.
    // На GPU сравниваем модули конечных double по их двоичному представлению.
    const double shared = mp5::curvature<true>(c - l, r - c);
    // У левого и правого полиномов свои боковые кривизны. Их считаем обычной
    // формулой LC, затем на GPU пересекаем готовые интервалы через упорядоченные ключи.
    auto bl = mp5::bounds<true, false>(fl, l, c, alpha, shared);
    auto br = mp5::bounds<true, false>(fr, r, c, alpha, -shared);
    if (weno) {
        const WenoIndicators li(c, l), ri(c, r);
        // Нулевой интервал фиксирует значение на грани даже на скачке. Для общего веса
        // сохраняем решение датчика WENO на каждой стороне.
        left = mpweno5::bounded_candidate<anchored>(fl, l, c, r, WenoIndicators(l, fl).outer, li.inner, ri.outer,
                                          settings.correction, bl, skip_l, raw_l.value, anchored ? left : 0.);
        right = mpweno5::bounded_candidate<anchored>(fr, r, c, l, WenoIndicators(r, fr).outer, ri.inner, li.outer,
                                           settings.correction, br, skip_r, raw_r.value, anchored ? right : 0.);
    } else if (!anchored) {
        left = skip_l ? bl.clamp(raw_l.value) : raw_l.value;
        right = skip_r ? br.clamp(raw_r.value) : raw_r.value;
    }
    if (anchored) {
        // Компонента исходного приращения после поворота может выйти за локальный MP-интервал.
        // Включаем исходную точку в интервал перед расчётом ограничений на среднее
        // и полуразность. Тогда нулевая добавка допустима даже при таком отклонении.
        bl.low = ::fmin(bl.low, left);
        bl.high = ::fmax(bl.high, left);
        br.low = ::fmin(br.low, right);
        br.high = ::fmax(br.high, right);
    }
    return correction.prepare<weno || anchored>(left, right, raw_l.value, raw_r.value, skip_l, skip_r, c, h, bl, br);
}

// Каждая строка хранит разности всех компонент. Порядок строк — центральная,
// ближняя слева, ближняя справа, дальняя слева, дальняя справа. Сначала собираем
// минимальный MP-вес по всем компонентам, затем применяем его к каждой по её назначению.
// Обычно left и right получают полные приращения. При anchored=true они уже
// содержат исходные приращения в осях грани и на выходе получают только поправки.
template <bool weno, bool anchored = false, unsigned capacity>
AERO_HOST_DEVICE inline void reconstruct(const double (*slopes)[capacity], unsigned width, Mp5Settings settings,
                                         double h, double *left, double *right) {
    Correction corrections[capacity];
    double weight = 1;
    for (unsigned k = 0; k < width; ++k) {
        const double c = slopes[0][k], l = slopes[1][k], r = slopes[2][k], fl = slopes[3][k], fr = slopes[4][k];
        const double t = prepare_wave<weno, anchored>(fl, l, c, r, fr, settings, h, corrections[k], left[k], right[k]);
        weight = ::fmin(weight, t);
        // Исходное приращение уже выражено в осях грани. Возвращаем только
        // добавку: вызывающий код повернёт её обратно и прибавит к исходному приращению.
        if (anchored)
            left[k] = right[k] = 0.;
    }
    for (unsigned k = 0; k < width; ++k)
        corrections[k].apply(face_mean_limit(k, width), weight, left[k], right[k]);
}
} // namespace mp6
} // namespace aero
#endif
