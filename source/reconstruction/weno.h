#ifndef AERO_WENO_H
#define AERO_WENO_H
#include "../parallel/kernel_math.h"
#ifndef __CUDACC_RTC__
#include <string.h>
#endif

// Скалярная формула WENO3/WENO5 общая для CPU и CUDA и пригодна для вызова внутри ядра.

namespace aero {
// Деление кандидата на три с округлением binary64. На GPU заранее известны обе части числа 1/3.
AERO_HOST_DEVICE inline double weno_third(double x) {
#ifdef __CUDA_ARCH__
    // Для нормальных x/3 расстояние до середины между соседними double не меньше 1/6 шага.
    // Ошибка младшей части значительно меньше этого расстояния. FMA даёт то же округление,
    // что и деление. Для очень малых x произведение с младшей частью могло бы потерять точность.
    const unsigned exponent = (static_cast<unsigned>(__double2hiint(x)) >> 20) & 2047u;
    if (exponent >= 80 && exponent < 2047) {
        constexpr double high = 1.0 / 3.0;
        constexpr double low = high / 18014398509481984.0; // (1/3 − high), округлённое к double.
        return __fma_rn(x, high, __dmul_rn(x, low));
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

// Индикатор гладкости Jiang–Shu для одного из трёх кандидатов пятого порядка.
// curvature задаёт изменение наклона, gradient — комбинацию наклонов для первой производной.
AERO_HOST_DEVICE inline double weno_smoothness(double curvature, double gradient) {
    return (13.0 / 12) * curvature * curvature + 0.25 * gradient * gradient;
}

// Формула (9) Бахвалова и Козубской (2017) с весами Jiang–Shu. Наклоны равны разделённым разностям, умноженным на h.
// Масштабирование и отношения весов позволяют обойти переполнение квадратов и прямой расчёт 1/(ε+β)².
// Здесь применяется исходная схема JS.
AERO_HOST_DEVICE inline double weno_increment2(unsigned order, const double* slopes, double epsilon_scale) {
    const unsigned count = order == 5 ? 3 : 2, slope_count = order == 5 ? 4 : 2;
    double scale = epsilon_scale;
    for (unsigned k = 0; k < slope_count; ++k) {
        const double magnitude = slopes[k] < 0 ? -slopes[k] : slopes[k];
        if (magnitude > scale)
            scale = magnitude;
    }
    if (scale == 0)
        return 0;
    const WenoScale units(scale);
    double d[4] = {0, 0, 0, 0};
    for (unsigned k = 0; k < slope_count; ++k)
        d[k] = units.normalize(slopes[k]);
    // У третьего порядка два кандидата. Их квадраты наклонов служат индикаторами гладкости β.
    double candidate[3] = {d[0], d[1], 0}, beta[3] = {d[0] * d[0], d[1] * d[1], 0};
    double ideal[3] = {2.0 / 3, 1.0 / 3, 0};
    if (order == 5) {
        // d[0] — центральный наклон, d[1] — ближний со своей стороны, d[2] — с противоположной,
        // d[3] — дальний со своей стороны. Каждая пара определяет один из трёх кандидатов.
        beta[0] = weno_smoothness(d[2] - d[0], d[2] - 3 * d[0]);
        beta[1] = weno_smoothness(d[0] - d[1], d[0] + d[1]);
        beta[2] = weno_smoothness(d[1] - d[3], 3 * d[1] - d[3]);
        candidate[0] = weno_third(4 * d[0] - d[2]);
        candidate[1] = weno_third(2 * d[0] + d[1]);
        candidate[2] = weno_third(5 * d[1] - 2 * d[3]);
        ideal[0] = 0.3;
        ideal[1] = 0.6;
        ideal[2] = 0.1;
    }
    // На гладком шаблоне знаменатель ε + β мал, и кандидат получает больший вес.
    const double t = units.normalize(epsilon_scale), epsilon = t * t;
    double denominator[3] = {0, 0, 0};
    for (unsigned k = 0; k < count; ++k)
        denominator[k] = epsilon + beta[k];
    double minimum = denominator[0];
    for (unsigned k = 1; k < count; ++k)
        if (denominator[k] < minimum)
            minimum = denominator[k];
    double factors[3] = {denominator[1], denominator[0], 0};
    if (minimum >= 1e-18) {
        // Нормированные наклоны ограничены по величине. Здесь можно умножить все веса
        // на общий квадрат произведения знаменателей: он сократится при делении на сумму.
        if (count == 3) {
            factors[2] = factors[0] * factors[1];
            factors[0] *= denominator[2];
            factors[1] *= denominator[2];
        }
    } else {
        // Вблизи нуля произведения могут потерять точность. Отношения к минимуму
        // лежат в [0, 1]. При нулевом минимуме остаются только гладкие кандидаты.
        for (unsigned k = 0; k < count; ++k)
            factors[k] = minimum == 0 ? double(denominator[k] == 0) : minimum / denominator[k];
    }
    double weights[3], sum = 0;
    for (unsigned k = 0; k < count; ++k) {
        weights[k] = ideal[k] * factors[k] * factors[k];
        sum += weights[k];
    }
    // Сначала нормируем веса: малый общий множитель не должен обнулить вклад малого наклона.
    const double inverse_sum = 1 / sum;
    double result = 0;
    for (unsigned k = 0; k < count; ++k)
        result += (weights[k] * inverse_sum) * candidate[k];
    return units.unit * result;
}

// Восстанавливает вектор на одной подготовленной стороне ребра. Веса Jiang–Shu зависят от текущего решения.
// Формула одинакова на CPU и CUDA.
template <class SideInput>
AERO_HOST_DEVICE inline void reconstruct_weno_increment2(const SideInput& input,
                                                         unsigned long long components, double* output,
                                                         unsigned long long output_stride = 1) {
    const unsigned order = input.order();
    for (unsigned long long c = 0; c < components; ++c) {
        if (!order) {
            output[c * output_stride] = input.linear(c);
            continue;
        }
        double slopes[4] = {0, 0, 0, 0};
        for (unsigned k = 0; k < (order == 5 ? 4u : 2u); ++k)
            slopes[k] = input.slope(k, c);
        output[c * output_stride] = weno_increment2(order, slopes, input.epsilon_scale());
    }
}

#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
class Mesh;
class ReconstructionStencil;
struct ReconstructionOptions;

// Готовит геометрию при options.scheme равном Weno3 или Weno5. Общие типы настроек и шаблона объявлены в reconstruction.h.
ReconstructionStencil prepare_weno(const Mesh& mesh, const ReconstructionOptions& options);
#endif
} // namespace aero
#endif
