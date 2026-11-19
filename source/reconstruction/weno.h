#ifndef AERO_WENO_H
#define AERO_WENO_H
#include "../parallel/kernel_math.h"

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
    double d[4] = {0, 0, 0, 0};
    for (unsigned k = 0; k < slope_count; ++k)
        d[k] = slopes[k] / scale;
    double candidate[3] = {d[0], d[1], 0}, beta[3] = {d[0] * d[0], d[1] * d[1], 0};
    double ideal[3] = {2.0 / 3, 1.0 / 3, 0};
    if (order == 5) {
        const double a = d[2] - d[0], b = d[2] - 3 * d[0];
        const double c = d[0] - d[1], e = d[0] + d[1];
        const double f = d[1] - d[3], g = 3 * d[1] - d[3];
        beta[0] = (13.0 / 12) * a * a + 0.25 * b * b;
        beta[1] = (13.0 / 12) * c * c + 0.25 * e * e;
        beta[2] = (13.0 / 12) * f * f + 0.25 * g * g;
        candidate[0] = weno_third(4 * d[0] - d[2]);
        candidate[1] = weno_third(2 * d[0] + d[1]);
        candidate[2] = weno_third(5 * d[1] - 2 * d[3]);
        ideal[0] = 0.3;
        ideal[1] = 0.6;
        ideal[2] = 0.1;
    }
    const double t = epsilon_scale / scale, epsilon = t * t;
    double denominator[3] = {0, 0, 0};
    for (unsigned k = 0; k < count; ++k)
        denominator[k] = epsilon + beta[k];
    double minimum = denominator[0];
    for (unsigned k = 1; k < count; ++k)
        if (denominator[k] < minimum)
            minimum = denominator[k];
    // У WENO3 два кандидата: отношение меньшего знаменателя к себе равно 1.
    // Одно деление даёт оставшийся вес независимо от того, какой наклон меньше.
    double smaller_ratio = 0;
    if (count == 2 && minimum != 0)
        smaller_ratio = minimum / (denominator[0] < denominator[1] ? denominator[1] : denominator[0]);
    double sum = 0, result = 0;
    for (unsigned k = 0; k < count; ++k) {
        double ratio;
        if (minimum == 0)
            ratio = denominator[k] == 0 ? 1.0 : 0.0;
        else if (count == 2)
            ratio = denominator[k] == minimum ? 1.0 : smaller_ratio;
        else
            ratio = minimum / denominator[k];
        const double weight = ideal[k] * ratio * ratio;
        sum += weight;
        result += weight * candidate[k];
    }
    return scale * (result / sum);
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
