#pragma once
#include "../parallel/kernel_math.h"
#include "../mesh/mesh_index.h"
#include <cmath>

namespace aero {
namespace turbulence {
// Параметры постановки. Коэффициенты стандартной модели SA одинаковы во всех задачах.
struct Settings {
    bool enabled = false;
    double nut_nu_amb = 0.; // Отношение физической турбулентной вязкости к молекулярной в набегающем потоке.
    double prandtl = .9;    // Турбулентное число Прандтля в уравнении полной энергии.
};
constexpr double cv1 = 7.1, sigma = 2. / 3., cb1 = .1355, cb2 = .622, kappa = .41;

// Демпфирование вязкости вблизи стенки. Отношение в каждой ветви не больше единицы.
AERO_HOST_DEVICE inline double fv1(double chi) {
    if (chi < cv1) {
        const double r = chi / cv1, cube = r * r * r;
        return cube / (1. + cube);
    }
    const double r = cv1 / chi;
    return 1. / (1. + r * r * r);
}
// На входе задано nu_t/nu. Обращаем монотонную функцию chi*fv1(chi).
inline double ambient_chi(double ratio) {
    if (ratio == 0.)
        return 0.;
    if (ratio >= 1.e7)
        return ratio; // Поправка fv1 уже меньше точности double.
    double low = ratio, high = ratio + cv1;
    // Для очень малых отношений начальный масштаб определяется четвёртым корнем.
    if (ratio < 1.)
        high = 2. * std::sqrt(std::sqrt(ratio)) * std::pow(cv1, .75);
    for (unsigned i = 0; i < 64; ++i) {
        const double mid = low + .5 * (high - low);
        if (mid * fv1(mid) < ratio)
            low = mid;
        else
            high = mid;
    }
    return low + .5 * (high - low);
}

// Три члена биномиального ряда для (1+e)^(-1/6). При |e| <= 1e-4
// первый отброшенный член меньше 6e-18, в том числе на стыке ветвей ниже.
AERO_HOST_DEVICE inline double small_inverse_sixth_root(double e) {
    return 1. + e * (-1. / 6. + e * (7. / 72. - e * (91. / 1296.)));
}

// Пристенная функция fw = g * (65/(g^6+64))^(1/6). Приводим её к корню
// от 1+x с 0 <= x <= 1. Так избегаем общего pow и больших шестых степеней.
AERO_HOST_DEVICE inline double wall_function(double g) {
    constexpr double scale = 1.00258737257521077; // (65/64)^(1/6)
    const double ratio = g > 2. ? 2. / g : .5 * g;
    const double amplitude = (g > 2. ? 2. : g) * scale;
    const double square = ratio * ratio;
    const double x = square * square * square;
    if (x <= 1.e-4)
        return amplitude * small_inverse_sixth_root(x);

    // Аппроксимация Паде [2/2] в середине отрезка даёт начальную оценку корня.
    // Величина (1+x)*y^6-1 показывает её относительную ошибку в шестой степени.
    // Эта величина меньше 6e-5. Одно уточнение по ней даёт точность double.
    const double t = (x - .5) * (2. / 3.);
    constexpr double center = .93465526518406714; // (1.5)^(-1/6)
    const double y = center * (432. + t * (396. + 55. * t)) / (432. + t * (468. + 91. * t));
    const double cube = y * y * y;
    const double error = (1. + x) * cube * cube - 1.;
    return amplitude * (y * small_inverse_sixth_root(error));
}

// Производство и разрушение рабочей вязкости. Возвращаем коэффициент при nu_tilde.
// На самой стенке закрепляем nu_tilde=0 до вызова этой функции. Если стенок нет, inv_d=0.
AERO_HOST_DEVICE inline double source_rate(double w, double nu, double rotation, double inv_d) {
    const double chi = w / nu;
    const double ft2 = 1.2 * ::exp(-.5 * chi * chi);
    // При отсутствии стенок члены с обратным расстоянием точно равны нулю.
    if (inv_d == 0.)
        return cb1 * (1. - ft2) * rotation;
    const double damp = fv1(chi);
    const double fv2 = 1. - chi / (1. + chi * damp);
    const double scale = (w * inv_d) * inv_d / (kappa * kappa);
    // Нижняя граница 0.3*Omega стабилизирует модифицированный модуль завихренности.
    const double lower = .3 * rotation, modified = rotation + scale * fv2;
    const double shear = modified > lower ? modified : lower;
    const double ratio = shear > 0. ? scale / shear : (scale > 0. ? 10. : 0.);
    const double r = ratio < 10. ? ratio : 10.;
    const double r2 = r * r, g = .7 * r + .3 * r2 * r2 * r2;
    const double fw = wall_function(g);
    const double cw1 = cb1 / (kappa * kappa) + (1. + cb2) / sigma;
    return cb1 * (1. - ft2) * shear - (cw1 * fw * kappa * kappa - cb1 * ft2) * scale;
}

// Подготовленное вхождение поверхности в строку узла. В area уже учтены
// ориентация и половинный вес скорости на внутренней площадке.
struct FaceTerm {
    MeshIndex encoded, neighbour;
    double area[3];
};
struct RowRange {
    MeshIndex first, end;
};
// Коэффициент одной связи узла с соседом в одном объёмном элементе AES.
// Нулевые коэффициенты при подготовке сетки пропускаем без численного порога.
struct DiffusionTerm {
    MeshIndex element, neighbour;
    double geometry;
};
struct Diffusion {
    const RowRange *rows;
    const MeshIndex *anchor; // Первый элемент строки при общей геометрии с другими узлами.
    unsigned stride;
    const DiffusionTerm *terms;
    double *molecular_mean, *element_mean; // Среднее (nu + nu_tilde) по вершинам текущего элемента.
};
// Массивы на выбранном исполнителе. old и previous содержат два принятых временных слоя.
struct View {
    Diffusion diffusion;
    const RowRange *face_rows;
    const FaceTerm *face_terms;
    unsigned face_stride;
    double *incoming; // Входящий поток массы, делённый на плотность, пока в BDF газ фиксирован.
    struct Material {
        double nu, eddy_mu;
    }; // Молекулярная кинематическая и турбулентная динамическая вязкости.
    Material *nodal_material;
    double *rotation; // Модуль ротора по последнему полю газа, общий для внутренних итераций BDF.
    bool active;
    double ambient, pr_ratio;
    double *w, *old, *previous, *backup, *rate, *diagonal;
    const double *inverse_distance;
};
}
}
