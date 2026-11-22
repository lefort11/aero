#pragma once
#include "../parallel/kernel_math.h"
#include "../mesh/mesh_index.h"
#include <cmath>
#include "sigma.h"
#include "ddes.h"

namespace aero {
namespace turbulence {
// Параметры постановки. Коэффициенты стандартной модели SA одинаковы во всех задачах.
struct Settings {
    bool enabled = false;
    LesSettings les;
    bool ddes = false;
    double c_des = .65; // Коэффициент сеточного масштаба в модели DDES Спаларта.
    bool uses_sa() const { return enabled && (ddes || les.scale==Scale::Off); }
    bool uses_sigma() const { return enabled && !uses_sa(); }
    bool high_order = false; // Восстанавливать nu_tilde на газовом шаблоне перед конвективным переносом.
    double nut_nu_amb = 0.; // Отношение физической турбулентной вязкости к молекулярной в набегающем потоке.
    double prandtl = .9;    // Турбулентное число Прандтля в уравнении полной энергии.
};
constexpr double cv1 = 7.1, sigma = 2. / 3., cb1 = .1355, cb2 = .622, kappa = .41;

// Демпфирование вязкости вблизи стенки. Отношение в каждой ветви не больше единицы.
AERO_HOST_DEVICE inline double fv1(double chi) {
    if (chi < cv1) {
        const double r = chi / cv1, cube = r * r * r;
        return cube * reciprocal(1. + cube);
    }
    const double r = cv1 * reciprocal(chi);
    return reciprocal(1. + r * r * r);
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
    const double ratio = g > 2. ? 2. * reciprocal(g) : .5 * g;
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
    const double y = center * (432. + t * (396. + 55. * t)) * reciprocal(432. + t * (468. + 91. * t));
    const double cube = y * y * y;
    const double error = (1. + x) * cube * cube - 1.;
    return amplitude * (y * small_inverse_sixth_root(error));
}

// Производство и разрушение рабочей вязкости. Возвращаем коэффициент при nu_tilde.
// На самой стенке закрепляем nu_tilde=0 до вызова этой функции. Если стенок нет, inv_d=0.
AERO_HOST_DEVICE inline double source_rate_from_chi(double w, double chi, double damp,
                                                    double rotation, double inv_d) {
    const double ft2 = 1.2 * ::exp(-.5 * chi * chi);
    // При отсутствии стенок члены с обратным расстоянием точно равны нулю.
    if (inv_d == 0.)
        return cb1 * (1. - ft2) * rotation;
    const double fv2 = 1. - chi * reciprocal(1. + chi * damp);
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

AERO_HOST_DEVICE inline double source_rate(double w, double nu, double rotation, double inv_d) {
    const double chi=w/nu;
    return source_rate_from_chi(w,chi,inv_d==0. ? 0. : fv1(chi),rotation,inv_d);
}

// Две обычные поправки с приблизительно постоянным отношением q дают
// остаток delta*q/(1-q) после текущей итерации. При |q|>1/2 остаёмся
// на обычном шаге, чтобы изменение рабочей вязкости не выросло больше чем вдвое.
AERO_HOST_DEVICE inline double extrapolated_viscosity(double plain,double delta,double previous) {
    if (delta!=0. && ::fabs(delta)<=.5*::fabs(previous)) {
        const double next=plain+delta*(delta/(previous-delta));
        if (next>=0. && next<=max_finite_double) return next;
    }
    return plain;
}

// Подготовленное вхождение поверхности в строку узла. В area уже учтены
// ориентация и половинный вес скорости на внутренней площадке.
struct FaceTerm {
    MeshIndex encoded, neighbour;
    double area[3];
};
// У площадки, перпендикулярной координатной оси, две компоненты площади
// точно нулевые. Храним одну компоненту со знаком и номер соответствующей оси.
struct AxisFace {
    std::uint32_t side_axis, neighbour;
    double area;
    // n × Δu даёт только две компоненты ротора. Градиент рабочей вязкости
    // на этой площадке имеет только компоненту вдоль нормали.
    template<bool with_curl>
    AERO_HOST_DEVICE void add_derivatives(double jump,const double* velocity,double* gradient,double* curl) const {
        switch (side_axis%4) {
        case 0:
            gradient[0]+=area*jump;
            if (with_curl) { curl[1]-=area*velocity[2]; curl[2]+=area*velocity[1]; }
            break;
        case 1:
            gradient[1]+=area*jump;
            if (with_curl) { curl[0]+=area*velocity[2]; curl[2]-=area*velocity[0]; }
            break;
        default:
            gradient[2]+=area*jump;
            if (with_curl) { curl[0]-=area*velocity[1]; curl[1]+=area*velocity[0]; }
        }
    }
};
// На GPU номера и компоненты вектора площади храним отдельными столбцами.
// При чтении одного соседа 32 исполнителя обращаются к смежным адресам.
struct FaceColumns {
    const std::uint32_t *orientation, *neighbour;
    const double* area;
    MeshIndex count;
    template<unsigned dimension>
    AERO_HOST_DEVICE FaceTerm read(MeshIndex p) const {
        FaceTerm face{orientation[p],neighbour[p],{}};
        for (unsigned d=0; d<dimension; ++d) face.area[d]=area[d*count+p];
        return face;
    }
};
struct RowRange {
    MeshIndex first, end;
};
// Коэффициент одной связи узла с соседом в одном объёмном элементе AES.
// Нулевые коэффициенты при подготовке сетки пропускаем без численного порога.
template<class Index> struct DiffusionEntry {
    Index element, neighbour;
    double geometry;
};
using DiffusionTerm=DiffusionEntry<MeshIndex>;
using LocalDiffusionTerm=DiffusionEntry<std::uint32_t>;
// У тетраэдра и у ортогонального гексаэдра в диффузионной строке часто
// остаются три соседа на элемент. У всех трёх одна средняя вязкость элемента.
struct DiffusionTriple {
    std::uint32_t element, neighbour[3];
    double geometry[3], diagonal;
};
// Соседние исполнители GPU читают один и тот же номер вершины в своих
// элементах. Каждая такая строка из 32 номеров или коэффициентов лежит подряд.
template<unsigned lanes> struct DiffusionBlock {
    enum { neighbours=7 };
    std::uint32_t element[lanes], node[neighbours][lanes];
    double geometry[neighbours][lanes], diagonal[lanes];
};
using DiffusionTile=DiffusionBlock<32>;
using DiffusionCell=DiffusionBlock<1>;
// Поток диффузии от элемента в вершину и его вклад в оценку шага.
struct ScalarContribution { double rate, diagonal; };
struct Diffusion {
    const RowRange *rows;
    const MeshIndex *anchor; // Первый элемент строки при общей геометрии с другими узлами.
    unsigned stride;
    const DiffusionTerm *terms;
    const LocalDiffusionTerm* local_terms;
    const DiffusionTile* tiles;
    const double* pair_weights;
    const MeshIndex* shape;
    ScalarContribution* vertex_contributions;
    const DiffusionCell* cell_rows; // На CPU вклады одной строки лежат подряд.
    const DiffusionTriple* triples; // Заполнено только при точном соответствии структуры строк.
    double *molecular_mean, *element_mean; // Среднее (nu + nu_tilde) по вершинам текущего элемента.
    bool short_blocks; // Ни у одного элемента нет более четырёх вершин.
};
// Массивы на выбранном исполнителе. old и previous содержат два принятых временных слоя.
// На вытянутых элементах две наибольшие связи скалярной диффузии образуют
// короткие линии. Их геометрические веса постоянны, а вязкость меняется в итерациях.
struct LineVertex { MeshIndex node; double left, right; };
struct SaLines {
    static constexpr unsigned length = 16;
    const LineVertex* vertices;
    const RowRange* segments;
    MeshIndex count;
};
struct View {
    SaLines lines;
    Diffusion diffusion;
    const RowRange *face_rows;
    const FaceTerm *face_terms;
    const AxisFace *axis_faces;
    FaceColumns face_columns;
    unsigned face_stride;
    double *incoming; // Входящий поток массы, делённый на плотность, пока в BDF газ фиксирован.
    struct Material {
        double nu, eddy_mu;
        double damping; // fv1 на текущей явной стадии. Внутри BDF рабочая вязкость меняется.
    }; // Молекулярная кинематическая и турбулентная динамическая вязкости.
    Material *nodal_material;
    double *rotation; // Модуль ротора по последнему полю газа, общий для внутренних итераций BDF.
    bool active;
    double ambient, pr_ratio;
    double *w, *old, *previous, *backup, *rate, *diagonal;
    double *last_correction, *plain_update; // Только для внутренних итераций неявного уравнения SA.
    const double *inverse_distance;
    DdesPoint* ddes; // Только собственные строки, для которых решаем перенос SA.
    // Рабочая вязкость обновляется теми же коэффициентами Эйлера или SSPRK3,
    // что и газ. В начале шага сохраняем значение для последующих стадий.
    template<class Stage>
    AERO_HOST_DEVICE void advance_explicit(MeshIndex i,const Stage& stage,double dt,bool save_start) const {
        if (save_start) old[i]=w[i];
        const double value=stage.combine(stage.uses_step_start() ? old[i] : w[i], w[i]+dt*rate[i]);
        w[i]=zero_working_viscosity(i) ? 0. : (value<0. ? 0. : value);
    }
    AERO_HOST_DEVICE bool zero_working_viscosity(MeshIndex i) const {
        return inverse_distance[i]<0. || (ddes && inverse_distance[i]==0. && ddes[i].grid_length==0.);
    }
};
}
}
