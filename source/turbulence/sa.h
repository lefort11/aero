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
    // Активирует вклад турбулентной модели.
    bool enabled = false;
    // Настройки подсеточного масштаба и модели LES.
    LesSettings les;
    // Включает ограничение длины SA по модели DDES.
    bool ddes = false;
    double c_des = .65; // Коэффициент сеточного масштаба в модели DDES Спаларта.
    // Проверяет, нужен ли перенос рабочей вязкости SA для выбранной модели.
    bool uses_sa() const { return enabled && (ddes || les.scale==Scale::Off); }
    // Проверяет, использует ли выбранная модель замыкание Sigma.
    bool uses_sigma() const { return enabled && !uses_sa(); }
    bool high_order = false; // Восстанавливать nu_tilde на газовом шаблоне перед конвективным переносом.
    double nut_nu_amb = 0.; // Отношение физической турбулентной вязкости к молекулярной в набегающем потоке.
    double prandtl = .9;    // Турбулентное число Прандтля в уравнении полной энергии.
};
constexpr double cv1 = 7.1, sigma = 2. / 3., cb1 = .1355, cb2 = .622, kappa = .41;
// Связь теплового коэффициента AES с геометрическим коэффициентом уравнения SA.
inline double diffusion_scale(double gamma,double prandtl) {
    return (gamma-1.)*prandtl/(gamma*sigma);
}

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

// Вычисляет локальные источники рождения и разрушения рабочей вязкости SA.
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
    // Ориентированный идентификатор площадки и соседний узел.
    MeshIndex encoded, neighbour;
    // Вектор площади для конвективного переноса.
    double area[3];
};
// У площадки, перпендикулярной координатной оси, две компоненты площади
// точно нулевые. Храним одну компоненту со знаком и номер соответствующей оси.
struct AxisFace {
    // Упакованные направление оси со стороной и номер соседа.
    std::uint32_t side_axis, neighbour;
    // Мера осевой площадки.
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
    // Раздельные столбцы ориентации и номеров соседей.
    const std::uint32_t *orientation, *neighbour;
    // Столбец площадей в упакованной таблице.
    const double* area;
    // Количество записей столбцового представления.
    MeshIndex count;
    // Восстанавливает запись площадки SA из раздельных массивов компонент.
    template<unsigned dimension>
    AERO_HOST_DEVICE FaceTerm read(MeshIndex p) const {
        FaceTerm face{orientation[p],neighbour[p],{}};
        for (unsigned d=0; d<dimension; ++d) face.area[d]=area[d*count+p];
        return face;
    }
};
// Полуинтервал элементов строки: начало включительно, конец исключительно.
struct RowRange {
    // Полуоткрытый диапазон записей строки.
    MeshIndex first, end;
};
// Коэффициент одной связи узла с соседом в одном объёмном элементе AES.
// Нулевые коэффициенты при подготовке сетки пропускаем без численного порога.
template<class Index> struct DiffusionEntry {
    // Номер элемента и соседней вершины для диффузионной связи.
    Index element, neighbour;
    // Геометрический множитель диффузионного вклада.
    double geometry;
};
using DiffusionTerm=DiffusionEntry<MeshIndex>;
using LocalDiffusionTerm=DiffusionEntry<std::uint32_t>;
// У тетраэдра и у ортогонального гексаэдра в диффузионной строке часто
// остаются три соседа на элемент. У всех трёх одна средняя вязкость элемента.
struct DiffusionTriple {
    // Элемент и три соседних узла локального шаблона.
    std::uint32_t element, neighbour[3];
    // Коэффициенты трёх связей и диагонали.
    double geometry[3], diagonal;
};
// Все соседи CPU-строки из одного элемента используют его среднюю вязкость.
struct DiffusionCell {
    // Число остальных вершин восьмиузловой ячейки.
    enum { neighbours=7 };
    // Элемент и его соседние узлы для данной строки.
    std::uint32_t element, node[neighbours];
    // Геометрические коэффициенты соседей и диагональный вклад.
    double geometry[neighbours], diagonal;
};
// Поток диффузии от элемента в вершину и его вклад в оценку шага.
struct ScalarContribution { double rate, diagonal; };
// Представление строк и коэффициентов диффузии рабочей вязкости SA.
struct Diffusion {
    // Диапазоны диффузионных связей по узлам.
    const RowRange *rows;
    const MeshIndex *anchor; // Первый элемент строки при общей геометрии с другими узлами.
    // Общая таблица диффузионных связей.
    const DiffusionTerm *terms;
    // Вклады в локальной нумерации элемента.
    const LocalDiffusionTerm* local_terms;
    // Геометрические веса пар вершин.
    const double* pair_weights;
    // Сопоставление элементов повторяющимся формам.
    const MeshIndex* shape;
    // Промежуточные вклады по вершинам до узловой сборки.
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
// Короткие цепочки узлов и их диапазоны для линейного сглаживания SA.
struct SaLines {
    // Максимальная длина сегмента для линейного сглаживателя.
    static constexpr unsigned length = 16;
    // Узлы и коэффициенты цепочек сглаживания.
    const LineVertex* vertices;
    // Границы отдельных участков цепочек.
    const RowRange* segments;
    // Количество участков для обработки.
    MeshIndex count;
};
// Невладеющее представление геометрии и коэффициентов турбулентного оператора.
struct View {
    // Связность цепочек для неявного переноса SA.
    SaLines lines;
    // Данные элементной диффузии рабочего поля.
    Diffusion diffusion;
    // Границы строк конвективных площадок.
    const RowRange *face_rows;
    // Полные записи конвективных площадок.
    const FaceTerm *face_terms;
    // Компактные записи граней, параллельных координатным плоскостям.
    const AxisFace *axis_faces;
    // Альтернативное представление геометрии по столбцам.
    FaceColumns face_columns;
    // Расстояние между последовательными площадками одной строки.
    unsigned face_stride;
    double *incoming; // Входящий поток массы, делённый на плотность, пока в BDF газ фиксирован.
    // Молекулярная и вихревая вязкости узла с коэффициентом демпфирования SA.
    struct Material {
        // Молекулярная кинематическая и турбулентная динамическая вязкости.
        double nu, eddy_mu;
        double damping; // fv1 на текущей явной стадии. Внутри BDF рабочая вязкость меняется.
    }; // Молекулярная кинематическая и турбулентная динамическая вязкости.
    // Рабочие коэффициенты материала в узлах.
    Material *nodal_material;
    double *rotation; // Модуль ротора по последнему полю газа, общий для внутренних итераций BDF.
    // Активирует вычисление переноса SA.
    bool active;
    // Фоновая рабочая вязкость и отношение чисел Прандтля.
    double ambient, pr_ratio;
    // Рабочая вязкость, временные слои и резервная копия, скорость изменения и диагональ.
    double *w, *old, *previous, *backup, *rate, *diagonal;
    double *last_correction, *plain_update; // Только для внутренних итераций неявного уравнения SA.
    // Обратное расстояние от узла до стенки.
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
    // Закрепляет нулевую рабочую вязкость на узлах стенки с прилипанием.
    AERO_HOST_DEVICE bool zero_working_viscosity(MeshIndex i) const {
        return inverse_distance[i]<0. || (ddes && inverse_distance[i]==0. && ddes[i].grid_length==0.);
    }
};
}
}
