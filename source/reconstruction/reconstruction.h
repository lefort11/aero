#ifndef AERO_RECONSTRUCTION_H
#define AERO_RECONSTRUCTION_H
#include "../parallel/kernel_math.h"
#include "ebr.h"
#include "mpweno5.h"
#include "weno.h"

namespace aero {
typedef unsigned long long ReconstructionIndex;

// Компактная геометрия всех сторон рёбер. Значения поля подставляются при каждом вызове реконструкции.
struct ReconstructionStencilView {
    ReconstructionIndex node_count, edge_count;
    const ReconstructionIndex*
        offsets; // Всего 2 * edge_count + 1 смещений. Строка 2*edge относится к левой стороне, следующая — к правой.
    const ReconstructionIndex* nodes;
    const ReconstructionIndex* row_owner; // Узел-владелец каждой строки стороны ребра.
    const double* weights; // Строки с нулевой суммой для удвоенных приращений. Пустая строка даёт ноль.
    // Для нелинейной реконструкции на сторону ребра приходятся четыре строки производных:
    // центральная, ближняя с этой стороны, ближняя с противоположной и дальняя с этой стороны.
    const ReconstructionIndex* slope_offsets;
    const ReconstructionIndex* slope_nodes;
    const double* slope_weights;
    const unsigned char*
        weno_order; // Ноль задаёт линейную формулу, 3 или 5 — соответствующий порядок WENO-JS.
    const double* weno_epsilon_scale; // h·√ε в единицах поля.
    Mp5Settings mp5; // Для полной строки MP5 подставляем ограничитель вместо весов WENO.
};

struct ReconstructionFieldView {
    const double* values;
    ReconstructionIndex node_stride, component_stride;
    AERO_HOST_DEVICE double operator()(ReconstructionIndex /*row*/, ReconstructionIndex node,
                                       ReconstructionIndex component) const {
        return values[node * node_stride + component * component_stride];
    }
};

namespace reconstruction_detail {
// Применяет одну строку геометрических весов сразу ко всем компонентам. Способ чтения
// узлов и весов задаёт вызывающая сторона: одна формула обслуживает оба формата шаблона.
template <unsigned fixed_width = 0, class Values, class Weight>
KOKKOS_FORCEINLINE_FUNCTION bool sum_slope(unsigned width, ReconstructionIndex count, const Values& values, const Weight& weight,
                                       double* slope) {
    const unsigned components = fixed_width ? fixed_width : width;
    for (unsigned c = 0; c < components; ++c)
        slope[c] = 0;
    if (!count)
        return true;
    // Состояния можно читать прямо из поля. Для режима Flux нужны буферы под вычисленные потоки.
    constexpr unsigned capacity = fixed_width ? fixed_width : 5;
    double reference_buffer[capacity], value_buffer[capacity];
    const double* reference = values(0, reference_buffer);
    if (!reference)
        return false;
    // Вычитание первого значения даёт точный нулевой наклон на постоянном поле.
    for (ReconstructionIndex i = 1; i < count; ++i) {
        const double* value = values(i, value_buffer);
        if (!value)
            return false;
        const double factor = weight(i);
        for (unsigned c = 0; c < components; ++c)
            slope[c] += factor * (value[c] - reference[c]);
    }
    return true;
}
}

// Способ чтения значения может зависеть от ребра и стороны, например при работе в характеристическом базисе.
// Возвращаем удвоенные приращения: 2·(Q_L−Q_i) слева и 2·(Q_j−Q_R) справа.
// Низкоуровневый вызов предполагает корректные индексы и доступную исполнителю память.
template <class Reader>
AERO_HOST_DEVICE inline double
evaluate_reconstruction_sparse(const ReconstructionIndex* offsets, const ReconstructionIndex* nodes,
                               const double* weights, ReconstructionIndex sparse_row,
                               ReconstructionIndex read_row, ReconstructionIndex component,
                               const Reader& read) {
    const ReconstructionIndex begin = offsets[sparse_row], end = offsets[sparse_row + 1];
    // Библиотечный читатель выдаёт одну компоненту. Тот же обход в солвере
    // читает сразу всё состояние или вычисляет один физический поток для всех компонент.
    const auto values = [&](ReconstructionIndex i, double* buffer) -> const double* {
        *buffer = read(read_row,nodes[begin+i],component);
        return buffer;
    };
    const auto weight = [&](ReconstructionIndex i) { return weights[begin+i]; };
    double result;
    reconstruction_detail::sum_slope<1>(1,end-begin,values,weight,&result);
    return result;
}

// При неполном шаблоне MP5 использует доступную WENO3; при нулевом порядке вызывающий код читает линейную строку.
AERO_HOST_DEVICE inline double nonlinear_increment(unsigned order, const double* slopes, double epsilon,
                                                   Mp5Settings settings) {
    if (order == 5 && settings.weno)
        return mpweno5::increment(slopes[3], slopes[1], slopes[0], slopes[2], settings.alpha, settings.correction);
    if (order == 5 && settings.enabled)
        return mp5::increment(slopes[3], slopes[1], slopes[0], slopes[2], settings.alpha);
    return weno_increment2(order, slopes, epsilon);
}

// Выбирает формулу по запрошенной схеме и числу доступных слоёв на этой стороне.
struct NonlinearFormula {
    Mp5Settings mp5;
    AERO_HOST_DEVICE double operator()(unsigned order, const double* slopes, double epsilon) const {
        return nonlinear_increment(order, slopes, epsilon, mp5);
    }
};

enum class ReconstructionSide { Left, Right };
enum class ReconstructionStatus { Success, InvalidEdge, InvalidSide, InvalidLayout, NullPointer };

// Состояние на конце ребра передаётся отдельно. Остальные узлы читает read(row,node,component),
// где row = 2·edge + side. Можно читать компактный локальный шаблон или внешние данные.
template <class Reader> struct ReconstructionSideInput {
    ReconstructionStencilView stencil;
    ReconstructionIndex row;
    const double* own;
    ReconstructionIndex own_stride; // Расстояние в элементах между компонентами состояния владельца.
    const Reader& neighbours; // Чтение остальных узлов, в том числе импортированных для MPI.
    AERO_HOST_DEVICE double operator()(ReconstructionIndex r, ReconstructionIndex node,
                                       ReconstructionIndex component) const {
        return node == stencil.row_owner[row] ? own[component * own_stride] : neighbours(r, node, component);
    }
    AERO_HOST_DEVICE double linear(ReconstructionIndex component) const {
        return evaluate_reconstruction_sparse(stencil.offsets, stencil.nodes, stencil.weights, row, row,
                                              component, *this);
    }
    AERO_HOST_DEVICE unsigned order() const { return stencil.weno_order ? stencil.weno_order[row] : 0; }
    AERO_HOST_DEVICE double slope(unsigned k, ReconstructionIndex component) const {
        return evaluate_reconstruction_sparse(stencil.slope_offsets, stencil.slope_nodes,
                                              stencil.slope_weights, 4 * row + k, row, component, *this);
    }
    AERO_HOST_DEVICE double epsilon_scale() const { return stencil.weno_epsilon_scale[row]; }
};

// За вызов обрабатывается одна сторона ребра, одинаково на CPU и в ядре CUDA. Массивы должны быть доступны исполнителю.
// Выход хранится отдельно от входа. Знак обоих приращений соответствует направлению i → j:
// слева 2·(Q_L−Q_i), справа 2·(Q_j−Q_R). Обход сетки и запуск ядра выполняет вызывающая сторона.
template <class Reader>
AERO_HOST_DEVICE inline ReconstructionStatus
reconstruct_increment2(ReconstructionStencilView stencil, ReconstructionIndex edge, ReconstructionSide side,
                       const double* own, ReconstructionIndex components, const Reader& neighbours,
                       double* output, ReconstructionIndex own_stride = 1,
                       ReconstructionIndex output_stride = 1) {
    if (edge >= stencil.edge_count)
        return ReconstructionStatus::InvalidEdge;
    if (side != ReconstructionSide::Left && side != ReconstructionSide::Right)
        return ReconstructionStatus::InvalidSide;
    const ReconstructionIndex limit = ~static_cast<ReconstructionIndex>(0);
    if (!components || !own_stride || !output_stride || (components - 1) > limit / own_stride ||
        (components - 1) > limit / output_stride)
        return ReconstructionStatus::InvalidLayout;
    if (!own || !output || !stencil.row_owner)
        return ReconstructionStatus::NullPointer;
    const ReconstructionIndex row = 2 * edge + (side == ReconstructionSide::Right ? 1 : 0);
    const ReconstructionSideInput<Reader> input = {stencil, row, own, own_stride, neighbours};
    // В линейной схеме коэффициенты можно объединить до первого шага.
    // В нелинейной схеме геометрия тоже постоянна, но наклоны и веса зависят от поля.
    if (stencil.weno_order)
        reconstruct_slope_rows(input, components, output, output_stride, NonlinearFormula{stencil.mp5});
    else
        reconstruct_ebr_increment2(input, components, output, output_stride);
    return ReconstructionStatus::Success;
}
} // namespace aero

// Объект владеет геометрией на CUDA. Обход рёбер и запуск ядер выполняет вызывающий код.
extern "C" {
int aero_reconstruction_device_count(int* count, char* error, unsigned capacity);
int aero_reconstruction_device_create(aero::ReconstructionStencilView view, int device, void** state,
                                    char* error, unsigned capacity);
void aero_reconstruction_device_destroy(void* state);
void aero_reconstruction_device_view(void* state, aero::ReconstructionStencilView* view);
}

#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY)
#include "scheme.h"
#include "../mesh/mesh.h"
#include <memory>
#include <map>

namespace aero {
enum class ReconstructionSideScheme { FirstOrder, EdgeAverage, Ebr3, Ebr5, Weno3, Weno5, Mp5, MPWENO5 };

enum class StencilGeometry { Straight, Ijk, Pl, Auto };
struct ReconstructionOptions {
    StencilGeometry geometry = StencilGeometry::Straight;
    double max_turn_degrees = 36.; // Максимальное отклонение опоры от продолжения ребра.
    std::vector<std::string> wall_labels; // Физические метки, от которых растут слои PL.
    std::vector<int> node_layers; // При разбиении номера слоёв передаются на вспомогательную сетку.

    ReconstructionScheme scheme; // Схема, запрошенная для обеих сторон всех рёбер.
    // Параметры геометрии относятся к EBR и WENO. Для первого порядка внешние точки не требуются.
    double relative_tolerance;
    bool mpweno_ad_corr = true; // Добавлять chi_boost к нелинейным весам MPWENO5.
    double mp5_alpha = 4; // По умолчанию 4. Для явной схемы солвер учитывает CFL.
    // Параметр ε для квадратов разделённых разностей WENO в единицах сетки.
    double weno_epsilon;
    // В строгом режиме понижение порядка относительно запрошенной схемы считается ошибкой.
    bool require_complete_stencil;
    ReconstructionOptions()
        : scheme(ReconstructionScheme::Ebr3), relative_tolerance(1.e-10), weno_epsilon(1.e-40),
          require_complete_stencil(false) {}
};

struct ReconstructionStencilData {
    ReconstructionScheme
        scheme; // Запрошенная схема; фактический выбор для каждой стороны хранит side_scheme.
    ReconstructionIndex node_count, edge_count; // Число узлов и рёбер в этой нумерации.
    std::vector<ReconstructionIndex> offsets; // Границы строки 2*ребро + сторона в массивах nodes и weights.
    std::vector<ReconstructionIndex> nodes;   // Номера опорных узлов в порядке коэффициентов.
    std::vector<ReconstructionIndex> row_owner; // Конец ребра, значение которого передаёт вызывающий код.
    bool mpweno_ad_corr = true; // Добавлять chi_boost к нелинейным весам MPWENO5.
    double mp5_alpha = 4; // Общий параметр MP5 для всех рёбер.
    std::vector<double> weights;                // Коэффициенты линейной части удвоенного приращения.
    // Для WENO верхние веса задают оптимальный линейный оператор. Дополнительные строки CSR
    // содержат геометрию для вычисления нелинейных весов по текущему полю.
    std::vector<ReconstructionIndex> slope_offsets, slope_nodes; // Смещения и номера узлов строк производных.
    std::vector<double> slope_weights;      // Коэффициенты четырёх строк производных на сторону.
    std::vector<double> weno_epsilon_scale; // h*sqrt(epsilon) для нормировки гладкости.
    std::vector<unsigned char> weno_order;  // 5 или 3 для WENO; 0 применяет линейную строку.
    // На каждой стороне 1 означает полную запрошенную схему, 0 — понижение порядка.
    // Первый порядок всегда полон, в том числе на граничных узлах.
    std::vector<unsigned char> complete;
    std::vector<ReconstructionSideScheme> side_scheme; // Фактическая схема после проверки доступных слоёв.
    // Расстояние до внешней точки в единицах координат сетки. Ноль означает отсутствие точки.
    std::vector<double> outer_distance;
    // EBR5 хранит расстояние от конца ребра до интерполированной позиции второго слоя только для диагностики.
    // Дальнюю производную EBR5 получают интерполяцией производных. При отсутствии слоя расстояние равно нулю.
    // У WENO5 дальняя точка — фактическое второе пересечение луча, а знаменатель наклона равен far_distance − outer_distance.
    std::vector<double> far_distance;
    ReconstructionStencilData()
        : scheme(ReconstructionScheme::Ebr3), node_count(0), edge_count(0), offsets(1, 0),
          slope_offsets(1, 0) {}
};

class ReconstructionStencil {
public:
    const ReconstructionStencilData& data() const { return data_; }
    ReconstructionStencilView view() const;
    ReconstructionIndex incomplete_sides() const;

private:
    ReconstructionStencilData data_;
    friend class PeriodicDomain;
    friend ReconstructionStencil prepare_reconstruction(const Mesh&, const ReconstructionOptions&);
};

// Общие геометрические операции подготовки EBR и WENO. Координаты и топология должны существовать всё время работы помощника.
namespace detail {
typedef std::map<ReconstructionIndex, double> SparseRow;
struct RayPoint {
    double distance;   // Расстояние от начала луча до пересечения.
    bool outward;      // Луч направлен наружу относительно грани.
    SparseRow weights; // Веса интерполяции в точке пересечения; пусто, если опоры нет.
    RayPoint();
};
void add_scaled_row(SparseRow& destination, const SparseRow& source, double factor);
SparseRow ray_derivative(MeshIndex node, const RayPoint& hit, double h, unsigned side);
class ReconstructionGeometry {
public:
    ReconstructionGeometry(const Mesh& mesh, double tolerance);
    ~ReconstructionGeometry();
    ReconstructionGeometry(const ReconstructionGeometry&) = delete;
    ReconstructionGeometry& operator=(const ReconstructionGeometry&) = delete;
    // Проверяет положительность и конечность длины ребра.
    double edge_length(MeshIndex edge) const;
    // Самое дальнее допустимое пересечение с гранями одного слоя соседей.
    RayPoint layer_hit(MeshIndex edge, unsigned side, const std::vector<MeshIndex>& faces) const;
    // Проверяет строку на постоянном и линейном поле и сохраняет ненулевые коэффициенты.
    void append_row(ReconstructionStencilData& data, const SparseRow& row, MeshIndex edge,
                    unsigned side, bool curved = false) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace detail

ReconstructionStencil prepare_reconstruction(const Mesh& mesh,
                                             const ReconstructionOptions& options = ReconstructionOptions());

// Обёртка CPU для одного вектора на конце ребра с проверкой входных параметров.
template <class Reader>
std::vector<double> reconstruct_increment2(const ReconstructionStencil& stencil, ReconstructionIndex edge,
                                           ReconstructionSide side, const std::vector<double>& own,
                                           const Reader& neighbours) {
    std::vector<double> result(own.size());
    const ReconstructionStatus status =
        reconstruct_increment2(stencil.view(), edge, side, own.data(),
                               static_cast<ReconstructionIndex>(own.size()), neighbours, result.data());
    if (status != ReconstructionStatus::Success)
        throw ReconstructionError("Invalid single-side reconstruction arguments");
    return result;
}

enum class ReconstructionBackend { Auto, Cpu, Cuda };

// Владеет подготовленной геометрией на CPU и, при необходимости, её копией на GPU.
// Поля решения и обход сетки принадлежат вызывающему коду. Режим Auto выбирает CPU при отсутствии CUDA или устройства.
class ReconstructionContext {
public:
    explicit ReconstructionContext(ReconstructionStencil stencil,
                                   ReconstructionBackend backend = ReconstructionBackend::Auto,
                                   int device = 0);
    ~ReconstructionContext();
    ReconstructionContext(const ReconstructionContext&) = delete;
    ReconstructionContext& operator=(const ReconstructionContext&) = delete;
    ReconstructionBackend backend() const;
    int device() const;
    ReconstructionStencilView host_view() const;
    ReconstructionStencilView device_view() const;
    static bool cuda_compiled();
    static int cuda_device_count();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace aero
#endif
#endif
