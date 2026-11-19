#ifndef AERO_RECONSTRUCTION_H
#define AERO_RECONSTRUCTION_H
#include "../parallel/kernel_math.h"
#include "ebr.h"
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
};

struct ReconstructionFieldView {
    const double* values;
    ReconstructionIndex node_stride, component_stride;
    AERO_HOST_DEVICE double operator()(ReconstructionIndex /*row*/, ReconstructionIndex node,
                                       ReconstructionIndex component) const {
        return values[node * node_stride + component * component_stride];
    }
};

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
    if (begin == end)
        return 0.0; // Первому порядку достаточно значения на конце ребра.
    const double reference = read(read_row, nodes[begin], component);
    double result = 0.0;
    // Разностная запись точно обнуляет константу и избегает вычитания двух больших восстановленных состояний.
    // Первый коэффициент задан нулевой суммой строки. Разность в его узле равна нулю.
    for (ReconstructionIndex k = begin + 1; k < end; ++k)
        result += weights[k] * (read(read_row, nodes[k], component) - reference);
    return result;
}

enum class ReconstructionSide { Left, Right };
enum class ReconstructionStatus { Success, InvalidEdge, InvalidSide, InvalidLayout, NullPointer };

// Состояние на конце ребра передаётся отдельно. Остальные узлы читает read(row,node,component),
// где row = 2·edge + side. Можно читать компактный локальный шаблон или внешние данные.
template <class Reader> struct ReconstructionSideInput {
    ReconstructionStencilView stencil;
    ReconstructionIndex row;
    const double* own;
    ReconstructionIndex own_stride;
    const Reader& neighbours;
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
    if (stencil.weno_order)
        reconstruct_weno_increment2(input, components, output, output_stride);
    else
        reconstruct_ebr_increment2(input, components, output, output_stride);
    return ReconstructionStatus::Success;
}
} // namespace aero

// Объект владеет геометрией на CUDA. Обход рёбер и запуск ядер выполняет вызывающий код.
extern "C" {
int aero_reconstruction_cuda_count(int* count, char* error, unsigned capacity);
int aero_reconstruction_cuda_create(aero::ReconstructionStencilView view, int device, void** state,
                                    char* error, unsigned capacity);
void aero_reconstruction_cuda_destroy(void* state);
void aero_reconstruction_cuda_view(void* state, aero::ReconstructionStencilView* view);
}

#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
#include "scheme.h"
#include "../mesh/mesh.h"
#include <memory>
#include <map>

namespace aero {
enum class ReconstructionSideScheme { FirstOrder, EdgeAverage, Ebr3, Ebr5, Weno3, Weno5 };

struct ReconstructionOptions {
    ReconstructionScheme scheme; // Схема, запрошенная для обеих сторон всех рёбер.
    // Параметры геометрии относятся к EBR и WENO. Для первого порядка внешние точки не требуются.
    double relative_tolerance;
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
    ReconstructionGeometry(const Mesh& mesh, double tolerance, bool build_stars);
    ~ReconstructionGeometry();
    ReconstructionGeometry(const ReconstructionGeometry&) = delete;
    ReconstructionGeometry& operator=(const ReconstructionGeometry&) = delete;
    // Проверяет положительность и конечность длины ребра.
    double edge_length(MeshIndex edge) const;
    // Ближайший выход при продолжении ребра из его левого или правого конца.
    RayPoint star_hit(MeshIndex edge, unsigned side) const;
    // Самое дальнее допустимое пересечение с гранями одного слоя соседей.
    RayPoint layer_hit(MeshIndex edge, unsigned side, const std::vector<MeshIndex>& faces) const;
    // Ближайший выход из звезды произвольного узла по направлению данного ребра.
    RayPoint parallel_star_hit(MeshIndex node, MeshIndex edge, unsigned side) const;
    // Проверяет строку на постоянном и линейном поле и сохраняет ненулевые коэффициенты.
    void append_row(ReconstructionStencilData& data, const SparseRow& row, MeshIndex edge,
                    unsigned side) const;

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
