#ifndef AERO_RECONSTRUCTION_H
#define AERO_RECONSTRUCTION_H
#include "../parallel/kernel_math.h"
#include "../cv/flux_surface.h"
#include "ebr.h"
#include "mpweno5.h"
#include "weno.h"

namespace aero {
typedef unsigned long long ReconstructionIndex;

// Для наклона по двум узлам сумма уже записана через разность поля.
// Первый коэффициент при этом не читается, а второй сохраняется в double.
struct TwoNodeSlope {
    // Номера двух узлов, задающих разность.
    unsigned first, second;
    // Масштаб двухточечной разности.
    double weight;
    // Зарезервированный код для специального представления строки.
    static constexpr unsigned marker = ~0u;
};
// Один узел объединённого шаблона EBR5 и номера двух точных весов в словаре плитки.
struct EbrTerm {
    unsigned node;
    unsigned short left,right;
};

// Разреженная строка весов и её привязка к узловому полю.
struct StencilRow {
    // Число доступных опорных узлов, включая два конца центральной разности.
    ReconstructionIndex count;
    // Список узлов общей строки либо первый конец центральной разности.
    union { const ReconstructionIndex* nodes; ReconstructionIndex central_first; };
    // Веса общей строки либо второй конец центральной разности.
    union { const double* weights; ReconstructionIndex central_second; };
    // Короткая запись для взвешенной разности двух узлов.
    const TwoNodeSlope* pair;
    unsigned char central; // В центральной строке два указателя заменены номерами концов ребра.
    unsigned term_mask; // Опоры стороны в объединённой строке плитки.
    const unsigned* nodes32; // На GPU небольшая сетка хранит номера опор без старших нулевых битов.
    // Пустая либо обычная строка не обращается к альтернативным полям объединений.
    AERO_HOST_DEVICE StencilRow(ReconstructionIndex n=0,const ReconstructionIndex* ids=nullptr,
            const double* factors=nullptr,const TwoNodeSlope* p=nullptr,bool center=false,const unsigned* ids32=nullptr)
        : count(n),nodes(ids),weights(factors),pair(p),central(center),term_mask(0),nodes32(ids32) {}
    // Концы уже прочитаны из широкой или короткой геометрии. Копия двух номеров
    // не требует указателя на временно распакованную площадку и дополнительного массива.
    AERO_HOST_DEVICE static StencilRow central_row(const residual_detail::Surface& edge) {
        StencilRow result;
        result.count=2; result.central=true;
        result.central_first=edge.begin; result.central_second=edge.end;
        return result;
    }
    // Позиция i-й опоры данной стороны в объединённом шаблоне на 32 ребра.
    AERO_HOST_DEVICE unsigned term_offset(unsigned i) const {
        unsigned mask=term_mask;
        for (unsigned j=0;j<i;++j) mask &= mask-1;
        unsigned bit=0;
        while (!(mask&1u)) { ++bit; mask>>=1; }
        return 32*bit;
    }
    // Номер опоры и её вес во всех поддерживаемых форматах геометрии.
    AERO_HOST_DEVICE ReconstructionIndex node(ReconstructionIndex i) const {
        if (central>=2) return reinterpret_cast<const EbrTerm*>(nodes)[term_offset(static_cast<unsigned>(i))].node;
        return central ? (i ? central_second : central_first) :
               (pair ? (i ? pair->second : pair->first) : (nodes32 ? nodes32[i] : nodes[i]));
    }
    AERO_HOST_DEVICE double weight(ReconstructionIndex i) const {
        if (central>=2) {
            const auto term=reinterpret_cast<const EbrTerm*>(nodes)[term_offset(static_cast<unsigned>(i))];
            return weights[central==2 ? term.left : term.right];
        }
        return central ? 1. : (pair ? pair->weight : weights[i]);
    }
    // Возвращает разреженное представление строки шаблона.
    AERO_HOST_DEVICE static StencilRow sparse(const ReconstructionIndex* starts, const ReconstructionIndex* ids,
                                              const double* factors, ReconstructionIndex row) {
        if (!starts) return {}; // Все строки таблицы пусты.
        const auto first = starts[row], count = starts[row+1]-first;
        return {count, count ? ids+first : nullptr, count ? factors+first : nullptr};
    }
    // В одной компактной таблице сначала смещения строк, затем номера опор.
    AERO_HOST_DEVICE static StencilRow packed(const unsigned* table, ReconstructionIndex rows,
                                               const double* factors, ReconstructionIndex row) {
        const ReconstructionIndex first=table[row], count=table[row+1]-first;
        return {count,nullptr,count ? factors+first : nullptr,nullptr,false,
                count ? table+rows+1+first : nullptr};
    }
};

// Компактная геометрия всех сторон рёбер. Значения поля подставляются при каждом вызове реконструкции.
struct ReconstructionStencilView {
    // Габариты узловой и рёберной нумерации.
    ReconstructionIndex node_count, edge_count;
    const ReconstructionIndex*
        offsets; // Всего 2 * edge_count + 1 смещений. Строка 2*edge относится к левой стороне, следующая — к правой.
    // Номера узлов геометрического шаблона.
    const ReconstructionIndex* nodes;
    const ReconstructionIndex* row_owner; // Узел-владелец каждой строки стороны ребра.
    const double* weights; // Строки с нулевой суммой для удвоенных приращений. Пустая строка даёт ноль.
    // Для нелинейной реконструкции на сторону ребра приходятся четыре строки производных:
    // центральная, ближняя с этой стороны, ближняя с противоположной и дальняя с этой стороны.
    const ReconstructionIndex* slope_offsets;
    // Узлы строк вычисления наклонов.
    const ReconstructionIndex* slope_nodes;
    // Веса узлов соответствующих строк.
    const double* slope_weights;
    const unsigned char*
        weno_order; // Ноль задаёт линейную формулу, 3 или 5 — соответствующий порядок WENO-JS.
    const double* weno_epsilon_scale; // h·√ε в единицах поля.
    Mp5Settings mp5; // Для полной строки MP5 подставляем ограничитель вместо весов WENO.
    // Таблица двухточечных разностей.
    const TwoNodeSlope* pairs;
    bool shared_edge_slopes; // Общие центральный и два ближних наклона хранятся один раз на ребро.
    residual_detail::SurfaceTable owner_edges; // Номера узлов уже хранятся в геометрии потоков.
    bool epsilon_per_edge; // Обе стороны используют один вес, если значения совпали точно.
    bool short_slope_rows; // WENO3 хранит только два наклона стороны: 3 общих либо 4 отдельных на ребро.
    bool central_from_edges; // Точная разность Q_end-Q_begin не требует отдельной строки коэффициентов.
    const unsigned* linear_edge_mask; // Один бит на сторону для линейных строк, равных центральной разности.
    const unsigned* linear_indices32; // Смещения и номера линейного CSR, если все значения помещаются в 32 бита.
    const unsigned* slope_indices32; // Та же упаковка для редких длинных наклонов либо исходного CSR.
    ReconstructionIndex slope_rows; // Размер таблицы исключений нужен для начала компактных номеров.
    // Проверяет наличие отдельной таблицы владельцев сторон шаблона.
    AERO_HOST_DEVICE bool has_owners() const { return row_owner || owner_edges; }
    // Возвращает узел, от которого отсчитывается приращение реконструкции.
    AERO_HOST_DEVICE ReconstructionIndex owner(ReconstructionIndex row) const {
        if (!owner_edges) return row_owner[row];
        const auto& edge = owner_edges[row/2];
        return row%2 ? edge.end : edge.begin;
    }
    // Возвращает масштабированный параметр регуляризации нелинейных весов.
    AERO_HOST_DEVICE double epsilon(ReconstructionIndex row) const {
        return weno_epsilon_scale[epsilon_per_edge ? row/2 : row];
    }
    // Возвращает число пар опорных наклонов текущей схемы.
    AERO_HOST_DEVICE ReconstructionIndex pair_count() const {
        return pairs ? edge_count * (shared_edge_slopes ? (short_slope_rows ? 3 : 5)-unsigned(central_from_edges)
                                                        : (short_slope_rows ? 4 : 8)-2*unsigned(central_from_edges)) : 0;
    }
    // При наличии плиток nodes хранит EbrTerm, weights — словари double, а
    // linear_indices32 содержит смещения плиток и маски обеих сторон каждого ребра.
    AERO_HOST_DEVICE bool tiled_linear() const { return linear_indices32 && nodes; }
    // Возвращает линейную строку реконструкции для выбранной стороны.
    AERO_HOST_DEVICE StencilRow linear_row(ReconstructionIndex row) const {
        if (tiled_linear()) {
            const auto edge=row/2,tile=edge/32;
            const auto* header=linear_indices32+3*tile;
            const unsigned mask=linear_indices32[3*((edge_count+31)/32)+row];
            unsigned count=0;
            for (unsigned bits=mask;bits;bits&=bits-1) ++count;
            StencilRow result;
            result.count=count;
            result.nodes=reinterpret_cast<const ReconstructionIndex*>(reinterpret_cast<const EbrTerm*>(nodes)+header[0]+edge%32);
            result.weights=weights+header[1];
            result.term_mask=mask;
            result.central=static_cast<unsigned char>(2+row%2);
            return result;
        }
        if (linear_edge_mask) {
            const bool active=(linear_edge_mask[row/32]>>(row%32))&1u;
            return active ? StencilRow::central_row(owner_edges[row/2]) : StencilRow{};
        }
        if (linear_indices32) return StencilRow::packed(linear_indices32,2*edge_count,weights,row);
        return StencilRow::sparse(offsets,nodes,weights,row);
    }
    // Возвращает строку одного опорного наклона WENO.
    AERO_HOST_DEVICE StencilRow slope_row(ReconstructionIndex row) const {
        if (central_from_edges && row%4==0) {
            const bool active=weno_order[row/4]!=0;
            return active ? StencilRow::central_row(owner_edges[row/8]) : StencilRow{};
        }
        if (pairs) {
            auto slot=row;
            if (short_slope_rows && row%4>=2) return {};
            if (shared_edge_slopes) {
                const auto side_row=row/4;
                const unsigned order=weno_order[side_row];
                if (!order || row%4>=weno_slope_count(order)) return {};
                // Слева: центральный, ближний левый, ближний правый, дальний левый.
                // Справа меняются местами ближние наклоны и выбирается дальний правый.
                const unsigned k=static_cast<unsigned>(row%8);
                const unsigned shared=k<4 ? k : (k==4 ? 0u : (k==5 ? 2u : (k==6 ? 1u : 4u)));
                slot=(row/8)*((short_slope_rows ? 3 : 5)-unsigned(central_from_edges))+shared-unsigned(central_from_edges);
            } else slot=(row/4)*((short_slope_rows ? 2 : 4)-unsigned(central_from_edges))+row%4-unsigned(central_from_edges);
            const auto& p = pairs[slot];
            if (p.first != TwoNodeSlope::marker) return {2,nullptr,nullptr,&p};
            if (!p.second) return {};
            row = p.second-1; // Нестандартная строка хранится целиком, в исходном порядке.
        }
        if (slope_indices32) return StencilRow::packed(slope_indices32,slope_rows,slope_weights,row);
        return StencilRow::sparse(slope_offsets,slope_nodes,slope_weights,row);
    }
};

// Читает скалярные значения поля через нумерацию узлов и шаг компонент.
struct ReconstructionFieldView {
    // Начало массива реконструируемых компонент.
    const double* values;
    // Смещения при переходе к следующему узлу и следующей компоненте.
    ReconstructionIndex node_stride, component_stride;
    // Читает компоненту узла с учётом шагов размещения поля.
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
evaluate_reconstruction_row(StencilRow row, ReconstructionIndex read_row, ReconstructionIndex component,
                            const Reader& read) {
    // Библиотечный читатель выдаёт одну компоненту. Тот же обход в солвере
    // читает сразу всё состояние или вычисляет один физический поток для всех компонент.
    const auto values = [&](ReconstructionIndex i, double* buffer) -> const double* {
        *buffer = read(read_row,row.node(i),component);
        return buffer;
    };
    // Возвращает вес опоры в текущей строке реконструкции.
    const auto weight = [&](ReconstructionIndex i) { return row.weight(i); };
    double result;
    reconstruction_detail::sum_slope<1>(1,row.count,values,weight,&result);
    return result;
}
// Вычисляет взвешенную сумму значений на опорах разреженной строки.
template <class Reader>
AERO_HOST_DEVICE inline double evaluate_reconstruction_sparse(const ReconstructionIndex* offsets,
        const ReconstructionIndex* nodes,const double* weights,ReconstructionIndex row,
        ReconstructionIndex read_row,ReconstructionIndex component,const Reader& read) {
    return evaluate_reconstruction_row(StencilRow::sparse(offsets,nodes,weights,row),read_row,component,read);
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
    // Настройки монотонного ограничения.
    Mp5Settings mp5;
    // Вычисляет нелинейное приращение выбранной схемы реконструкции.
    AERO_HOST_DEVICE double operator()(unsigned order, const double* slopes, double epsilon) const {
        return nonlinear_increment(order, slopes, epsilon, mp5);
    }
};

enum class ReconstructionSide { Left, Right };
enum class ReconstructionStatus { Success, InvalidEdge, InvalidSide, InvalidLayout, NullPointer };

// Состояние на конце ребра передаётся отдельно. Остальные узлы читает read(row,node,component),
// где row = 2·edge + side. Можно читать компактный локальный шаблон или внешние данные.
template <class Reader> struct ReconstructionSideInput {
    // Геометрические строки, доступные при реконструкции этой стороны.
    ReconstructionStencilView stencil;
    // Номер строки шаблона для данной стороны.
    ReconstructionIndex row;
    // Значения со стороны узла-владельца.
    const double* own;
    ReconstructionIndex own_stride; // Расстояние в элементах между компонентами состояния владельца.
    const Reader& neighbours; // Чтение остальных узлов, в том числе импортированных для MPI.
    // Подставляет собственный след владельца и читает остальные опоры из соседнего поля.
    AERO_HOST_DEVICE double operator()(ReconstructionIndex r, ReconstructionIndex node,
                                       ReconstructionIndex component) const {
        return node == stencil.owner(row) ? own[component * own_stride] : neighbours(r, node, component);
    }
    // Вычисляет линейную часть приращения на стороне ребра.
    AERO_HOST_DEVICE double linear(ReconstructionIndex component) const {
        return evaluate_reconstruction_row(stencil.linear_row(row),row,component,*this);
    }
    // Возвращает доступный порядок реконструкции выбранной стороны.
    AERO_HOST_DEVICE unsigned order() const { return stencil.weno_order ? stencil.weno_order[row] : 0; }
    // Вычисляет опорный наклон по соответствующей строке весов.
    AERO_HOST_DEVICE double slope(unsigned k, ReconstructionIndex component) const {
        return evaluate_reconstruction_row(stencil.slope_row(4*row+k),row,component,*this);
    }
    // Выбирает масштаб регуляризации WENO для текущей стороны.
    AERO_HOST_DEVICE double epsilon_scale() const { return stencil.epsilon(row); }
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
    if (!own || !output || !stencil.has_owners())
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
// Возвращает число доступных устройств через интерфейс GPU-модуля реконструкции.
int aero_reconstruction_device_count(int* count, char* error, unsigned capacity);
// Создаёт GPU-копию шаблона реконструкции и возвращает её дескриптор.
int aero_reconstruction_device_create(aero::ReconstructionStencilView view, int device, void** state,
                                    char* error, unsigned capacity);
// Освобождает GPU-копию шаблона реконструкции.
void aero_reconstruction_device_destroy(void* state);
// Возвращает представление шаблона из GPU-дескриптора.
void aero_reconstruction_device_view(void* state, aero::ReconstructionStencilView* view);
}

#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY)
#include "scheme.h"
#include "../parallel/small_weight_row.h"
#include "../mesh/mesh.h"
#include <memory>
#include <map>

namespace aero {
enum class ReconstructionSideScheme { FirstOrder, EdgeAverage, Ebr3, Ebr5, Weno3, Weno5, Mp5, MPWENO5 };

enum class StencilGeometry { Straight, Ijk, Pl, Auto };
// Выбранная схема, допуски и параметры построения опор реконструкции.
struct ReconstructionOptions {
    // Способ геометрического продолжения шаблона.
    StencilGeometry geometry = StencilGeometry::Straight;
    double max_turn_degrees = 36.; // Максимальное отклонение опоры от продолжения ребра.
    std::vector<std::string> wall_labels; // Физические метки, от которых растут слои PL.
    std::vector<int> node_layers; // При разбиении номера слоёв передаются на вспомогательную сетку.

    ReconstructionScheme scheme; // Схема, запрошенная для обеих сторон всех рёбер.
    // Параметры геометрии относятся к EBR и WENO. Для первого порядка внешние точки не требуются.
    double relative_tolerance;
    bool mpweno_ad_corr = true; // Добавлять chi_boost к нелинейным весам MPWENO5.
    double mp5_alpha = 4; // По умолчанию 4. Для явной схемы солвер учитывает CFL.
    bool solver_rows_only = false; // В рабочем шаблоне храним только строки, читаемые потоком.
    // Параметр ε для квадратов разделённых разностей WENO в единицах сетки.
    double weno_epsilon;
    // В строгом режиме понижение порядка относительно запрошенной схемы считается ошибкой.
    bool require_complete_stencil;
    // Задаёт исходную схему EBR3 и стандартные допуски реконструкции.
    ReconstructionOptions()
        : scheme(ReconstructionScheme::Ebr3), relative_tolerance(1.e-10), weno_epsilon(1.e-40),
          require_complete_stencil(false) {}
};

// Владеет CSR-строками, весами и геометрическими масштабами шаблона.
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
    // Суммарная длина от конца ребра до дальней опоры через ближнюю.
    // Все схемы пятого порядка берут дальний наклон со знаменателем far_distance − outer_distance.
    std::vector<double> far_distance;
    // Создаёт пустые CSR-таблицы шаблона реконструкции EBR3.
    ReconstructionStencilData()
        : scheme(ReconstructionScheme::Ebr3), node_count(0), edge_count(0), offsets(1, 0),
          slope_offsets(1, 0) {}
};

// Проверяет и хранит подготовленные данные шаблона реконструкции.
class ReconstructionStencil {
public:
    // Возвращает владеющее хранилище коэффициентов шаблона.
    const ReconstructionStencilData& data() const { return data_; }
    // Передаёт владение массивами шаблона вызывающему коду.
    ReconstructionStencilData release_data() && { return std::move(data_); }
    // Формирует невладеющее представление массивов подготовленного шаблона.
    ReconstructionStencilView view() const;
    // Подсчитывает стороны рёбер с неполным высокопорядковым шаблоном.
    ReconstructionIndex incomplete_sides() const;

private:
    // Владеет построенными таблицами шаблонов.
    ReconstructionStencilData data_;
    // Периодическая склейка перенумеровывает узлы шаблона.
    friend class PeriodicDomain;
    // Одна точка входа для обычной сетки, периодического покрытия и локальной сетки MPI.
    friend ReconstructionStencil prepare_reconstruction(const Mesh&, const ReconstructionOptions&);
};

// Общие геометрические операции подготовки EBR и WENO. Координаты и топология должны существовать всё время работы помощника.
namespace detail {
using SparseRow = SmallWeightRow<ReconstructionIndex,8>;
// Пересечение луча с геометрией и строка интерполяции в найденной точке.
struct RayPoint {
    double distance;   // Расстояние от начала луча до пересечения.
    bool outward;      // Луч направлен наружу относительно грани.
    SparseRow weights; // Веса интерполяции в точке пересечения; пусто, если опоры нет.
    // Создаёт пустой результат пересечения луча с бесконечным расстоянием.
    RayPoint();
};
// Добавляет строку коэффициентов с заданным множителем.
void add_scaled_row(SparseRow& destination, const SparseRow& source, double factor);
// Строит строку направленной производной по геометрическим опорам луча.
SparseRow ray_derivative(MeshIndex node, const RayPoint& hit, double h, unsigned side);
// Управляет геометрическим поиском опор прямолинейной реконструкции.
class ReconstructionGeometry {
public:
    // Готовит геометрический поиск опор на заданной сетке с указанным допуском.
    ReconstructionGeometry(const Mesh& mesh, double tolerance);
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~ReconstructionGeometry();
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    ReconstructionGeometry(const ReconstructionGeometry&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
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
    // Внутренние данные поиска геометрического шаблона.
    std::unique_ptr<Impl> impl_;
};
} // namespace detail

// Одна точка входа для обычной сетки, периодического покрытия и локальной сетки MPI.
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
    // Размещает подготовленный шаблон на выбранном исполнителе.
    explicit ReconstructionContext(ReconstructionStencil stencil,
                                   ReconstructionBackend backend = ReconstructionBackend::Auto,
                                   int device = 0);
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~ReconstructionContext();
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    ReconstructionContext(const ReconstructionContext&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    ReconstructionContext& operator=(const ReconstructionContext&) = delete;
    // Возвращает выбранный исполнитель вычислений.
    ReconstructionBackend backend() const;
    // Возвращает номер устройства; отрицательное значение соответствует CPU.
    int device() const;
    // Возвращает представление шаблона с адресами, доступными CPU.
    ReconstructionStencilView host_view() const;
    // Возвращает представление шаблона с адресами выбранного GPU.
    ReconstructionStencilView device_view() const;
    // Сообщает, собран ли модуль с поддержкой CUDA.
    static bool cuda_compiled();
    // Возвращает число доступных CUDA-устройств.
    static int cuda_device_count();

private:
    struct Impl;
    // Рабочее состояние выбранного исполнителя реконструкции.
    std::unique_ptr<Impl> impl_;
};
} // namespace aero
#endif
#endif
