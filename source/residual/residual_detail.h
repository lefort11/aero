#ifndef AERO_RESIDUAL_DETAIL_H
#define AERO_RESIDUAL_DETAIL_H
#include "../parallel/kernel_math.h"
#include "residual.h"
#include "../mesh/mesh_index.h"
#include "../bc/bc.h"
#include "../eulersolver/roe.h"
#include "../bc/linear_boundary.h"
#include "../eulersolver/lax_friedrichs.h"
#include "../reconstruction/reconstruction.h"
#include "mp6_cartesian.h"
namespace aero {
namespace residual_detail {
using Index = ReconstructionIndex;

// Запись списка узла хранит номер поверхности и направление её нормали относительно узла.
struct OrientedFace {
    Index encoded; // Младший бит отмечает нормаль, направленную внутрь контрольного объёма.

    // Создаёт ссылку на площадку с положительной ориентацией для текущего узла.
    AERO_HOST_DEVICE static OrientedFace outward(Index face) { return {2 * face}; }
    // Создаёт ссылку на площадку с отрицательной ориентацией для текущего узла.
    AERO_HOST_DEVICE static OrientedFace inward(Index face) { return {2 * face + 1}; }
    // При плотном обходе k уже является номером грани. Выделенная группа
    // или подобласть MPI передаёт свой список; для GPU достаточно его адреса.
    AERO_HOST_DEVICE Index face() const { return encoded / 2; }

    // Приводит компоненту сохранённого потока к наружной нормали этого узла.
    AERO_HOST_DEVICE double outward_flux(double component) const {
        return (encoded & 1) ? -component : component;
    }
};
#if !defined(__CUDACC_RTC__) && !defined(AERO_RESIDUAL_KERNEL_ONLY)
// Переводит вектор площади в базис компонент импульса решателя.
void project_surface(Surface&, CvVector area, unsigned dimension, const ResidualFrame&);
// Однородная среда у неподвижной стенки может двигаться только вдоль неё.
void validate_linear_walls(const Geometry&);
// Строит CSR-списки ориентированных площадок каждого узла.
void build_incidence(Index nodes, Index edges, const std::vector<Surface>& surfaces,
                     std::vector<Index>& offsets, std::vector<OrientedFace>& incidence);
// Определяет, требуется ли заменить реконструированные следы узловыми значениями.
inline bool needs_nodal_traces(BoundaryType type) {
    // На открытой метке режим меняется во времени. Для всех её узлов нужны внутренние состояния.
    return type == BoundaryType::Free;
}
#endif
// Порядок строк в упаковке обоих концов ребра. Центральная и ближние строки хранятся по одному разу.
enum WenoSlopeRow : unsigned { CentralSlope, NearLeftSlope, NearRightSlope, FarLeftSlope, FarRightSlope };

// В исходном шаблоне первые четыре строки принадлежат левой стороне, следующие четыре — правой.
// При укороченной левой стороне её противоположный наклон берём из собственной строки правой.
AERO_HOST_DEVICE constexpr unsigned edge_slope_offset(unsigned slope, unsigned left_order) {
    return slope == FarRightSlope ? 7u : (slope == NearRightSlope && left_order != 5 ? 5u : slope);
}

// Число узлов в наклоне: концы ребра, интерполяция ближней грани либо разность двух интерполяций.
AERO_HOST_DEVICE constexpr unsigned slope_terms(unsigned row) {
    return row == CentralSlope ? 2 : (row <= NearRightSlope ? 3 : 4);
}

// Номер первого слагаемого в списке всех строк. При чтении шаблона известен при компиляции.
AERO_HOST_DEVICE constexpr unsigned first_slope_term(unsigned row) {
    return row ? first_slope_term(row - 1) + slope_terms(row - 1) : 0;
}

// Порядки обеих сторон проходят с ребром от сборки шаблона до записи потока.
struct PackedEdgeOrder {
    unsigned char flags; // 0 — пропустить ребро, бит 0 — готово, биты 1 и 2 — укороченные стороны.
    // Кодирует параметры обеих сторон в общее представление ребра.
    static PackedEdgeOrder from_sides(bool short_left, bool short_right) {
        return {static_cast<unsigned char>(1u | (unsigned(short_left) << 1) | (unsigned(short_right) << 2))};
    }
    // Проверяет наличие готовых упакованных данных обработки ребра.
    AERO_HOST_DEVICE bool ready() const { return flags != 0; }
    // Проверяет полноту высокопорядкового шаблона ребра.
    AERO_HOST_DEVICE bool complete() const { return flags == 1; }
    // Определяет, используется ли укороченный шаблон реконструкции.
    AERO_HOST_DEVICE bool shortened(unsigned side) const { return (flags & (2u << side)) != 0; }
};

// Одна раскладка для постоянной геометрии WENO и временных данных MPWENO5.
// Номер ребра задаёт номер плитки и позицию внутри неё на обоих исполнителях.
namespace edge_tile {
constexpr unsigned width = 32;
// Возвращает индекс записи в упакованной группе рёбер.
AERO_HOST_DEVICE constexpr Index index(Index edge) { return edge / width; }
// Возвращает позицию ребра внутри упакованной группы.
AERO_HOST_DEVICE constexpr unsigned lane(Index edge) { return static_cast<unsigned>(edge % width); }
// Возвращает исходный номер ребра для позиции в упакованной группе.
AERO_HOST_DEVICE constexpr Index edge(Index tile, unsigned position) { return tile * width + position; }
// Последняя неполная плитка тоже занимает место. Сложение не переполняется даже при максимальном n.
AERO_HOST_DEVICE constexpr Index count(Index n) { return index(n) + (lane(n) != 0); }
}

// В плитке соседние потоки CUDA читают подряд данные одного и того же элемента шаблона.
// Для полного двумерного WENO5 пять разных наклонов содержат 2, 3, 3, 4 и 4 слагаемых.
template <unsigned scheme_order> struct WenoEdgeTile {
    // Обеим сторонам достаточно трёх общих наклонов. Пятый порядок добавляет по дальнему с каждой стороны.
    static constexpr unsigned rows_per_edge = scheme_order == 5 ? 5 : 3;
    // Возвращает начало опорных коэффициентов наклона выбранного ребра.
    FixedMatrix<unsigned,first_slope_term(rows_per_edge),edge_tile::width> nodes; // Индексы слагаемых по строкам наклонов.
    PackedEdgeOrder order[edge_tile::width]; // Хранится вместе с постоянной геометрией ребра.
    // В каждой строке первый узел задаёт начало отсчёта. Для него вес хранить не требуется.
    FixedMatrix<double,first_slope_term(rows_per_edge) - rows_per_edge,edge_tile::width> weights;
};
// Рабочая плитка чистой MPWENO5 на GPU. В одной строке подряд лежат значения
// 32 рёбер: так соседние исполнители читают один и тот же наклон соседних рёбер.
struct MpEdgeTile {
    static constexpr Index batch_limit = 131072; // До 4096 плиток; память одинакова для всех стадий.
    PackedEdgeOrder order[edge_tile::width]; // Копируется только после успешной проекции текущего поля.
    FixedTensor<double,5,4,edge_tile::width> waves; // Пять строк по четырём волнам; затем строки 0 и 1 содержат следы.

    // Средний газ один на обе стороны ребра. Нормаль уже хранится с геометрией.
    FixedMatrix<double,2,edge_tile::width> velocity;
    double enthalpy[edge_tile::width], sound[edge_tile::width];
    double kinetic[edge_tile::width], normal_speed[edge_tile::width], tangent_speed[edge_tile::width];

    // Сохраняет характеристический базис ребра для повторного использования.
    AERO_HOST_DEVICE void store_basis(unsigned lane, const euler_detail::PlaneWaveBasis& basis) {
        velocity(0, lane) = basis.velocity[0];
        velocity(1, lane) = basis.velocity[1];
        enthalpy[lane] = basis.enthalpy;
        sound[lane] = basis.sound_speed;
        kinetic[lane] = basis.kinetic;
        normal_speed[lane] = basis.normal_speed;
        tangent_speed[lane] = basis.tangent_speed;
    }

    // Возвращаем именно тот базис, в котором подготовили наклоны на этой стадии.
    AERO_HOST_DEVICE void load_basis(unsigned lane, const double* normal, euler_detail::PlaneWaveBasis& basis) const {
        basis.normal[0] = normal[0];
        basis.normal[1] = normal[1];
        basis.velocity[0] = velocity(0, lane);
        basis.velocity[1] = velocity(1, lane);
        basis.enthalpy = enthalpy[lane];
        basis.sound_speed = sound[lane];
        basis.kinetic = kinetic[lane];
        basis.normal_speed = normal_speed[lane];
        basis.tangent_speed = tangent_speed[lane];
    }
};

// Единичная нормаль площадки в базисе компонент импульса.
struct FaceNormal { double axis[3]; }; // Единичный вектор в базисе решателя.
// Невладеющие геометрия, реконструкция и параметры расчёта потоков.
struct Geometry {
    unsigned dim;              // Число компонент импульса.
    Index nodes, edges, faces; // faces включает рёбра и следующие за ними физические площадки.
    const Surface* surfaces;   // Геометрия в порядке буфера потоков.
    const Index* offsets;      // Узлу i принадлежат записи от offsets[i] до offsets[i+1], не включая конец.
    const OrientedFace* incidence;     // Поверхности узла со знаками для суммирования.
    BoundaryPatchView patches;         // Метки и отдельные данные для free на текущем устройстве.
    ReconstructionStencilView stencil; // Геометрические веса для восстановления состояний на рёбрах.
    ResidualOptions options;           // Переменные реконструкции, газ и решатель потока.

    struct {
        const void* tiles;  // Необязательная упаковка полных шаблонов для CPU и GPU.
        unsigned order;             // Выбирает раскладку WENO3 или WENO5.
        const MeshIndex* remaining; // Остальные площадки текущего процесса, включая физические границы.
        Index count;                // Размер списка remaining.
        Index sparse_edges;         // Сначала идут рёбра с разреженной реконструкцией.
        Index mixed_edges;          // В конце разреженного списка стоят рёбра с одной или двумя укороченными сторонами.
        Index nodal_edges;          // За ними идут рёбра с узловыми следами, затем внешние площадки.
        FaceNormal* normals;        // Единичные нормали рёбер, подготовленные на текущем устройстве.
        EulerState* gas;            // Физические величины в узлах текущей стадии.
        MpEdgeTile* mp_work;        // Буфер проекций, базисов и следов для негибридной MPWENO5 в 2D.
        Index state_nodes;          // Включает удалённые узлы шаблона.
    } packed;
    bool initial_on_boundary; // На GPU начальные данные можно хранить по номеру граничной площадки.

    // Возвращает число консервативных компонент состояния: размерность плюс две.
    AERO_HOST_DEVICE unsigned components() const { return dim + 2; } // Плотность, импульс и энергия.
    // Возвращает строку площадок указанного узла.
    template <class T> AERO_HOST_DEVICE T* node_row(T* field, Index node) const {
        return field + node * components();
    }
    // Вызывается только на физической границе. На CPU исходное поле адресуется
    // по узлу; на карте одна запись соответствует одной граничной площадке.
    AERO_HOST_DEVICE const double* boundary_initial(const double* initial, Index face) const {
        if (!initial) return nullptr;
        return node_row(initial, initial_on_boundary ? face - edges : surfaces[face].begin);
    }
};
// Величины одного узла нужны всем его рёбрам. Нулевая плотность отмечает ошибку разбора.
AERO_HOST_DEVICE inline void decode_nodal_gas(unsigned dimension, const double* field, double gamma,
                                              Index i, EulerState* gas) {
    if (euler_state(dimension, field + (dimension + 2) * i, gamma, gas + i) != EulerStatus::Success)
        gas[i].density = 0;
}

// Читает и преобразует компоненты текущего поля для реконструкции потока.
struct Reader {
    const Geometry& g;           // Размерность, параметры и готовый шаблон.
    const double* field;         // Консервативные переменные текущей стадии по узлам.
    const Surface& surface;      // Ребро, для которого восстанавливаем состояния с двух сторон.
    const EulerRoeBasis& basis;  // Один базис Роу для всех узлов шаблона данного ребра.
    const double* normal;        // Единичная нормаль этого базиса.
    ResidualVariables variables; // Величины, к которым применяются веса шаблона.
    // Переводит уже собранный наклон консервативных переменных в амплитуды волн на ребре.
    AERO_HOST_DEVICE EulerStatus project(const double* u, double* q) const {
        return euler_detail::project_characteristic(basis, normal, g.options.gamma,
                                                    basis.kinetic_energy(), u, q);
    }
    // Физический поток в узле проецируем на нормаль текущего ребра.
    AERO_HOST_DEVICE EulerStatus physical_flux(Index node, const double* u, double* q) const {
        double speed;
        if (g.packed.gas) {
            // Номер берём из шаблона: он уже известен до чтения состояния.
            // Все компоненты физического потока используют один и тот же газ.
            const auto& gas = g.packed.gas[node];
            if (!gas.density)
                return EulerStatus::InvalidDensity;
            return euler_detail::project_flux(g.dim, u, gas, normal, q, &speed);
        }
        return euler_flux(g.dim, u, surface.area, g.options.gamma, q, &speed);
    }
};

// Уравнения выбираются до обхода шаблона. Один и тот же код геометрических
// сумм, нелинейных весов и ограничителей работает с обоими типами полей.
struct LinearReader {
    const Geometry& g;
    const double* field;
    const Surface& surface;
    const double* normal;
    ResidualVariables variables;
    const LinearFace& face;
    FixedMatrix<double,2,3> tangent;
    // Физический поток в узле проецируем на нормаль текущего ребра.
    AERO_HOST_DEVICE EulerStatus physical_flux(Index, const double* q, double* flux) const {
        face.physical(q,flux);
        return EulerStatus::Success;
    }
};
// Проецирует приращение состояния или потока в выбранный базис реконструкции.
AERO_HOST_DEVICE inline bool project_slope(const Reader& r, bool characteristic, double* slope) {
    return !characteristic || r.project(slope,slope) == EulerStatus::Success;
}
// Проецирует приращение состояния или потока в выбранный базис реконструкции.
AERO_HOST_DEVICE inline bool project_slope(const LinearReader& r, bool characteristic, double* q) {
    const auto& gas = r.face.gas;
    const unsigned d = r.g.dim;
    if (characteristic) {
        // Порядок совпадает с полными уравнениями: акустика, энтропия,
        // касательные компоненты и вторая акустическая характеристика.
        double w[5] = {};
        r.face.acoustic(q,w[0],w[d+1]);
        w[1] = q[0]-(w[0]+w[d+1]);
        for (unsigned t=0; t+1<d; ++t) w[2+t] = gas.rho*euler_detail::dot(d,q+1,r.tangent.row(t));
        for (unsigned k=0; k<d+2; ++k) q[k] = w[k];
    } else if (r.variables == ResidualVariables::Conservative) {
        gas.to_conservative(d,q,r.g.options.gamma);
    }
    return true;
}

// Для характеристик переносим линейную проекцию за сумму. Физический поток нелинеен,
// поэтому в режиме Flux по-прежнему вычисляем его отдельно в каждом узле до суммирования.
// Для неполного плоского WENO тип переменных уже известен при выборе обхода.
// В разреженной строке всё равно используем её собственные индексы и локальный порядок.
template <unsigned linear_width = 0, bool characteristic_2d = false, class FieldReader>
KOKKOS_FORCEINLINE_FUNCTION bool transformed_row(const FieldReader& reader, Index owner, const double* own,
                                             StencilRow row, double* out, unsigned stride) {
    const Index count = row.count;
    // У линейного шаблона EBR все компоненты читаются из поля консервативных переменных.
    // Когда ширина известна, сумма одной строки целиком помещается в регистры.
    const unsigned width = characteristic_2d ? 4 : (linear_width ? linear_width : reader.g.components());
    // Пустая строка даёт точный нулевой наклон. У такого шаблона массивы номеров и весов могут быть пусты.
    if (!count) {
        for (unsigned c = 0; c < width; ++c)
            out[c * stride] = 0;
        return true;
    }
    // Возвращает компоненты опорного поля в выбранных переменных реконструкции.
    const auto values = [&](Index i, double* buffer) -> const double* {
        const Index node = row.node(i);
        const double* state = reader.field + node * width;
        if (characteristic_2d || linear_width || reader.variables != ResidualVariables::Flux)
            return state;
        if (node == owner)
            return own;
        return reader.physical_flux(node, state, buffer) == EulerStatus::Success ? buffer : nullptr;
    };
    // Возвращает вес опоры в текущей строке реконструкции.
    const auto weight = [&](Index i) { return row.weight(i); };
    double slope[5];
    if (!reconstruction_detail::sum_slope<characteristic_2d ? 4 : linear_width>(width, count, values, weight, slope))
        return false;
    if (!project_slope(reader, characteristic_2d || (!linear_width && reader.variables == ResidualVariables::Characteristic), slope))
        return false;
    for (unsigned c = 0; c < width; ++c)
        out[c * stride] = slope[c];
    return true;
}

// Сохраняет для каждой компоненты строки одной стороны в порядке, принятом
// в nonlinear_increment: центральная, ближняя, противоположная ближняя и дальняя.
template <bool characteristic_2d = false, class FieldReader>
KOKKOS_FORCEINLINE_FUNCTION bool read_side_slopes(const FieldReader& reader, Index row, const double* own,
                                                           double* slopes) {
    const auto& st = reader.g.stencil;
    for (unsigned k = 0; k < weno_slope_count(st.weno_order[row]); ++k)
        if (!transformed_row<0, characteristic_2d>(reader, st.owner(row), own, st.slope_row(4*row+k), slopes + k, 4))
            return false;
    return true;
}

// По строкам этой стороны восстанавливает удвоенное приращение от узла до грани.
KOKKOS_FORCEINLINE_FUNCTION void side_increments(const ReconstructionStencilView& st, Index row, unsigned width,
                                             const double* slopes, double* out) {
    for (unsigned k = 0; k < width; ++k)
        out[k] = nonlinear_increment(st.weno_order[row], slopes + 4*k, st.epsilon(row), st.mp5);
}

// Вычисляет WENO-приращение в преобразованных переменных.
template <bool characteristic_2d = false, class FieldReader>
KOKKOS_FORCEINLINE_FUNCTION bool transformed_weno_increment(const FieldReader& reader, Index row, const double* own,
                                                                    double* out) {
    const auto& st = reader.g.stencil;
    if (!st.weno_order[row])
        return transformed_row<0, characteristic_2d>(reader, st.owner(row), own, st.linear_row(row), out, 1);
    double slopes[5 * 4];
    if (!read_side_slopes<characteristic_2d>(reader, row, own, slopes))
        return false;
    side_increments(st, row, characteristic_2d ? 4 : reader.g.components(), slopes, out);
    return true;
}

// Восстановленные значения с двух сторон одной грани, до решения задачи о распаде разрыва.
struct EdgeTraces {
    double left[5];  // Значение со стороны begin.
    double right[5]; // Значение со стороны end.
};

// Смешивает два следа EBR с их общим центральным значением до ограничения.
// Пусть L = Qi + Ql_EBR/2 и R = Qj - Qr_EBR/2. При GradC = Qj - Qi
// центральные приращения GradC ± (Ql_EBR - Qr_EBR)/2 дают один след (L + R)/2.
// Поэтому достаточно сохранить среднее следов и умножить их полуразность на upw_coeff.
AERO_HOST_DEVICE inline void blend_ebr_traces(EdgeTraces& traces, unsigned width, double upw_coeff) {
    for (unsigned k = 0; k < width; ++k) {
        const double left = 0.5 * traces.left[k];
        const double right = 0.5 * traces.right[k];
        const double center = left + right;
        const double half_jump = upw_coeff * (left - right);
        traces.left[k] = center + half_jump;
        traces.right[k] = center - half_jump;
    }
}

// После вычисления приращения сдвигаемся от своего узла к середине ребра.
AERO_HOST_DEVICE inline void add_half_increment(unsigned width, const double* node, const double* increment,
                                                 bool right, double* value) {
    for (unsigned k = 0; k < width; ++k)
        value[k] = node[k] + (right ? -0.5 : 0.5) * increment[k];
}

// Приращение схемы вдвое больше сдвига от узла к середине ребра.
// При работе с волнами сначала переводим только приращение обратно в консервативные переменные.
template <unsigned fixed_width = 0, class Basis>
KOKKOS_FORCEINLINE_FUNCTION bool trace_from_increment(unsigned width, const Basis* basis, const double* own,
                                                  double* increment, bool right, double* trace) {
    if (basis && euler_detail::expand_characteristic<fixed_width>(width, *basis, increment, increment)
                     != EulerStatus::Success)
        return false;
    add_half_increment(width,own,increment,right,trace);
    return true;
}

// Добавляет восстановленное приращение к узловому следу.
template<unsigned Width>
AERO_HOST_DEVICE inline bool add_increment(const Reader& r, bool characteristic, const double* own,
                                          double* increment, bool right, double* trace) {
    return trace_from_increment<Width>(r.g.components(),characteristic ? &r.basis : nullptr,own,increment,right,trace);
}
// Добавляет восстановленное приращение к узловому следу.
template<unsigned Width>
AERO_HOST_DEVICE inline bool add_increment(const LinearReader& r, bool characteristic, const double* own,
                                          double* increment, bool right, double* trace) {
    const unsigned d = r.g.dim;
    double q[5] = {};
    if (characteristic) {
        q[0] = increment[1];
        for (unsigned t=0; t+1<d; ++t)
            for (unsigned j=0; j<d; ++j) q[j+1] += r.tangent(t, j)*increment[2+t]/r.face.gas.rho;
        r.face.add_acoustic(increment[0],increment[d+1],q);
        increment = q;
    } else if (r.variables == ResidualVariables::Conservative) {
        r.face.gas.to_primitive(d,increment,r.g.options.gamma);
    }
    return trace_from_increment<Width,EulerRoeBasis>(d+2,nullptr,own,increment,right,trace);
}

// Для conservative и flux базовые приращения считаем по собственным четырём
// строкам каждой стороны, тем же вызовом, что при выключенном гибриде.
template<class FieldReader>
AERO_HOST_DEVICE inline bool cartesian_hybrid_increments(const FieldReader& reader, Index row, const FixedMatrix<double,2,5>& own,
                                                         EdgeTraces& increments) {
    const auto& st = reader.g.stencil;
    const unsigned width = reader.g.components();
    double left_rows[5*4], right_rows[5*4];
    FixedMatrix<double,5,5> slopes;
    // У сторон разные геометрические строки. Отдельные массивы дают ядру GPU
    // известный адрес четырёх наклонов каждой компоненты даже в трёхмерном случае.
    if (!read_side_slopes(reader, row, own.row(0), left_rows) ||
        !read_side_slopes(reader, row+1, own.row(1), right_rows))
        return false;
    side_increments(st, row, width, left_rows, increments.left);
    side_increments(st, row+1, width, right_rows, increments.right);
    // Центральную и обе ближние разности берём в ориентации ребра. Только
    // для дальней точки справа нужна последняя строка правой стороны.
    for (unsigned k = 0; k < width; ++k) {
        for (unsigned j = 0; j < 4; ++j)
            slopes(j, k) = left_rows[4*k+j];
        slopes(FarRightSlope, k) = right_rows[4*k+3];
    }
    const FaceMomentumAxes axes(reader.g.dim, reader.surface.area, reader.normal);
    const double h = reader.g.options.mp6_hybrid;
    if (st.mp5.weno)
        correct_cartesian_pair<true>(axes, slopes, st.mp5, h, increments.left, increments.right);
    else
        correct_cartesian_pair<false>(axes, slopes, st.mp5, h, increments.left, increments.right);
    return true;
}

// На полном шаблоне нужны пять наклонов: центральный, два ближних и два дальних.
// В характеристических переменных обе стороны пользуются одним базисом Роу.
template <bool characteristic_2d, class FieldReader>
AERO_HOST_DEVICE inline bool full_mp_edge_traces(const FieldReader& reader, Index edge, EdgeTraces& traces) {
    const auto& st = reader.g.stencil;
    const unsigned width = characteristic_2d ? 4 : reader.g.components();
    const Index row = 2*edge;
    FixedMatrix<double,2,5> own;
    FixedMatrix<double,5,5> slopes;
    for (unsigned side = 0; side < 2; ++side) {
        const Index node = side ? reader.surface.end : reader.surface.begin;
        const double* state = reader.field + node * width;
        if (reader.variables == ResidualVariables::Flux) {
            if (reader.physical_flux(node, state, own.row(side)) != EulerStatus::Success) return false;
        } else {
            for (unsigned k = 0; k < width; ++k) own(side, k) = state[k];
        }
    }
    if (characteristic_2d || reader.variables == ResidualVariables::Characteristic) {
        // Проецируем каждую из пяти строк ровно один раз. Центральная и обе
        // ближние строки понадобятся и слева, и справа.
        for (unsigned k = 0; k < 5; ++k) {
            const unsigned offset = edge_slope_offset(k, 5);
            const unsigned side = offset / 4;
            const Index slope_row = 4*row + offset;
            if (!transformed_row<0,characteristic_2d>(reader, st.owner(row+side), own.row(side), st.slope_row(slope_row), slopes.row(k), 1))
                return false;
        }
        if (reader.g.options.mp6_hybrid == 0.) {
            // При нулевом весе сюда попадает чистая MP5. Сохраняем прежний
            // порядок аргументов и операций ограничителя отдельно для каждой стороны.
            for (unsigned c=0; c<width; ++c) {
                const double center=slopes(CentralSlope,c);
                const double left=slopes(NearLeftSlope,c), right=slopes(NearRightSlope,c);
                traces.left[c]=mp5::increment(slopes(FarLeftSlope,c),left,center,right,st.mp5.alpha);
                traces.right[c]=mp5::increment(slopes(FarRightSlope,c),right,center,left,st.mp5.alpha);
            }
        } else if (st.mp5.weno)
            mp6::reconstruct<true>(slopes, width, st.mp5, reader.g.options.mp6_hybrid, traces.left, traces.right);
        else
            mp6::reconstruct<false>(slopes, width, st.mp5, reader.g.options.mp6_hybrid, traces.left, traces.right);
    } else if (!cartesian_hybrid_increments(reader, row, own, traces))
        return false;
    const bool characteristic = reader.variables == ResidualVariables::Characteristic;
    return add_increment<0>(reader,characteristic,own.row(0),traces.left,false,traces.left) &&
           add_increment<0>(reader,characteristic,own.row(1),traces.right,true,traces.right);
}

// Граница с узловыми следами уже обработана в integrated_edge_flux. Здесь обе стороны используют шаблон.
template <unsigned linear_width = 0, bool characteristic_2d = false, class FieldReader>
KOKKOS_FORCEINLINE_FUNCTION bool reconstruct_edge_traces(const FieldReader& reader, Index edge, EdgeTraces& traces) {
    const auto& stencil = reader.g.stencil;
    if (!linear_width && (stencil.mp5.enabled || stencil.mp5.weno)) {
        // Для чистой MP5 используем общие строки только после проверки их
        // совпадения при подготовке сетки. Укороченные стороны идут в обычный обход.
        const bool shared_mp5 = stencil.shared_edge_slopes && stencil.mp5.enabled && !stencil.mp5.weno &&
                                reader.variables == ResidualVariables::Characteristic;
        if ((reader.g.options.mp6_hybrid > 0. || shared_mp5) &&
            stencil.weno_order[2*edge] == 5 && stencil.weno_order[2*edge+1] == 5)
            return full_mp_edge_traces<characteristic_2d>(reader,edge,traces);
    }
    const unsigned width = characteristic_2d ? 4 : (linear_width ? linear_width : reader.g.components());
    for (unsigned side = 0; side < 2; ++side) {
        const Index node = side ? reader.surface.end : reader.surface.begin;
        const double* u = reader.field + node * width;
        double* dst = side ? traces.right : traces.left;
        double physical_flux[5], increment[5];
        const double* own = u;
        if (!characteristic_2d && !linear_width && reader.variables == ResidualVariables::Flux) {
            if (reader.physical_flux(node, u, physical_flux) != EulerStatus::Success)
                return false;
            own = physical_flux;
        }
        const Index row = 2 * edge + side;
        const bool valid = characteristic_2d || (!linear_width && stencil.weno_order)
            ? transformed_weno_increment<characteristic_2d>(reader, row, own, increment)
            : transformed_row<linear_width>(reader, node, own, stencil.linear_row(row), increment, 1);
        if (!valid)
            return false;
        const bool characteristic = characteristic_2d || (!linear_width && reader.variables == ResidualVariables::Characteristic);
        if (!add_increment<characteristic_2d ? 4 : linear_width>(reader, characteristic, characteristic ? u : own,
                                  increment, side != 0, dst))
            return false;
    }
    // Узловые следы обрабатываются до входа сюда. Из схем со строками шаблона
    // только EBR хранит линейные веса без массива порядков WENO. Для чистой EBR
    // возвращаем исходные значения без дополнительного округления.
    if (!characteristic_2d && !stencil.weno_order && reader.g.options.upw_coeff < 1)
        blend_ebr_traces(traces, width, reader.g.options.upw_coeff);
    return true;
}

// Выбирает допустимую долю приращения по ограничениям физического состояния.
AERO_HOST_DEVICE inline double limit_ratio(double base, double value, const ResidualOptions& o) {
    const double lo = base * o.trace_ratio_min, hi = base * o.trace_ratio_max;
    if (value < lo)
        return (lo - base) / (value - base);
    if (value > hi)
        return (hi - base) / (value - base);
    return 1;
}

// Ограничиваем примитивные переменные одним коэффициентом смешения.
KOKKOS_FORCEINLINE_FUNCTION bool limit_trace(unsigned d, const ResidualOptions& o, const double* node,
                                         double* trace, const double* area, double* reconstructed_flux,
                                         const EulerState* known_base = nullptr, EulerState* limited_gas = nullptr) {
    if (d < 1 || d > 3)
        return false;
    EulerState decoded;
    if (!known_base && euler_state(d, node, o.gamma, &decoded) != EulerStatus::Success)
        return false;
    const EulerState& base = known_base ? *known_base : decoded;
    EulerState candidate = {};
    double weight = 0;
    if (trace[0] > 0 && trace[0] <= 1e16) {
        const auto status = euler_detail::decode_state(d, trace, o.gamma, candidate);
        if (status != EulerStatus::Success && status != EulerStatus::InvalidPressure)
            return false;
        if (status == EulerStatus::Success && candidate.pressure > 0 && candidate.pressure <= 1e16)
            weight = ::fmin(limit_ratio(base.density, candidate.density, o),
                            limit_ratio(base.pressure, candidate.pressure, o));
        else if (status == EulerStatus::InvalidPressure && candidate.pressure <= 0) {
            // При p <= 0 оставляем запас до нуля, даже если относительная нижняя граница отключена.
            const double floor = base.pressure * ::fmax(0.01, o.trace_ratio_min);
            weight = ::fmin(limit_ratio(base.density, candidate.density, o),
                            (base.pressure - floor) / (base.pressure - candidate.pressure));
        }
    } else {
        // В обычном диапазоне все компоненты уже проверены в euler_state. При возврате
        // к узловому следу проверяем их здесь, чтобы отличить выброс плотности от NaN.
        for (unsigned k = 0; k < d + 2; ++k)
            if (!euler_detail::finite(trace[k]))
                return false;
    }
    if (weight == 1) {
        if (limited_gas)
            *limited_gas = candidate;
        return true;
    }
    if (weight == 0) {
        for (unsigned k = 0; k < d + 2; ++k)
            trace[k] = node[k];
    } else {
        const double keep = 1 - weight;
        trace[0] = keep * base.density + weight * candidate.density;
        const double p = keep * base.pressure + weight * candidate.pressure;
        double kinetic = 0;
        for (unsigned k = 0; k < d; ++k) {
            const double v = keep * base.velocity[k] + weight * candidate.velocity[k];
            trace[k + 1] = trace[0] * v;
            kinetic += 0.5 * trace[k + 1] * v;
        }
        trace[d + 1] = p / (o.gamma - 1) + kinetic;
        // При больших скоростях округление полной энергии может поглотить внутреннюю.
        if (candidate.pressure <= 0) {
            EulerState checked;
            if (euler_state(d, trace, o.gamma, &checked) != EulerStatus::Success) {
                weight = 0;
                for (unsigned k = 0; k < d + 2; ++k)
                    trace[k] = node[k];
            }
        }
    }
    if (limited_gas) {
        if (weight == 0)
            *limited_gas = base;
        else if (euler_state(d, trace, o.gamma, limited_gas) != EulerStatus::Success)
            return false;
    }
    if (reconstructed_flux) {
        // В потоковых переменных тем же коэффициентом ограничиваем и восстановленный физический поток.
        double nodal_flux[5], speed;
        if (euler_flux(d, node, area, o.gamma, nodal_flux, &speed) != EulerStatus::Success)
            return false;
        for (unsigned k = 0; k < d + 2; ++k)
            reconstructed_flux[k] = (1 - weight) * nodal_flux[k] + weight * reconstructed_flux[k];
    }
    return true;
}

// Решатель выдаёт поток на единицу площади. Умножаем компоненты на площадь один раз.
// max_wave_speed остаётся скоростью: площадь для шага CFL учитывается отдельно.
AERO_HOST_DEVICE inline bool integrate_flux(unsigned width, double measure, EulerFluxResult& out) {
    for (unsigned c = 0; c < width; ++c) {
        out.flux[c] *= measure;
        if (!euler_detail::finite(out.flux[c]))
            return false;
    }
    return true;
}

// Решатель использует скорости и давления, которые уже вычислены при ограничении следов.
template <bool roe_2d = false, bool states_only = false>
KOKKOS_FORCEINLINE_FUNCTION bool solve_edge_flux(const Geometry& g, const Surface& surface,
                                              const double* left, const double* right,
                                              const euler_detail::InterfaceGas& gas, const EdgeTraces* physical,
                                              EulerFluxResult& out, unsigned nodal_lf_dim = 0) {
    // Для LF первого порядка размерность выбирается перед обходом рёбер. Тогда
    // компилятор знает число компонент при проекции потока и при его интегрировании.
    const unsigned dim = nodal_lf_dim ? nodal_lf_dim : (roe_2d ? 2 : g.dim);
    EdgeTraces local;
    double speed;
    if (!states_only && physical) {
        if (euler_detail::interface_wave_speed(dim, gas, speed) != EulerStatus::Success)
            return false;
    } else {
        // В режиме реконструкции состояний физический поток получаем по двум следам.
        double sl, sr;
        if (euler_detail::project_flux(dim, left, gas.left, gas.normal, local.left, &sl) != EulerStatus::Success ||
            euler_detail::project_flux(dim, right, gas.right, gas.normal, local.right, &sr) != EulerStatus::Success)
            return false;
        speed = sl > sr ? sl : sr;
        physical = &local;
    }
    const euler_detail::FaceStates face = {dim, left, right, surface.area, g.options.gamma, g.options.upw_coeff};
    const auto status = roe_2d || (!nodal_lf_dim && g.options.solver == ResidualSolver::Roe)
        ? roe_detail::RoeFlux::combine<roe_2d ? 2 : 0>(face, gas, physical->left, physical->right, speed, &out)
        : euler_detail::RusanovFlux::combine(face, gas, physical->left, physical->right, speed, &out);
    return status == EulerStatus::Success && integrate_flux(dim + 2, surface.measure, out);
}

// Нормаль хранится в трёх координатах, но в 1D и 2D заполнены только рабочие оси.
// Проверяем размерность до записи, в том числе при чтении уже подготовленной нормали.
AERO_HOST_DEVICE inline bool copy_face_normal(unsigned dim, const double (&source)[3], double (&target)[3]) {
    if (dim < 1 || dim > 3)
        return false;
    for (unsigned k = 0; k < dim; ++k)
        target[k] = source[k];
    return true;
}

// Из полей узлов собирает физические величины для базиса, ограничителя и потока первого порядка.
template <bool packed_plane = false>
AERO_HOST_DEVICE inline bool edge_gas(const Geometry& g, const Surface& surface, const double* begin,
                                      const double* end, euler_detail::InterfaceGas& base) {
    // Для упакованного плоского шаблона нормали и узловой газ подготовлены вместе с сеткой.
    if (packed_plane || g.packed.gas) {
        base.left = g.packed.gas[surface.begin];
        base.right = g.packed.gas[surface.end];
        if (!base.left.density || !base.right.density)
            return false;
        if (packed_plane || g.packed.normals) {
            const auto& n = g.packed.normals[&surface - g.surfaces];
            return copy_face_normal(packed_plane ? 2u : g.dim, n.axis, base.normal);
        }
        return euler_unit_direction(g.dim, surface.area, base.normal) == EulerStatus::Success;
    }
    return euler_detail::read_interface_states(g.dim, begin, end, surface.area, g.options.gamma,
                                               base.left, base.right, base.normal) == EulerStatus::Success;
}

// После реконструкции подставляем состояние узла только с отказавшей стороны.
// Для реконструкции потоков заменяем также её физический поток в нормали ребра.
template <unsigned fixed_dim>
KOKKOS_FORCEINLINE_FUNCTION bool use_first_order_side(unsigned dim, const double* node, const EulerState& gas,
        const double* normal, double* state, double* flux) {
    const unsigned d = fixed_dim ? fixed_dim : dim;
    for (unsigned k = 0; k < d + 2; ++k) state[k] = node[k];
    double speed;
    return !flux || euler_detail::project_flux(d, node, gas, normal, flux, &speed) == EulerStatus::Success;
}

// Ограничиваем след каждой стороны относительно своего узла. В тот же момент получаем газ для потока.
template <bool roe_2d = false, bool states_only = false>
KOKKOS_FORCEINLINE_FUNCTION bool finish_edge_flux(const Geometry& g, const Surface& surface, const double* begin,
                                              const double* end, const euler_detail::InterfaceGas& base,
                                              EdgeTraces& state, EdgeTraces* physical, EulerFluxResult& out) {
    const unsigned dim = roe_2d ? 2 : g.dim;
    euler_detail::InterfaceGas gas;
    if (!copy_face_normal(dim, base.normal, gas.normal))
        return false;
    // Маска принадлежит ребру и обновляется при откате или снятии меток.
    // Этим кодом пользуются как полные шаблоны WENO/MP, так и разреженные.
    const unsigned mask = surface.use_nodal_traces;
    if (mask > 1) {
        if ((mask & 2) && !use_first_order_side<roe_2d ? 2 : 0>(dim, begin, base.left, base.normal,
                state.left, !states_only && physical ? physical->left : nullptr)) return false;
        if ((mask & 4) && !use_first_order_side<roe_2d ? 2 : 0>(dim, end, base.right, base.normal,
                state.right, !states_only && physical ? physical->right : nullptr)) return false;
    }
    // Если восстановлены состояния, ограничителю нужны только они. В режиме Flux дополнительно
    // передаём физический поток соответствующей стороны, который ограничивается тем же коэффициентом.
    if (!limit_trace(dim, g.options, begin, state.left, surface.area, !states_only && physical ? physical->left : nullptr,
                     &base.left, &gas.left) ||
        !limit_trace(dim, g.options, end, state.right, surface.area, !states_only && physical ? physical->right : nullptr,
                     &base.right, &gas.right))
        return false;
    return solve_edge_flux<roe_2d, states_only>(g, surface, state.left, state.right, gas, physical, out);
}

// Решает, нужен ли базис Роу для реконструкции или только проверка допустимости газа.
template <bool roe_2d, bool nodal_only>
KOKKOS_FORCEINLINE_FUNCTION bool prepare_edge_basis(const Geometry& g, const Surface& surface,
                                                            const euler_detail::InterfaceGas& base,
                                                            EulerRoeBasis& basis) {
    // При узловых следах поток Роу сам проверяет базис. Для характеристического LF проверяем его здесь.
    const bool reconstruct_states = !nodal_only && g.stencil.has_owners() && surface.use_nodal_traces != 1;
    const bool project_waves = reconstruct_states && g.stencil.weno_order;
    if (g.options.variables != ResidualVariables::Characteristic ||
        (!reconstruct_states && g.options.solver == ResidualSolver::Roe))
        return true;
    // Для линейной EBR достаточно проверить средний газ. Матрицы нужны только
    // при проекции наклонов WENO и MP в характеристики.
    return euler_detail::build_roe_basis<true, roe_2d ? 2 : 0>(g.dim, base.left, base.right, surface.area, base.normal,
                                      g.options.gamma, project_waves ? &basis : nullptr) == EulerStatus::Success;
}

// Для EBR одинаковая линейная строка применяется к U и F(U). Номера узлов
// и геометрические веса можно читать один раз сразу для обоих векторов.
template<unsigned width, class FieldReader>
KOKKOS_FORCEINLINE_FUNCTION bool ebr_state_and_flux(const FieldReader& reader, Index edge, EdgeTraces& state,
                                                      EdgeTraces& flux) {
    const auto& stencil = reader.g.stencil;
    for (unsigned side = 0; side < 2; ++side) {
        const auto entries = stencil.linear_row(2 * edge + side);
        const Index owner = side ? reader.surface.end : reader.surface.begin;
        const double* own = reader.field + width * owner;
        double own_flux[width], increment[2*width];
        if (reader.physical_flux(owner,own,own_flux) != EulerStatus::Success) return false;
        // Возвращает компоненты опорного поля в выбранных переменных реконструкции.
        const auto values = [&](Index k, double* pair) -> const double* {
            const Index node = entries.node(k);
            const double* u = reader.field + width*node;
            for (unsigned c = 0; c < width; ++c) pair[c] = u[c];
            if (node == owner) {
                for (unsigned c = 0; c < width; ++c) pair[width+c] = own_flux[c];
            } else if (reader.physical_flux(node,u,pair+width) != EulerStatus::Success) return nullptr;
            return pair;
        };
        // Возвращает вес опоры в текущей строке реконструкции.
        const auto weight = [&](Index k) { return entries.weight(k); };
        if (!reconstruction_detail::sum_slope<2*width>(2*width,entries.count,values,weight,increment))
            return false;
        add_half_increment(width,own,increment,side!=0,side ? state.right : state.left);
        add_half_increment(width,own_flux,increment+width,side!=0,side ? flux.right : flux.left);
    }
    if (reader.g.options.upw_coeff < 1.) {
        blend_ebr_traces(state,width,reader.g.options.upw_coeff);
        blend_ebr_traces(flux,width,reader.g.options.upw_coeff);
    }
    return true;
}

// Восстанавливает левый и правый следы на внутреннем ребре.
template<unsigned linear_width, bool characteristic_2d, bool ebr_flux = false, class FieldReader>
KOKKOS_FORCEINLINE_FUNCTION bool reconstruct_pair(const Geometry& g, Index edge, FieldReader& reader,
        EdgeTraces& state, EdgeTraces& reconstructed_flux, EdgeTraces*& physical_flux) {
    // Для обоих концов нужны две строки одного шаблона, в том числе при совместном расчёте U и F(U).
    if (edge >= g.stencil.edge_count || !g.stencil.has_owners())
        return false;
    // У EBR все компоненты имеют одинаковые постоянные веса. Для базиса этого ребра
    // R·(Σ aᵢ L·Uᵢ) = Σ aᵢ Uᵢ, поскольку R и L взаимно обратны. Считаем линейную
    // строку сразу по состояниям. У WENO веса зависят от каждой характеристики отдельно.
    if (reader.variables == ResidualVariables::Characteristic && !g.stencil.weno_order)
        reader.variables = ResidualVariables::Conservative;

    // Вариант EBR выбирается перед запуском ядра. При реконструкции состояний
    // ядру не нужны ни физические потоки во всех узлах шаблона, ни место под их приращения.
    static_assert(!ebr_flux || linear_width == 4 || linear_width == 5, "Invalid EBR component count");
    if (ebr_flux) {
        physical_flux = &reconstructed_flux;
        if (linear_width == 4) return ebr_state_and_flux<4>(reader, edge, state, reconstructed_flux);
        return ebr_state_and_flux<5>(reader, edge, state, reconstructed_flux);
    }
    // В общем ядре тип реконструкции определяется во время расчёта.
    if (!linear_width && !characteristic_2d && reader.variables == ResidualVariables::Flux) {
        physical_flux = &reconstructed_flux;
        if (!reconstruct_edge_traces(reader, edge, reconstructed_flux))
            return false;
        reader.variables = ResidualVariables::Conservative;
    }
    return reconstruct_edge_traces<linear_width, characteristic_2d>(reader, edge, state);
}

// Для линейных уравнений фильтр положительности к пульсациям неприменим:
// обе стороны могут иметь любое конечное значение, включая ноль.
AERO_HOST_DEVICE inline bool linear_edge_flux(const Geometry& g, Index edge, const double* field, EulerFluxResult& out) {
    const auto& s = g.surfaces[edge];
    double n[3] = {};
    if (euler_unit_direction(g.dim,s.area,n) != EulerStatus::Success) return false;
    const LinearFace face(g.options.background,g.dim,n);
    const double* left = g.node_row(field,s.begin);
    const double* right = g.node_row(field,s.end);
    EdgeTraces state, flux;
    EdgeTraces* physical = nullptr;
    if (s.use_nodal_traces != 1 && g.stencil.has_owners()) {
        LinearReader reader = {g,field,s,n,g.options.variables,face,{}};
        if (reader.variables == ResidualVariables::Characteristic)
            euler_detail::face_tangents(g.dim,s.area,n,reader.tangent);
        if (!reconstruct_pair<0,false>(g,edge,reader,state,flux,physical)) return false;
        left = state.left;
        right = state.right;
    }
    if (!physical) {
        face.physical(left,flux.left);
        face.physical(right,flux.right);
    }
    return face.numerical(g.options.solver == ResidualSolver::Roe,g.options.upw_coeff,left,right,
                          flux.left,flux.right,out) && integrate_flux(g.components(),s.measure,out);
}

// Вычисляет поток внутренних следов с однократным умножением на площадь.
template <bool roe_2d = false, bool nodal_only = false, unsigned linear_width = 0, bool characteristic_2d = false,
          bool ebr_flux = false>
KOKKOS_FORCEINLINE_FUNCTION bool integrated_edge_flux(const Geometry& g, Index edge, const double* field,
                                                  EulerFluxResult& out, unsigned nodal_lf_dim = 0) {
    const Surface& surface = g.surfaces[edge];
    const double* begin = g.node_row(field, surface.begin);
    const double* end = g.node_row(field, surface.end);
    euler_detail::InterfaceGas base;
    if (!edge_gas(g, surface, begin, end, base))
        return false;
    EulerRoeBasis basis{};
    Reader reader = {g, field, surface, basis, base.normal, g.options.variables};
    if (!prepare_edge_basis<roe_2d, nodal_only>(g, surface, base, basis))
        return false;
    if (nodal_only || !g.stencil.has_owners() || surface.use_nodal_traces == 1)
        return solve_edge_flux<roe_2d>(g, surface, begin, end, base, nullptr, out, nodal_lf_dim);

    EdgeTraces state, reconstructed_flux;
    EdgeTraces* physical_flux = nullptr;
    if (!reconstruct_pair<linear_width,characteristic_2d,ebr_flux>(g,edge,reader,state,reconstructed_flux,physical_flux))
        return false;
    return finish_edge_flux<roe_2d>(g, surface, begin, end, base, state, physical_flux, out);
}

// У внешней площадки один узел. Граничное условие сразу возвращает поток через всю площадку наружу.
AERO_HOST_DEVICE inline bool integrated_boundary_flux(const Geometry& g, Index face, const double* state,
                                                      const double* initial, EulerFluxResult& out) {
    const auto& piece = g.surfaces[face];
    const auto patch = g.patches.by_patch[piece.patch];
    const double* initial_state = g.boundary_initial(initial, face);
    if (g.options.background.active()) {
        double n[3] = {};
        if (euler_unit_direction(g.dim,piece.area,n) != EulerStatus::Success) return false;
        return linear_boundary_flux(LinearFace(g.options.background,g.dim,n),patch,g.patches.free_rules,
                     g.node_row(state,piece.begin),initial_state,false,out) && integrate_flux(g.components(),piece.measure,out);
    }
    return boundary_flux(g.dim, patch, g.patches.free_rules, g.node_row(state, piece.begin), initial_state,
                         piece.area, g.options.gamma, &out, g.patches.profile(face-g.edges)) == BoundaryStatus::Success;
}

// При отсутствии списка индексов обходим все площадки в исходном порядке.
struct FaceBatch {
    Geometry geometry;        // Геометрия и схемы для всей локальной сетки.
    const MeshIndex* indices; // Поверхности данного процесса. nullptr означает последовательную нумерацию.
    Index count;              // Число поверхностей, на которых нужно вычислить поток.
    const double *state, *initial;         // Текущее и начальное состояния, необходимые граничному условию.
    EulerFluxResult* flux;                 // Одна запись на поверхность, общая для её узлов.
    double *edge_speeds, *boundary_speeds; // Необязательные выходные массивы скоростей волн.
    bool gas_current; // Стадия уже пересчитала газ во всех узлах этого поля.
    // Переключаем только представление для данной сборки потоков. Шаблоны
    // и локальные метки стабилизации остаются готовыми к использованию после t_fo.
    void use_first_order() {
        geometry.stencil = ReconstructionStencilView{};
        geometry.packed.tiles = nullptr;
    }
    // Определяет, требуется ли обновить кэш примитивных газовых параметров.
    bool needs_gas_update() const { return geometry.packed.gas && !gas_current; }
    // При плотном обходе k уже является номером грани. Выделенная группа
    // или подобласть MPI передаёт свой список; для GPU достаточно его адреса.
    AERO_HOST_DEVICE static Index face(const MeshIndex* selected, Index k) { return selected ? selected[k] : k; }
    // При плотном обходе k уже является номером грани. Выделенная группа
    // или подобласть MPI передаёт свой список; для GPU достаточно его адреса.
    AERO_HOST_DEVICE Index face(Index k) const { return face(indices, k); }
    // Для первого порядка размерность и решатель известны до обхода. Проверки и формулы потока общие.
    AERO_HOST_DEVICE bool roe_2d() const {
        return geometry.dim == 2 && geometry.options.solver == ResidualSolver::Roe;
    }
    // Первый порядок LF читает только два узла. Размерность выбираем один раз
    // до обхода всего списка, чтобы ядро работало с постоянной шириной состояния.
    bool nodal_lf() const { return !geometry.stencil.has_owners() && geometry.options.solver == ResidualSolver::LocalLF; }
    // Строит двумерный базис Роу по узловым значениям.
    AERO_HOST_DEVICE bool nodal_roe_2d() const {
        return roe_2d() && !geometry.stencil.has_owners();
    }
    // В двумерном EBR3 и EBR5 меняются только веса в строках. Потоковым переменным
    // нужен отдельный физический поток в каждом узле шаблона, для них оставляем общий обход.
    AERO_HOST_DEVICE bool linear_roe_2d() const {
        return roe_2d() && geometry.stencil.has_owners() && !geometry.stencil.weno_order &&
               geometry.options.variables != ResidualVariables::Flux;
    }
    // Вычисляет численный поток внутренней площадки между узлами.
    template <bool roe, bool nodal, unsigned linear_width = 0, bool characteristic_2d = false, bool ebr_flux = false>
    AERO_HOST_DEVICE bool compute_edge(Index f, unsigned nodal_lf_dim = 0) const {
        if (!integrated_edge_flux<roe, nodal, linear_width, characteristic_2d, ebr_flux>(geometry, f, state, flux[f], nodal_lf_dim))
            return false;
        save_speed(f);
        return true;
    }
    // Вычисляет поток внешней площадки с заданным граничным условием.
    AERO_HOST_DEVICE bool compute_boundary(Index f) const {
        if (!integrated_boundary_flux(geometry, f, state, initial, flux[f]))
            return false;
        save_speed(f);
        return true;
    }
    // Список MPI может перемежать внутренние рёбра и граничные площадки.
    // Для раздельных диапазонов исполнитель вызывает нужную функцию напрямую.
    AERO_HOST_DEVICE bool is_boundary(Index face) const { return face >= geometry.edges; }
    // Выбирает внутренний или граничный расчёт для данной площадки.
    template <bool roe_2d = false, bool nodal = false, unsigned linear_width = 0, bool ebr_flux = false>
    AERO_HOST_DEVICE bool compute_face(Index face, unsigned lf_dim = 0) const {
        return is_boundary(face) ? compute_boundary(face) : compute_edge<roe_2d,nodal,linear_width,false,ebr_flux>(face,lf_dim);
    }
    // Вычисляет поток площадки для линеаризованной системы Эйлера.
    AERO_HOST_DEVICE bool compute_linear(Index f) const {
        if (is_boundary(f)) return compute_boundary(f);
        if (!linear_edge_flux(geometry,f,state,flux[f])) return false;
        save_speed(f);
        return true;
    }
    // Сохраняет скорость волн площадки для последующего ограничения CFL.
    AERO_HOST_DEVICE void save_speed(Index f) const {
        if (f < geometry.edges) {
            if (edge_speeds)
                edge_speeds[f] = flux[f].max_wave_speed;
        } else if (boundary_speeds)
            boundary_speeds[f - geometry.edges] = flux[f].max_wave_speed;
    }
};
// Вычисляет потоки набора внутренних и граничных площадок на CPU.
void cpu_face_fluxes(const FaceBatch&, const MeshIndex* global_nodes = nullptr);
// Готовит единичные нормали площадок на GPU.
void gpu_prepare_face_normals(const Geometry&);
// Вычисляет потоки набора площадок на GPU.
void gpu_face_fluxes(const FaceBatch&, Index* bad, void* stream, const FaceBatch* device_faces = nullptr);

// Собирает R одного узла в row. Один поток исполнения пишет все его компоненты.
template <unsigned fixed_width = 0>
AERO_HOST_DEVICE inline bool assemble_node_residual(const Geometry& g, Index node,
                                                    const EulerFluxResult* flux, double* row) {
    const unsigned width = fixed_width ? fixed_width : g.components();
    double sum[5] = {};
    // Обходим только смежные поверхности в зафиксированном при подготовке порядке.
    for (Index i = g.offsets[node]; i < g.offsets[node + 1]; ++i) {
        const OrientedFace entry = g.incidence[i];
        const auto& stored = flux[entry.face()];
        // Поток наружу даёт +F, противоположная нормаль даёт −F. Площадь уже учтена.
        for (unsigned c = 0; c < width; ++c)
            sum[c] += entry.outward_flux(stored.flux[c]);
    }
    for (unsigned c = 0; c < width; ++c) {
#ifdef AERO_HAS_VISCOSITY
        if (g.options.diffusion) {
            sum[c] += g.options.diffusion[node].residual[c];
            // На неподвижной стенке скорость закреплена. Плотность и энергия остаются неизвестными.
            if (c > 0 && c+1 < width && g.options.wall[node]) sum[c] = 0.;
        }
#endif
        if (!euler_detail::finite(sum[c]))
            return false;
        row[c] = sum[c]; // Интегратор применит минус и разделит эту сумму на объём узла.
    }
    return true;
}
}
}
#endif
