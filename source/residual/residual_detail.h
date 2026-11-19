#ifndef AERO_RESIDUAL_DETAIL_H
#define AERO_RESIDUAL_DETAIL_H
#include "../parallel/kernel_math.h"
#include "residual.h"
#include "../mesh/mesh_index.h"
#include "../bc/bc.h"
#include "../eulersolver/roe.h"
#include "../eulersolver/lax_friedrichs.h"
#include "../reconstruction/reconstruction.h"
namespace aero {
namespace residual_detail {
using Index = ReconstructionIndex;
struct Surface {
    Index begin;           // Начало ребра или единственный узел физической площадки.
    Index end;             // Второй узел внутреннего ребра.
    Index patch;           // Метка физической площадки для выбора граничного условия.
    double area[3];        // Вектор площади: от begin к end на ребре, наружу на границе.
    double measure;        // Модуль вектора площади.
    bool use_nodal_traces; // На ребре у открытой границы берём значения прямо из узлов.
};

// Запись списка узла хранит номер поверхности и направление её нормали относительно узла.
struct OrientedFace {
    Index encoded; // Младший бит отмечает нормаль, направленную внутрь контрольного объёма.

    AERO_HOST_DEVICE static OrientedFace outward(Index face) { return {2 * face}; }
    AERO_HOST_DEVICE static OrientedFace inward(Index face) { return {2 * face + 1}; }
    AERO_HOST_DEVICE Index face() const { return encoded / 2; }

    // Приводит компоненту сохранённого потока к наружной нормали этого узла.
    AERO_HOST_DEVICE double outward_flux(double component) const {
        return (encoded & 1) ? -component : component;
    }
};
#if !defined(__CUDACC_RTC__) && !defined(AERO_RESIDUAL_KERNEL_ONLY)
void project_surface(Surface&, CvVector area, unsigned dimension, const ResidualFrame&);
void build_incidence(Index nodes, Index edges, const std::vector<Surface>& surfaces,
                     std::vector<Index>& offsets, std::vector<OrientedFace>& incidence);
inline bool needs_nodal_traces(BoundaryType type) {
    // На открытой метке режим меняется во времени. Для всех её узлов нужны внутренние состояния.
    return type == BoundaryType::Free;
}
#endif
// Порядок строк в упаковке обоих концов ребра. Центральная и ближние строки хранятся по одному разу.
enum WenoSlopeRow : unsigned { CentralSlope, NearLeftSlope, NearRightSlope, FarLeftSlope, FarRightSlope };

// Число узлов в наклоне: концы ребра, интерполяция ближней грани либо разность двух интерполяций.
AERO_HOST_DEVICE constexpr unsigned slope_terms(unsigned row) {
    return row == CentralSlope ? 2 : (row <= NearRightSlope ? 3 : 4);
}

// Номер первого слагаемого в списке всех строк. При чтении шаблона известен при компиляции.
AERO_HOST_DEVICE constexpr unsigned first_slope_term(unsigned row) {
    return row ? first_slope_term(row - 1) + slope_terms(row - 1) : 0;
}

// В плитке соседние потоки CUDA читают подряд данные одного и того же элемента шаблона.
// Для полного двумерного WENO5 пять разных наклонов содержат 2, 3, 3, 4 и 4 слагаемых.
template <unsigned scheme_order> struct WenoEdgeTile {
    static constexpr unsigned nodes_per_edge = scheme_order == 5 ? 10 : 6;
    static constexpr unsigned rows_per_edge = scheme_order == 5 ? 5 : 3;
    unsigned nodes[nodes_per_edge][32];    // Индексы в поле узлов. WENO3 использует 6 узлов, WENO5 — 10.
    Index slots[32];           // По четыре бита на номер узла каждого слагаемого. Все единицы — общий путь.
    double weights[first_slope_term(rows_per_edge) - rows_per_edge][32];    // В каждой разностной строке коэффициент опорного узла не используется.
};
struct FaceNormal { double axis[3]; }; // Единичный вектор в базисе решателя.
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
        FaceNormal* normals;        // Единичные нормали рёбер, подготовленные на текущем устройстве.
        EulerState* gas;            // Физические величины в узлах текущей стадии.
        Index state_nodes;          // Включает удалённые узлы шаблона.
    } packed;

    AERO_HOST_DEVICE unsigned components() const { return dim + 2; } // Плотность, импульс и энергия.
    template <class T> AERO_HOST_DEVICE T* node_row(T* field, Index node) const {
        return field + node * components();
    }
};
struct Reader {
    const Geometry& g;           // Размерность, параметры и готовый шаблон.
    const double* field;         // Консервативные переменные текущей стадии по узлам.
    const Surface& surface;      // Ребро, для которого восстанавливаем состояния с двух сторон.
    const EulerRoeBasis& basis;  // Один базис Роу для всех узлов шаблона данного ребра.
    const double* normal;        // Единичная нормаль этого базиса.
    ResidualVariables variables; // Величины, к которым применяются веса шаблона.
    AERO_HOST_DEVICE EulerStatus project(const double* u, double* q) const {
        return euler_detail::project_characteristic(basis, normal, g.options.gamma,
                                                    basis.kinetic_energy(), u, q);
    }
    AERO_HOST_DEVICE EulerStatus transform(const double* u, double* q) const {
        if (variables == ResidualVariables::Characteristic)
            return project(u, q);
        if (variables == ResidualVariables::Flux) {
            double speed;
            if (g.packed.gas) {
                const auto& gas = g.packed.gas[(u - field) / g.components()];
                if (!gas.density)
                    return EulerStatus::InvalidDensity;
                return euler_detail::project_flux(g.dim, u, gas, normal, q, &speed);
            }
            return euler_flux(g.dim, u, surface.area, g.options.gamma, q, &speed);
        }
        for (unsigned k = 0; k < g.dim + 2; ++k)
            q[k] = u[k];
        return EulerStatus::Success;
    }
    // NaN проходит через реконструкцию и обнаруживается при проверке перед использованием результата.
    AERO_HOST_DEVICE double operator()(Index, Index node, Index component) const {
        const double* u = field + node * (g.dim + 2);
        if (variables == ResidualVariables::Conservative)
            return u[component];
        double q[5];
        return transform(u, q) == EulerStatus::Success ? q[component] : ::nan("");
    }
};

// Складываем сразу все компоненты выбранных переменных в исходном порядке разреженной строки.
// Первый узел даёт опорное значение разностной формулы с нулевой суммой весов.
AERO_HOST_DEVICE inline bool transformed_row(const Reader& reader, Index owner, const double* own,
                                                const Index* offsets, const Index* nodes,
                                                const double* weights, Index row, double* sum,
                                                unsigned stride) {
    const unsigned width = reader.g.dim + 2;
    for (unsigned c = 0; c < width; ++c)
        sum[c * stride] = 0;
    const Index begin = offsets[row], end = offsets[row + 1];
    if (begin == end)
        return true;
    // Вычитаем значение в первом узле строки: на постоянном поле остаётся точный нуль.
    double first[5];
    const double* reference = own;
    if (nodes[begin] != owner) {
        if (reader.transform(reader.field + nodes[begin] * width, first) != EulerStatus::Success)
            return false;
        reference = first;
    }
    for (Index i = begin + 1; i < end; ++i) {
        double projected[5];
        const double* q = own;
        if (nodes[i] != owner) {
            if (reader.transform(reader.field + nodes[i] * width, projected) != EulerStatus::Success)
                return false;
            q = projected;
        }
        for (unsigned c = 0; c < width; ++c)
            sum[c * stride] += weights[i] * (q[c] - reference[c]);
    }
    return true;
}

AERO_HOST_DEVICE inline bool transformed_weno_increment(const Reader& reader, Index row, const double* own,
                                                           double* out) {
    const auto& s = reader.g.stencil;
    const Index owner = s.row_owner[row];
    const unsigned order = s.weno_order[row];
    if (!order)
        return transformed_row(reader, owner, own, s.offsets, s.nodes, s.weights, row, out, 1);
    // Проецируем в базис ребра сразу все компоненты, затем применяем веса WENO к каждой отдельно.
    double slopes[5 * 4];
    for (unsigned k = 0; k < (order == 5 ? 4u : 2u); ++k)
        if (!transformed_row(reader, owner, own, s.slope_offsets, s.slope_nodes, s.slope_weights,
                                4 * row + k, slopes + k, 4))
            return false;
    for (unsigned c = 0; c < reader.g.dim + 2; ++c)
        out[c] = weno_increment2(order, slopes + 4 * c, s.weno_epsilon_scale[row]);
    return true;
}

// Восстановленные значения с двух сторон одной грани, до решения задачи о распаде разрыва.
struct EdgeTraces {
    double left[5];  // Значение со стороны begin.
    double right[5]; // Значение со стороны end.
};

// Приращение схемы вдвое больше сдвига от узла к середине ребра.
// При работе с волнами сначала переводим только приращение обратно в консервативные переменные.
template <unsigned fixed_width = 0>
AERO_HOST_DEVICE inline bool trace_from_increment(unsigned width, const EulerRoeBasis* basis, const double* own,
                                                  double* increment, bool right, double* trace) {
    if (basis && euler_detail::expand_characteristic<fixed_width>(width, *basis, increment, increment) != EulerStatus::Success)
        return false;
    for (unsigned k = 0; k < width; ++k)
        trace[k] = own[k] + (right ? -0.5 : 0.5) * increment[k];
    return true;
}

// Граница с узловыми следами уже обработана в integrated_edge_flux. Здесь обе стороны используют шаблон.
AERO_HOST_DEVICE inline bool reconstruct_edge_traces(const Reader& reader, Index edge, EdgeTraces& traces) {
    const unsigned width = reader.g.dim + 2;
    for (unsigned side = 0; side < 2; ++side) {
        const Index node = side ? reader.surface.end : reader.surface.begin;
        const double* u = reader.g.node_row(reader.field, node);
        double* dst = side ? traces.right : traces.left;
        double own[5], increment[5];
        if (reader.transform(u, own) != EulerStatus::Success)
            return false;
        // Шаблон возвращает удвоенное приращение, ориентированное от begin к end для обеих сторон.
        const auto which = side ? ReconstructionSide::Right : ReconstructionSide::Left;
        if (reader.variables != ResidualVariables::Conservative) {
            const Index row = 2 * edge + side;
            const auto& stencil = reader.g.stencil;
            // Для EBR и WENO читаем и преобразуем опорный узел сразу для всех компонент.
            // EBR хранит готовую линейную строку, а WENO комбинирует строки наклонов по текущему полю.
            const bool valid = stencil.weno_order
                ? transformed_weno_increment(reader, row, own, increment)
                : transformed_row(reader, node, own, stencil.offsets, stencil.nodes, stencil.weights, row, increment, 1);
            if (!valid)
                return false;
        } else if (reconstruct_increment2(reader.g.stencil, edge, which, own, width, reader, increment) !=
                   ReconstructionStatus::Success)
            return false;
        const bool characteristic = reader.variables == ResidualVariables::Characteristic;
        if (!trace_from_increment(width, characteristic ? &reader.basis : nullptr, characteristic ? u : own,
                                  increment, side != 0, dst))
            return false;
    }
    return true;
}

AERO_HOST_DEVICE inline double limit_ratio(double base, double value, const ResidualOptions& o) {
    const double lo = base * o.trace_ratio_min, hi = base * o.trace_ratio_max;
    if (value < lo)
        return (lo - base) / (value - base);
    if (value > hi)
        return (hi - base) / (value - base);
    return 1;
}

// Вне абсолютных границ плотности или давления используем узловое состояние.
// Внутри допустимого диапазона ограничиваем все примитивные переменные одним коэффициентом смешения.
AERO_HOST_DEVICE inline bool limit_trace(unsigned d, const ResidualOptions& o, const double* node,
                                         double* trace, const double* area, double* reconstructed_flux,
                                         const EulerState* known_base = nullptr, EulerState* limited_gas = nullptr) {
    EulerState decoded;
    if (!known_base && euler_state(d, node, o.gamma, &decoded) != EulerStatus::Success)
        return false;
    const EulerState& base = known_base ? *known_base : decoded;
    for (unsigned k = 0; k < d + 2; ++k)
        if (!euler_detail::finite(trace[k]))
            return false;
    // По умолчанию возвращаемся к узловому состоянию. Кандидат участвует только при допустимых rho и p.
    EulerState candidate = {};
    double weight = 0;
    if (trace[0] >= 1e-16 && trace[0] <= 1e16) {
        const auto status = euler_state(d, trace, o.gamma, &candidate);
        if (status != EulerStatus::Success && status != EulerStatus::InvalidPressure)
            return false;
        if (status == EulerStatus::Success && candidate.pressure >= 1e-16 && candidate.pressure <= 1e16)
            weight = ::fmin(limit_ratio(base.density, candidate.density, o),
                            limit_ratio(base.pressure, candidate.pressure, o));
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
AERO_HOST_DEVICE inline bool solve_edge_flux(const Geometry& g, const Surface& surface,
                                              const double* left, const double* right,
                                              const euler_detail::InterfaceGas& gas, const EdgeTraces* physical,
                                              EulerFluxResult& out) {
    EdgeTraces local;
    double speed;
    if (physical) {
        if (euler_detail::interface_wave_speed(g.dim, gas, speed) != EulerStatus::Success)
            return false;
    } else {
        // В режиме реконструкции состояний физический поток получаем по двум следам.
        double sl, sr;
        if (euler_detail::project_flux(g.dim, left, gas.left, gas.normal, local.left, &sl) != EulerStatus::Success ||
            euler_detail::project_flux(g.dim, right, gas.right, gas.normal, local.right, &sr) != EulerStatus::Success)
            return false;
        speed = sl > sr ? sl : sr;
        physical = &local;
    }
    const euler_detail::FaceStates face = {g.dim, left, right, surface.area, g.options.gamma};
    const auto status = g.options.solver == ResidualSolver::Roe
        ? roe_detail::RoeFlux::combine(face, gas, physical->left, physical->right, speed, &out)
        : euler_detail::RusanovFlux::combine(face, gas, physical->left, physical->right, speed, &out);
    return status == EulerStatus::Success && integrate_flux(g.components(), surface.measure, out);
}

// Из полей узлов собирает физические величины для базиса, ограничителя и потока первого порядка.
AERO_HOST_DEVICE inline bool edge_gas(const Geometry& g, const Surface& surface, const double* begin,
                                      const double* end, euler_detail::InterfaceGas& base) {
    if (g.packed.gas) {
        base.left = g.packed.gas[surface.begin];
        base.right = g.packed.gas[surface.end];
        if (!base.left.density || !base.right.density)
            return false;
        if (g.packed.normals) {
            const auto& n = g.packed.normals[&surface - g.surfaces];
            for (unsigned k = 0; k < g.dim; ++k)
                base.normal[k] = n.axis[k];
            return true;
        }
        return euler_unit_direction(g.dim, surface.area, base.normal) == EulerStatus::Success;
    }
    return euler_detail::read_interface_states(g.dim, begin, end, surface.area, g.options.gamma,
                                               base.left, base.right, base.normal) == EulerStatus::Success;
}

// Ограничиваем след каждой стороны относительно своего узла. В тот же момент получаем газ для потока.
AERO_HOST_DEVICE inline bool finish_edge_flux(const Geometry& g, const Surface& surface, const double* begin,
                                              const double* end, const euler_detail::InterfaceGas& base,
                                              EdgeTraces& state, EdgeTraces* physical, EulerFluxResult& out) {
    euler_detail::InterfaceGas gas;
    for (unsigned k = 0; k < g.dim; ++k)
        gas.normal[k] = base.normal[k];
    if (!limit_trace(g.dim, g.options, begin, state.left, surface.area, physical ? physical->left : nullptr,
                     &base.left, &gas.left) ||
        !limit_trace(g.dim, g.options, end, state.right, surface.area, physical ? physical->right : nullptr,
                     &base.right, &gas.right))
        return false;
    return solve_edge_flux(g, surface, state.left, state.right, gas, physical, out);
}

template <bool nodal_only = false>
AERO_HOST_DEVICE inline bool integrated_edge_flux(const Geometry& g, Index edge, const double* field,
                                                  EulerFluxResult& out) {
    const Surface& surface = g.surfaces[edge];
    const double* begin = g.node_row(field, surface.begin);
    const double* end = g.node_row(field, surface.end);
    euler_detail::InterfaceGas base;
    if (!edge_gas(g, surface, begin, end, base))
        return false;
    EulerRoeBasis basis{};
    Reader reader = {g, field, surface, basis, base.normal, g.options.variables};
    // При узловых следах поток Роу сам проверяет базис. Для характеристического LF проверяем его здесь.
    if (reader.variables == ResidualVariables::Characteristic &&
        ((!nodal_only && !surface.use_nodal_traces) || g.options.solver != ResidualSolver::Roe) &&
        euler_detail::build_roe_basis(g.dim, base.left, base.right, surface.area, base.normal,
                                      g.options.gamma, &basis) != EulerStatus::Success)
        return false;
    if (nodal_only || surface.use_nodal_traces)
        return solve_edge_flux(g, surface, begin, end, base, nullptr, out);

    // В режиме flux нужны две пары: физические потоки и консервативные состояния для диссипации.
    EdgeTraces reconstructed_flux;
    EdgeTraces* physical_flux = nullptr;
    if (reader.variables == ResidualVariables::Flux) {
        physical_flux = &reconstructed_flux;
        if (!reconstruct_edge_traces(reader, edge, reconstructed_flux))
            return false;
        reader.variables = ResidualVariables::Conservative;
    }
    EdgeTraces state;
    if (!reconstruct_edge_traces(reader, edge, state))
        return false;
    return finish_edge_flux(g, surface, begin, end, base, state, physical_flux, out);
}

// Вычисляет значение для одной записи в общем буфере: сначала рёбра, затем физические площадки.
AERO_HOST_DEVICE inline bool integrated_surface_flux(const Geometry& g, Index face, const double* state,
                                                     const double* initial, EulerFluxResult& out) {
    if (face < g.edges)
        return integrated_edge_flux(g, face, state, out);

    // У внешней площадки один узел. Граничное условие сразу возвращает поток через всю площадку наружу.
    const auto& piece = g.surfaces[face];
    const auto patch = g.patches.by_patch[piece.patch];
    const double* initial_state = initial ? g.node_row(initial, piece.begin) : nullptr;
    return boundary_flux(g.dim, patch, g.patches.free_rules, g.node_row(state, piece.begin), initial_state,
                         piece.area, g.options.gamma, &out) == BoundaryStatus::Success;
}

// При отсутствии списка индексов обходим все площадки в исходном порядке.
struct FaceBatch {
    Geometry geometry;        // Геометрия и схемы для всей локальной сетки.
    const MeshIndex* indices; // Поверхности данного процесса. nullptr означает последовательную нумерацию.
    Index count;              // Число поверхностей, на которых нужно вычислить поток.
    const double *state, *initial;         // Текущее и начальное состояния, необходимые граничному условию.
    EulerFluxResult* flux;                 // Одна запись на поверхность, общая для её узлов.
    double *edge_speeds, *boundary_speeds; // Необязательные выходные массивы скоростей волн.
    AERO_HOST_DEVICE Index face(Index k) const { return indices ? indices[k] : k; }
    // Для первого порядка размерность и решатель известны до обхода. Проверки и формулы потока общие.
    AERO_HOST_DEVICE bool nodal_roe_2d() const {
        return geometry.dim == 2 && geometry.options.solver == ResidualSolver::Roe && !geometry.stencil.row_owner;
    }
    AERO_HOST_DEVICE bool compute_nodal_roe_2d(Index f) const {
        if (f >= geometry.edges)
            return compute(f);
        Geometry g = geometry;
        g.dim = 2;
        g.options.solver = ResidualSolver::Roe;
        if (!integrated_edge_flux<true>(g, f, state, flux[f]))
            return false;
        save_speed(f);
        return true;
    }
    AERO_HOST_DEVICE bool compute(Index f) const {
        // Сохраняем поток даже тогда, когда вызывающей стороне не нужны отдельные массивы скоростей.
        if (!integrated_surface_flux(geometry, f, state, initial, flux[f]))
            return false;
        save_speed(f);
        return true;
    }
    AERO_HOST_DEVICE void save_speed(Index f) const {
        if (f < geometry.edges) {
            if (edge_speeds)
                edge_speeds[f] = flux[f].max_wave_speed;
        } else if (boundary_speeds)
            boundary_speeds[f - geometry.edges] = flux[f].max_wave_speed;
    }
};
void cpu_face_fluxes(const FaceBatch&, const MeshIndex* global_nodes = nullptr);
void gpu_prepare_face_normals(const Geometry&);
void gpu_face_fluxes(const FaceBatch&, Index* bad, void* stream);

// Собирает R одного узла в row. Один поток исполнения пишет все его компоненты.
AERO_HOST_DEVICE inline bool assemble_node_residual(const Geometry& g, Index node,
                                                    const EulerFluxResult* flux, double* row) {
    double sum[5] = {};
    // Обходим только смежные поверхности в зафиксированном при подготовке порядке.
    for (Index i = g.offsets[node]; i < g.offsets[node + 1]; ++i) {
        const OrientedFace entry = g.incidence[i];
        const auto& stored = flux[entry.face()];
        // Поток наружу даёт +F, противоположная нормаль даёт −F. Площадь уже учтена.
        for (unsigned c = 0; c < g.components(); ++c)
            sum[c] += entry.outward_flux(stored.flux[c]);
    }
    for (unsigned c = 0; c < g.components(); ++c) {
        if (!euler_detail::finite(sum[c]))
            return false;
        row[c] = sum[c]; // Интегратор применит минус и разделит эту сумму на объём узла.
    }
    return true;
}
}
}
#endif
