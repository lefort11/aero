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
struct Geometry {
    unsigned dim;              // Число компонент импульса.
    Index nodes, edges, faces; // faces включает рёбра и следующие за ними физические площадки.
    const Surface* surfaces;   // Геометрия в порядке буфера потоков.
    const Index* offsets;      // Узлу i принадлежат записи от offsets[i] до offsets[i+1], не включая конец.
    const OrientedFace* incidence;     // Поверхности узла со знаками для суммирования.
    BoundaryPatchView patches;         // Метки и отдельные данные для free на текущем устройстве.
    ReconstructionStencilView stencil; // Геометрические веса для восстановления состояний на рёбрах.
    ResidualOptions options;           // Переменные реконструкции, газ и решатель потока.

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
        // Компонента энергии у энтропийного собственного вектора равна
        // v_Roe
        // ²/2.
        return euler_detail::project_characteristic(basis, normal, g.options.gamma,
                                                    basis.right[5 * (g.dim + 1) + 1], u, q);
    }
    AERO_HOST_DEVICE EulerStatus transform(const double* u, double* q) const {
        if (variables == ResidualVariables::Characteristic)
            return project(u, q);
        if (variables == ResidualVariables::Flux) {
            double speed;
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

// Складываем все характеристические компоненты в исходном порядке разреженной строки.
// Первый узел даёт опорное значение разностной формулы с нулевой суммой весов.
AERO_HOST_DEVICE inline bool characteristic_row(const Reader& reader, Index owner, const double* own,
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
        if (reader.project(reader.field + nodes[begin] * width, first) != EulerStatus::Success)
            return false;
        reference = first;
    }
    for (Index i = begin + 1; i < end; ++i) {
        double projected[5];
        const double* q = own;
        if (nodes[i] != owner) {
            if (reader.project(reader.field + nodes[i] * width, projected) != EulerStatus::Success)
                return false;
            q = projected;
        }
        for (unsigned c = 0; c < width; ++c)
            sum[c * stride] += weights[i] * (q[c] - reference[c]);
    }
    return true;
}

AERO_HOST_DEVICE inline bool characteristic_weno_increment(const Reader& reader, Index row, const double* own,
                                                           double* out) {
    const auto& s = reader.g.stencil;
    const Index owner = s.row_owner[row];
    const unsigned order = s.weno_order[row];
    if (!order)
        return characteristic_row(reader, owner, own, s.offsets, s.nodes, s.weights, row, out, 1);
    // Проецируем в базис ребра сразу все компоненты, затем применяем веса WENO к каждой отдельно.
    double slopes[5 * 4];
    for (unsigned k = 0; k < (order == 5 ? 4u : 2u); ++k)
        if (!characteristic_row(reader, owner, own, s.slope_offsets, s.slope_nodes, s.slope_weights,
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

AERO_HOST_DEVICE inline bool reconstruct_edge_traces(const Reader& reader, Index edge, EdgeTraces& traces) {
    const unsigned width = reader.g.dim + 2;
    for (unsigned side = 0; side < 2; ++side) {
        const Index node = side ? reader.surface.end : reader.surface.begin;
        const double* u = reader.g.node_row(reader.field, node);
        double* dst = side ? traces.right : traces.left;
        // У открытой границы поток на соседнем ребре строится по самим концам ребра.
        if (reader.surface.use_nodal_traces) {
            if (reader.variables == ResidualVariables::Flux) {
                if (reader.transform(u, dst) != EulerStatus::Success)
                    return false;
            } else {
                for (unsigned k = 0; k < width; ++k)
                    dst[k] = u[k];
            }
            continue;
        }
        double own[5], increment[5];
        if (reader.transform(u, own) != EulerStatus::Success)
            return false;
        // Шаблон возвращает удвоенное приращение, ориентированное от begin к end для обеих сторон.
        const auto which = side ? ReconstructionSide::Right : ReconstructionSide::Left;
        if (reader.variables == ResidualVariables::Characteristic && reader.g.stencil.weno_order) {
            if (!characteristic_weno_increment(reader, 2 * edge + side, own, increment))
                return false;
        } else if (reconstruct_increment2(reader.g.stencil, edge, which, own, width, reader, increment) !=
                   ReconstructionStatus::Success)
            return false;
        if (reader.variables == ResidualVariables::Characteristic) {
            // Базис линейный: переводим обратно только приращение. На постоянном поле исходное U сохраняется точно.
            if (characteristic_to_conservative(reader.basis, increment, increment) != EulerStatus::Success)
                return false;
            for (unsigned k = 0; k < width; ++k)
                own[k] = u[k];
        }
        // От begin движемся по направлению ребра, от end — навстречу ему.
        for (unsigned k = 0; k < width; ++k)
            dst[k] = own[k] + (side ? -0.5 : 0.5) * increment[k];
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
                                         double* trace, const double* area, double* reconstructed_flux) {
    EulerState base;
    if (euler_state(d, node, o.gamma, &base) != EulerStatus::Success)
        return false;
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
    if (weight == 1)
        return true;
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

// Выбирает численный поток вдоль единичной нормали. В режиме flux физические потоки уже восстановлены.
AERO_HOST_DEVICE inline EulerStatus solve_edge_flux(const Geometry& g, const Surface& surface,
                                                    const EdgeTraces& state, const EdgeTraces* physical_flux,
                                                    EulerFluxResult& result) {
    const double gamma = g.options.gamma;
    const bool roe = g.options.solver == ResidualSolver::Roe;
    if (physical_flux) {
        const auto& flux = *physical_flux;
        return roe ? roe_fluxes(g.dim, flux.left, flux.right, state.left, state.right, surface.area, gamma,
                                &result)
                   : lax_friedrichs_fluxes(g.dim, flux.left, flux.right, state.left, state.right,
                                           surface.area, gamma, &result);
    }
    return roe ? roe_conservative(g.dim, state.left, state.right, surface.area, gamma, &result)
               : lax_friedrichs_conservative(g.dim, state.left, state.right, surface.area, gamma, &result);
}

// Возвращает один интегральный поток в ориентации begin → end для обоих смежных узлов.
AERO_HOST_DEVICE inline bool integrated_edge_flux(const Geometry& g, Index edge, const double* field,
                                                  EulerFluxResult& out) {
    const Surface& surface = g.surfaces[edge];
    const double* begin = g.node_row(field, surface.begin);
    const double* end = g.node_row(field, surface.end);

    // Для характеристической реконструкции проецируем все узлы шаблона в один базис этого ребра.
    EulerRoeBasis basis{};
    double normal[3];
    Reader reader = {g, field, surface, basis, normal, g.options.variables};
    if (reader.variables == ResidualVariables::Characteristic &&
        euler_detail::roe_basis_with_normal(g.dim, begin, end, surface.area, g.options.gamma, &basis,
                                            normal) != EulerStatus::Success)
        return false;

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
    // У каждой стороны свои пределы по плотности и давлению относительно её исходного узла.
    if (!limit_trace(g.dim, g.options, begin, state.left, surface.area,
                     physical_flux ? physical_flux->left : nullptr) ||
        !limit_trace(g.dim, g.options, end, state.right, surface.area,
                     physical_flux ? physical_flux->right : nullptr))
        return false;

    if (solve_edge_flux(g, surface, state, physical_flux, out) != EulerStatus::Success)
        return false;
    // Решатель выдаёт поток на единицу площади. Умножаем компоненты на площадь один раз.
    // max_wave_speed остаётся скоростью: площадь для шага CFL учитывается отдельно.
    for (unsigned c = 0; c < g.components(); ++c) {
        out.flux[c] *= surface.measure;
        if (!euler_detail::finite(out.flux[c]))
            return false;
    }
    return true;
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
    AERO_HOST_DEVICE bool compute(Index f) const {
        // Сохраняем поток даже тогда, когда вызывающей стороне не нужны отдельные массивы скоростей.
        if (!integrated_surface_flux(geometry, f, state, initial, flux[f]))
            return false;
        if (f < geometry.edges) {
            if (edge_speeds)
                edge_speeds[f] = flux[f].max_wave_speed;
        } else if (boundary_speeds)
            boundary_speeds[f - geometry.edges] = flux[f].max_wave_speed;
        return true;
    }
};
void cpu_face_fluxes(const FaceBatch&, const MeshIndex* global_nodes = nullptr);
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
