#include <mutex>
#include "stencil_builder.h"
#include "curved_stencil.h"
#include <cmath>
#include <algorithm>

namespace aero {
namespace detail {
namespace {
// Все производные умножены на длину ребра и ориентированы от начала ребра к концу.
struct SideSupport {
    RayPoint near_point;     // Пересечение продолжения ребра с первым слоем.
    SparseRow near_slope;    // Производная между концом ребра и ближней точкой.
    SparseRow far_slope;     // Дальняя производная; пустая строка означает, что её построить не удалось.
    double far_distance = 0; // Расстояние для диагностики, в единицах координат сетки.

    // Дальняя опора строится только после обеих ближних. Без своей ближней
    // точки доступна лишь центральная разность между концами ребра.
    unsigned char available_order() const {
        if (!far_slope.empty())
            return 5;
        return near_slope.empty() ? 0 : 3;
    }
};

// Весовая строка удвоенного приращения. Для WENO это оптимальная линейная комбинация.
SparseRow linear_increment(unsigned order, const SparseRow& central, const SideSupport& own,
                           const SideSupport& opposite) {
    SparseRow row;
    if (order == 5) {
        add_scaled_row(row, own.far_slope, -1.0 / 15);
        add_scaled_row(row, own.near_slope, 11.0 / 30);
        add_scaled_row(row, central, 4.0 / 5);
        add_scaled_row(row, opposite.near_slope, -1.0 / 10);
    } else if (order == 3) {
        add_scaled_row(row, central, 2.0 / 3);
        add_scaled_row(row, own.near_slope, 1.0 / 3);
    } else {
        // Без ближней опоры состояние на стороне ребра равно среднему двух концов.
        return central;
    }
    return row;
}

// Сохраняет геометрию четырёх производных в порядке, который ожидает weno_increment2.
// Нелинейные веса зависят от полей и вычисляются при применении шаблона.
void append_weno_slopes(ReconstructionStencilData& rows, unsigned char order, double epsilon_scale,
                        const SparseRow& central, const SideSupport& own, const SideSupport& opposite) {
    rows.weno_order.push_back(order);
    rows.weno_epsilon_scale.push_back(epsilon_scale);
    const SparseRow* slopes[] = {&central, &own.near_slope, &opposite.near_slope, &own.far_slope};
    const unsigned used = order ? weno_slope_count(order) : 0;
    for (unsigned k = 0; k < 4; ++k) {
        if (k < used) {
            for (const auto& term : *slopes[k]) {
                if (!std::isfinite(term.second)) {
                    throw ReconstructionError("Non-finite WENO derivative weight");
                }
                if (term.second == 0) {
                    continue;
                }
                rows.slope_nodes.push_back(term.first);
                rows.slope_weights.push_back(term.second);
            }
        }
        // Даже пустая производная занимает строку: номер следующей стороны остаётся предсказуемым.
        rows.slope_offsets.push_back(rows.slope_nodes.size());
    }
}

// Геометрия определяет достижимый порядок отдельно слева и справа.
// Укороченные MP-стороны используют ту же формулу WENO3, что и укороченная WENO5.
ReconstructionSideScheme effective_side_scheme(unsigned order, const ReconstructionSchemeInfo& scheme) {
    if (order == 0)
        return ReconstructionSideScheme::EdgeAverage;
    if (order == 3)
        return uses_slope_rows(scheme.family) ? ReconstructionSideScheme::Weno3 : ReconstructionSideScheme::Ebr3;
    if (scheme.scheme == ReconstructionScheme::Mp5)
        return ReconstructionSideScheme::Mp5;
    if (scheme.scheme == ReconstructionScheme::MPWENO5)
        return ReconstructionSideScheme::MPWENO5;
    return uses_slope_rows(scheme.family) ? ReconstructionSideScheme::Weno5 : ReconstructionSideScheme::Ebr5;
}

// Записывает линейную строку и, для WENO, строки наклонов на каждой стороне ребра.
void append_stencil_sides(ReconstructionStencilData& rows, const ReconstructionGeometry& geometry,
                          const ReconstructionOptions& options, const ReconstructionSchemeInfo& scheme,
                          MeshIndex edge, double length, const SparseRow& central,
                          const SideSupport (&support)[2], bool curved) {
    const bool nonlinear = uses_slope_rows(scheme.family);
    const unsigned requested_order = scheme.layers == 2 ? 5 : 3;
    for (unsigned side = 0; side < 2; ++side) {
        const auto& own = support[side];
        const auto& opposite = support[1 - side];
        // Понижаем только ту сторону, для которой не хватило опорных точек.
        const unsigned char order = own.available_order();
        const auto actual = effective_side_scheme(order, scheme);
        const bool complete = order == requested_order;
        if (!complete && options.require_complete_stencil) {
            const std::string family = scheme.name;
            throw ReconstructionError("Incomplete requested " + family + " stencil at edge " +
                                      std::to_string(edge) + " side " + std::to_string(side));
        }
        geometry.append_row(rows, linear_increment(order, central, own, opposite), edge, side, curved);
        rows.complete.push_back(complete ? 1 : 0);
        rows.side_scheme.push_back(actual);
        rows.outer_distance.push_back(order ? own.near_point.distance : 0);
        rows.far_distance.push_back(own.far_distance);
        if (nonlinear) {
            const double scale = length * std::sqrt(options.weno_epsilon);
            if (!std::isfinite(scale)) {
                throw ReconstructionError("WENO epsilon/edge scale overflow");
            }
            append_weno_slopes(rows, order, scale, central, own, opposite);
        }
    }
}

using NodeList = std::vector<MeshIndex>;
using FaceLists = std::vector<std::vector<MeshIndex>>;

// Один номер может встретиться через несколько узлов. Оставляем его один раз.
void sort_unique(NodeList& ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

// В опорной грани все вершины должны принадлежать одному слою.
NodeList faces_inside_layer(MeshView mesh, const NodeList& layer, const FaceLists& node_faces) {
    NodeList candidates;
    for (MeshIndex node : layer) {
        candidates.insert(candidates.end(), node_faces[node].begin(), node_faces[node].end());
    }
    sort_unique(candidates);
    NodeList faces;
    for (MeshIndex face : candidates) {
        const auto vertices = mesh.face_nodes(face);
        if (std::all_of(vertices.begin(), vertices.end(),
                        [&](MeshIndex node) { return std::binary_search(layer.begin(), layer.end(), node); })) {
            faces.push_back(face);
        }
    }
    return faces;
}

// Первый слой — все вершины общих ячеек. Второй — их соседи за вычетом центра и первого слоя.
class StraightLayers {
    MeshView mesh_;
    bool fifth_;
    FaceLists neighbours_, node_faces_, near_, far_;
    std::vector<std::once_flag> pending_;

    // Формируем обе оболочки узла за один вызов. Несколько рёбер могут пользоваться ими одновременно.
    void prepare(MeshIndex node) {
        const auto& first = neighbours_[node];
        near_[node] = faces_inside_layer(mesh_, first, node_faces_);
        if (!fifth_)
            return;
        NodeList second;
        for (MeshIndex adjacent : first) {
            for (MeshIndex candidate : neighbours_[adjacent]) {
                if (candidate != node && !std::binary_search(first.begin(), first.end(), candidate))
                    second.push_back(candidate);
            }
        }
        sort_unique(second);
        far_[node] = faces_inside_layer(mesh_, second, node_faces_);
    }

  public:
    StraightLayers(MeshView mesh, bool fifth, bool on_demand)
        : mesh_(mesh), fifth_(fifth), neighbours_(mesh.node_count), node_faces_(mesh.node_count),
          near_(mesh.node_count), far_(fifth ? mesh.node_count : 0),
          pending_(on_demand ? mesh.node_count : 0) {
        for (MeshIndex cell = 0; cell < mesh.cell_count; ++cell) {
            const auto nodes = mesh.cell_nodes(cell);
            for (MeshIndex node : nodes)
                for (MeshIndex other : nodes)
                    if (other != node)
                        neighbours_[node].push_back(other);
        }
        for (MeshIndex face = 0; face < mesh.face_count; ++face)
            for (MeshIndex node : mesh.face_nodes(face))
                node_faces_[node].push_back(face);
        parallel_for(mesh.node_count, [&](MeshIndex n) { sort_unique(neighbours_[n]); });
        // В обычном режиме все узлы нужны заранее. После заполнения оболочек
        // освобождаем связность, чтобы не хранить её вместе с готовыми строками.
        if (!on_demand) {
            parallel_for(mesh.node_count, [&](MeshIndex n) { prepare(n); });
            FaceLists().swap(neighbours_);
            FaceLists().swap(node_faces_);
        }
    }

    const NodeList& faces(MeshIndex node, bool far) {
        // На структуре рёбра сразу получают опорные точки. Оболочка потребуется
        // только при возврате к прямому шаблону, один раз на все рёбра данного узла.
        if (!pending_.empty())
            std::call_once(pending_[node], [&] { prepare(node); });
        return far ? far_[node] : near_[node];
    }
};

// Обе схемы пятого порядка берут разность интерполированных значений в двух пересечениях одного луча.
void build_far_layer_slope(const RayPoint& far, unsigned side, double length, double tolerance,
                          SideSupport& support) {
    const double spacing = far.distance - support.near_point.distance;
    if (far.weights.empty() || !(spacing > tolerance * length)) {
        return;
    }
    const double factor = (side ? 1.0 : -1.0) * (length / spacing);
    add_scaled_row(support.far_slope, far.weights, factor);
    add_scaled_row(support.far_slope, support.near_point.weights, -factor);
    support.far_distance = far.distance;
}
}

void build_layer_stencil(const Mesh& mesh, const ReconstructionOptions& options,
                        const ReconstructionSchemeInfo& scheme, ReconstructionStencilData& data) {
    const ReconstructionGeometry geometry(mesh, options.relative_tolerance);
    const auto view = mesh.view();
    const CurvedStencil curves(mesh, options);
    const bool fifth = scheme.layers == 2;
    StraightLayers layers(view, fifth, options.geometry != StencilGeometry::Straight);
    build_edge_rows(data, [&](MeshIndex edge, ReconstructionStencilData& rows) {
        const MeshIndex endpoints[] = {view.edge_begin[edge], view.edge_end[edge]};
        const double length = geometry.edge_length(edge);
        const SparseRow central = {{endpoints[0], -1}, {endpoints[1], 1}}; // Q_j − Q_i.
        SideSupport support[2];

        CurvedPoints points;
        const bool use_curve = curves.points(edge, fifth, points);
        // Продлеваем ребро за каждый конец. Первый слой даёт ближнюю опору и наклон от узла к ней.
        for (unsigned side = 0; side < 2; ++side) {
            const MeshIndex node = endpoints[side];
            auto& s = support[side];
            s.near_point =
                use_curve ? points.near[side] : geometry.layer_hit(edge, side, layers.faces(node, false));
            if (!s.near_point.weights.empty())
                s.near_slope = ray_derivative(node, s.near_point, length, side);
        }
        // Обе формулы пятого порядка требуют ближних опор с двух сторон и дальней опоры со своей стороны.
        if (scheme.layers == 2 && !support[0].near_slope.empty() && !support[1].near_slope.empty()) {
            for (unsigned side = 0; side < 2; ++side) {
                const auto far = use_curve
                                     ? points.far[side]
                                     : geometry.layer_hit(edge, side, layers.faces(endpoints[side], true));
                build_far_layer_slope(far, side, length, options.relative_tolerance, support[side]);
            }
        }
        append_stencil_sides(rows, geometry, options, scheme, edge, length, central, support, use_curve);
    });
}
}
}
