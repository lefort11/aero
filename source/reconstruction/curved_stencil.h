#pragma once
#include "../parallel/row_table.h"
#include "reconstruction.h"
#include <array>
#include <functional>

namespace aero {
class NodeIncidence;
namespace detail {
// В призме связываем основание с противоположным основанием. У гексаэдра и
// четырёхугольника направление роста выбирает исходная грань стенки.
struct PrismLayers {
    std::vector<std::vector<MeshIndex>> opposite;
    std::vector<MeshIndex> wall_faces;
    PrismLayers(const Mesh&, const std::vector<std::string>& labels);
    using Frontier = std::vector<MeshIndex>;
    using Sync = std::function<bool(Frontier&)>;
    std::vector<int> local_depths(MeshView, const Sync& sync = {}) const;
};

struct CurvedPoints {
    RayPoint near[2], far[2];
};
class CurvedStencil {
    using List = std::vector<MeshIndex>;
    MeshView mesh_;
    double cosine_; // Косинус предельного отклонения от прямого продолжения ребра.
    RowTable<MeshIndex,2> next_; // Следующее ребро за каждым концом или edge_count.
    std::vector<int> layer_;         // Минимальное число слоёв от стенки; -1 вне призматической области.
    std::vector<Coordinate> normal_; // Для PL нормаль к слою, усреднённая на ребре.
    std::vector<List> layer_faces_;  // Грани своего слоя, содержащие узел.
    std::vector<List> layer_edges_;  // Тангенциальные рёбра (в том числе в 2D).
    void prepare_ijk(const NodeIncidence&);
    void prepare_pl(const Mesh&, const ReconstructionOptions&);
    template <class Next>
    bool follow_line(MeshIndex edge, bool fifth, bool limit_each_turn, Next next, CurvedPoints&) const;
    bool pl_points(MeshIndex, bool, CurvedPoints&) const;

  public:
    // При совместной подготовке IJK и прямых шаблонов используем общую таблицу инцидентных ячеек.
    CurvedStencil(const Mesh&, const ReconstructionOptions&, const NodeIncidence* cells = nullptr);
    // Возвращает обе ближние опоры и доступные дальние; false включает прямой шаблон.
    bool points(MeshIndex edge, bool fifth, CurvedPoints&) const;
};
} // namespace detail
} // namespace aero
