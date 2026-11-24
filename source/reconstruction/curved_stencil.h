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
    // Сопоставление противоположных граней для продвижения по слоям.
    std::vector<std::vector<MeshIndex>> opposite;
    // Исходные грани стенки для построения слоёв.
    std::vector<MeshIndex> wall_faces;
    // Находит призматические слои, связанные с выбранными граничными участками.
    PrismLayers(const Mesh&, const std::vector<std::string>& labels);
    // Список сущностей на текущем фронте обхода.
    using Frontier = std::vector<MeshIndex>;
    // Согласует фронт на границах подобластей.
    using Sync = std::function<bool(Frontier&)>;
    // Определяет доступную глубину призматических слоёв для опор шаблона.
    std::vector<int> local_depths(MeshView, const Sync& sync = {}) const;
};

// Набор опорных точек криволинейной стороны шаблона.
struct CurvedPoints {
    // Ближние и дальние точки по обе стороны ребра.
    RayPoint near[2], far[2];
};
// Строит опоры реконструкции вдоль структурированных или призматических линий.
class CurvedStencil {
    // Последовательность индексов для поиска шаблона.
    using List = std::vector<MeshIndex>;
    // Сетка, на которой прослеживаются лучи.
    MeshView mesh_;
    double cosine_; // Косинус предельного отклонения от прямого продолжения ребра.
    RowTable<MeshIndex,2> next_; // Следующее ребро за каждым концом или edge_count.
    std::vector<int> layer_;         // Минимальное число слоёв от стенки; -1 вне призматической области.
    std::vector<Coordinate> normal_; // Для PL нормаль к слою, усреднённая на ребре.
    std::vector<List> layer_faces_;  // Грани своего слоя, содержащие узел.
    std::vector<List> layer_edges_;  // Тангенциальные рёбра (в том числе в 2D).
    // Строит направления и связи структурированных линий i, j, k.
    void prepare_ijk(const NodeIncidence&);
    // Подготавливает линии реконструкции по призматическим слоям.
    void prepare_pl(const Mesh&, const ReconstructionOptions&);
    // Если продолжение задано связностью, обе опорные точки — узлы сетки.
    // В 2D пристенный слой сам является ломаной. Для него используем тот же
    // обход, что и для IJK: пересечения с плоскостью нужны только для PL в 3D.
    template <class Next>
    bool follow_line(MeshIndex edge, bool fifth, bool limit_each_turn, Next next, CurvedPoints&) const;
    // Выбирает опорные точки вдоль призматической линии.
    bool pl_points(MeshIndex, bool, CurvedPoints&) const;

  public:
    // При совместной подготовке IJK и прямых шаблонов используем общую таблицу инцидентных ячеек.
    CurvedStencil(const Mesh&, const ReconstructionOptions&, const NodeIncidence* cells = nullptr);
    // Возвращает обе ближние опоры и доступные дальние; false включает прямой шаблон.
    bool points(MeshIndex edge, bool fifth, CurvedPoints&) const;
};
} // namespace detail
} // namespace aero
