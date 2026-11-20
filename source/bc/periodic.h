#ifndef AERO_PERIODIC_H
#define AERO_PERIODIC_H
#include "bc.h"
#include "../cv/cv.h"
#include "../reconstruction/reconstruction.h"
#include <array>

namespace aero {
// Каждая запись описывает расчётное ребро, ориентированное как исходный представитель.
// Параллельные рёбра и петли при периоде в одну ячейку учитываются отдельно.
struct PeriodicEdge {
    MeshIndex begin, end, original;
    CvVector area;
};
struct PeriodicView {
    MeshIndex node_count, edge_count, boundary_count, original_node_count;
    // Для каждого исходного узла хранится номер общей степени свободы после склейки.
    const MeshIndex* node_dof;
    const double* volumes;
    const PeriodicEdge* edges;
    const CvBoundary*
        boundary; // Только физические площадки. Поле node указывает номер степени свободы после склейки.
};
namespace periodic_detail {
double match_tolerance(const std::vector<PeriodicPair>& pairs, double requested = 0);
bool compatible_areas(const std::array<double, 3>& a, const std::array<double, 3>& b);
}
// Временная сетка сдвинутых ячеек нужна для подготовки шаблона. field связывает каждую вершину образа
// с расчётным узлом; это относится и к исходным вершинам в начале массива.
Mesh periodic_stencil_cover(const Mesh&, const std::vector<MeshIndex>& node_dof, MeshIndex count,
                            double tolerance, unsigned rings, std::vector<MeshIndex>& field);
class PeriodicDomain {
public:
    // Периодические связи строятся на CPU. Допуск задаётся в единицах входных координат.
    // При нуле используем 10⁻¹⁰ от длины наименьшего заданного периода.
    PeriodicDomain(const Mesh&, const std::vector<PeriodicPair>&, double tol = 0);
    PeriodicView view() const;
    // Исходная сетка должна существовать и сохранять геометрию всё время жизни периодической области.
    ReconstructionStencil prepare(const ReconstructionOptions&) const;
    PeriodicDomain(Mesh&&, const std::vector<PeriodicPair>&, double = 0) = delete;

private:
    const Mesh& mesh_;
    double tolerance_;
    std::vector<MeshIndex> node_dof_;
    // Объём степени свободы равен сумме объёмов всех её периодических копий.
    std::vector<double> volumes_;
    // Каждое расчётное ребро встречается один раз со своей суммарной ориентированной площадью.
    std::vector<PeriodicEdge> edges_;
    std::vector<CvBoundary> boundary_;
};
}
#endif
