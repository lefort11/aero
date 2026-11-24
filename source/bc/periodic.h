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
    // Концы после склейки и номер исходного ребра.
    MeshIndex begin, end, original;
    // Ориентированный вектор дуальной площадки.
    CvVector area;
};
// Массивы периодически склеенных степеней свободы и их геометрии.
struct PeriodicView {
    // Числа узлов, рёбер и границ после склейки и исходное число узлов.
    MeshIndex node_count, edge_count, boundary_count, original_node_count;
    // Для каждого исходного узла хранится номер общей степени свободы после склейки.
    const MeshIndex* node_dof;
    // Контрольные объёмы объединённых узлов.
    const double* volumes;
    // Массив рёбер после отождествления узлов.
    const PeriodicEdge* edges;
    const CvBoundary*
        boundary; // Только физические площадки. Поле node указывает номер степени свободы после склейки.
};
namespace periodic_detail {
// Выбирает геометрический допуск сопоставления периодических участков.
double match_tolerance(const std::vector<PeriodicPair>& pairs, double requested = 0);
// Проверяет согласованность площадей сопряжённых периодических граней.
bool compatible_areas(const Coordinate& a, const Coordinate& b);
}
// Временная сетка сдвинутых ячеек нужна для подготовки шаблона. field связывает каждую вершину образа
// с расчётным узлом; это относится и к исходным вершинам в начале массива.
Mesh periodic_stencil_cover(const Mesh&, const std::vector<MeshIndex>& node_dof, MeshIndex count,
                            double tolerance, unsigned rings, std::vector<MeshIndex>& field);
// Строит и хранит периодическое объединение узлов, рёбер и контрольных объёмов.
class PeriodicDomain {
public:
    // Периодические связи строятся на CPU. Допуск задаётся в единицах входных координат.
    // При нуле используем 10⁻¹⁰ от длины наименьшего заданного периода.
    PeriodicDomain(const Mesh&, const std::vector<PeriodicPair>&, double tol = 0);
    // Возвращает невладеющее представление склеенной периодической геометрии.
    PeriodicView view() const;
    // Исходная сетка должна существовать и сохранять геометрию всё время жизни периодической области.
    ReconstructionStencil prepare(const ReconstructionOptions&) const;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    PeriodicDomain(Mesh&&, const std::vector<PeriodicPair>&, double = 0) = delete;

private:
    // Исходная сетка, задающая геометрию до склейки.
    const Mesh& mesh_;
    // Допуск при сопоставлении координат.
    double tolerance_;
    // Отображение исходного узла в степень свободы после склейки.
    std::vector<MeshIndex> node_dof_;
    // Объём степени свободы равен сумме объёмов всех её периодических копий.
    std::vector<double> volumes_;
    // Каждое расчётное ребро встречается один раз со своей суммарной ориентированной площадью.
    std::vector<PeriodicEdge> edges_;
    // Оставшиеся физические граничные площадки.
    std::vector<CvBoundary> boundary_;
};
}
#endif
