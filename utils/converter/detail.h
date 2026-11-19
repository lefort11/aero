#ifndef AERO_CONVERTER_DETAIL_H
#define AERO_CONVERTER_DETAIL_H
#include "converter.h"
#include <map>
#include <vector>
namespace aero {
namespace converter {
namespace detail {
typedef std::vector<MeshIndex> Nodes;
typedef std::vector<Nodes> Faces;
// Промежуточная запись элемента из файла, до перевода ссылок в нумерацию выходной сетки.
struct Element {
    // Код формы определяет размерность и порядок обхода вершин.
    int kind;
    // Последовательность вершин элемента в порядке входного формата.
    Nodes nodes;
    // Имя граничной группы, если элемент описывает внешнюю грань.
    std::string patch;
};
// Коды линейных элементов Gmsh: 15 — точка, 1 — отрезок, 2 — треугольник, 3 — четырёхугольник,
// 4 — тетраэдр, 5 — гексаэдр, 6 — призма, 7 — пирамида. Для многоугольника используем код 100.
unsigned dimension(int kind);
std::size_t node_count(int kind);
// Разворачивает вершины одной ячейки в ориентированные контуры её граней.
Faces cell_faces(int kind, const Nodes& nodes);
// Возвращает отсортированные номера узлов. По ним можно найти ту же грань при другом обходе.
Nodes key(Nodes nodes);
class Builder {
public:
    MeshData data;
    explicit Builder(unsigned dim, const Options& options);
    // Добавляет ячейку и переиспользует уже созданные грани соседей.
    void cell(const Faces& faces);
    void mark(const Nodes& nodes, const std::string& name, bool require_boundary = true);
    // Находит внешние грани, все вершины которых входят в заданную группу.
    void mark_vertices(const Nodes& nodes, const std::string& name);
    // Назначает оставшиеся граничные метки и строит проверенную сетку.
    Mesh finish();

private:
    Options options_;
    // Сопоставляет набор узлов с единственным номером грани.
    std::map<Nodes, MeshIndex> face_ids_;
    // Ориентированный обход каждой грани при её первом появлении.
    std::vector<Nodes> faces_;
    // Число примыкающих ячеек. Единица означает внешнюю грань.
    std::vector<unsigned> uses_;
    // Имена внешних граней до уплотнения их в таблицу меток сетки.
    std::map<MeshIndex, std::string> labels_;
};
}
}
}
#endif
