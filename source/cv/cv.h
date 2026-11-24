#ifndef AERO_CV_H
#define AERO_CV_H

#include "../mesh/mesh_index.h"
#include <cmath>
#include "../mesh/coordinate.h"
#include <vector>

namespace aero {
class Mesh;
// Трёхмерный вектор геометрии контрольного объёма.
struct CvVector {
    // Декартовы компоненты геометрического вектора.
    double x, y, z;

    // Вычисляет скалярное произведение трёхмерных геометрических векторов.
    double dot(CvVector b) const { return x*b.x + y*b.y + z*b.z; }
    // Вычисляет векторное произведение.
    CvVector cross(CvVector b) const { return {y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x}; }
    // hypot сохраняет масштаб и для очень коротких, и для очень длинных векторов.
    double length() const { return std::hypot(std::hypot(x,y),z); }
};
// Внешняя площадка контрольного объёма с владельцем и граничной меткой.
struct CvBoundary {
    // Номера узла-владельца, исходной внешней грани и её граничной метки.
    MeshIndex node, face, patch;
    CvVector area; // Направлена наружу из контрольного объёма этого узла.
    Coordinate point; // Исходная координата узла до периодического отождествления.
};

// Массивы принадлежат MedianDual и индексируются исходными номерами сетки. Для CUDA нужно перенести сами массивы
// в доступную устройству память и записать в представление полученные адреса.
struct CvView {
    // Размерность контрольных объёмов.
    unsigned dimension;
    // Длины узлового, рёберного и граничного массивов.
    MeshIndex node_count, edge_count, boundary_count;
    const double* volumes;      // Контрольный объём в единицах длины сетки в степени dimension.
    const CvVector* edge_areas; // Вектор площадки в тех же единицах в степени dimension−1, от начала ребра к концу.
    const CvBoundary* boundary; // Элемент на каждую пару из внешней грани и примыкающего узла.
    const double* min_heights;  // Минимальная высота исходных ячеек, примыкающих к узлу, в единицах сетки.
};

// Владеет геометрией медианных контрольных объёмов исходной сетки.
class MedianDual {
public:
    // Строит медианные объёмы на неподвижной сетке. Центр равен среднему координат вершин.
    // Каждая часть многоугольника или пирамиды должна иметь положительную ориентированную меру.
    explicit MedianDual(const Mesh& mesh);
    // Даёт доступ к геометрии без копирования массивов MedianDual.
    CvView view() const noexcept;

private:
    // Пространственная размерность исходной сетки.
    unsigned dimension_;
    // Длина, площадь или объём доли узла в зависимости от размерности сетки.
    std::vector<double> volumes_;
    // Минимальная характерная высота для оценки шага в узле.
    std::vector<double> min_heights_;
    // Интегральная нормаль внутренней площадки, ориентированной по ребру.
    std::vector<CvVector> edge_areas_;
    // Внешняя площадка для каждой пары «грань–узел».
    std::vector<CvBoundary> boundary_;
};
}
#endif
