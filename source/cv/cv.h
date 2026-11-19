#ifndef AERO_CV_H
#define AERO_CV_H

#include "../mesh/mesh_index.h"
#ifndef __CUDACC_RTC__
#include <vector>
#endif

namespace aero {
class Mesh;
struct CvVector {
    double x, y, z;
};
struct CvBoundary {
    // Номера узла-владельца, исходной внешней грани и её граничной метки.
    MeshIndex node, face, patch;
    CvVector area; // Направлена наружу из контрольного объёма этого узла.
};

// Массивы принадлежат MedianDual и индексируются исходными номерами сетки. Для CUDA нужно перенести сами массивы
// в доступную устройству память и записать в представление полученные адреса.
struct CvView {
    unsigned dimension;
    MeshIndex node_count, edge_count, boundary_count;
    const double* volumes;      // Объём в единицах м^dimension.
    const CvVector* edge_areas; // Площадь в м^(dimension−1), направлена от edge_begin к edge_end.
    const CvBoundary* boundary; // Элемент на каждую пару из внешней грани и примыкающего узла.
};

#ifndef __CUDACC_RTC__
class MedianDual {
public:
    // Строит медианные объёмы на неподвижной сетке. Центр равен среднему координат вершин.
    // Каждая часть многоугольника или пирамиды должна иметь положительную ориентированную меру.
    explicit MedianDual(const Mesh& mesh);
    // Даёт доступ к геометрии без копирования массивов MedianDual.
    CvView view() const noexcept;

private:
    unsigned dimension_;
    // Длина, площадь или объём доли узла в зависимости от размерности сетки.
    std::vector<double> volumes_;
    // Интегральная нормаль внутренней площадки, ориентированной по ребру.
    std::vector<CvVector> edge_areas_;
    // Внешняя площадка для каждой пары «грань–узел».
    std::vector<CvBoundary> boundary_;
};
#endif
}
#endif
