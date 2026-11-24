#ifndef AERO_MPI_PERIODIC_H
#define AERO_MPI_PERIODIC_H
#include "../bc/periodic.h"
#include "../mesh/partition.h"
#include "../parallel/processes.h"

namespace aero {
// Номер класса равен наименьшему исходному номеру его вершины. Разные рёбра могут иметь одну пару приведённых концов.
struct PeriodicPartition {
    // ids даёт исходный номер представителя, node_dof переводит исходные локальные вершины в расчётные.
    std::vector<MeshIndex> ids, node_dof;
    // Процесс, владеющий значением каждой расчётной степени свободы.
    std::vector<int> owners;
    // Сумма контрольных объёмов всех периодических копий соответствующего узла.
    std::vector<double> volumes;
    // Для CFL берём наименьшую высоту со всех сторон периодического шва.
    std::vector<double> min_heights;
    // Объединённое периодическое ребро с площадью и исходными номерами концов.
    struct Edge {
        // Локальные номера концов ориентированного ребра.
        MeshIndex begin, end;
        // Вектор площади от начала ребра к концу.
        CvVector area;
        // Идентификаторы концов представителя исходного ребра, необходимые при построении шаблона.
        std::pair<MeshIndex, MeshIndex> original;
    };
    // Внутренние связи после периодической склейки.
    std::vector<Edge> edges;
    // Физические граничные площадки после склейки.
    std::vector<CvBoundary> boundary;
};
// Строит компактные данные для тех же узловых сумм, что и в обычной периодической области.
PeriodicPartition partition_periodic(const MeshPartition&, CvView, const std::vector<PeriodicPair>&,
                                     const Processes&);
}
#endif
