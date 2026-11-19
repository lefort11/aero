#ifndef AERO_PARTITION_STATE_H
#define AERO_PARTITION_STATE_H
#include "mpi_periodic.h"
#include "../io/fields.h"

namespace aero {
struct RunSettings;

// Поля газа на узлах подобласти, включая узлы, полученные от соседних процессов.
struct PartitionState {
    std::vector<double> initial; // Исходные значения, в том числе опорное состояние для границ.
    std::vector<double> current; // Текущее решение на CPU, а при CUDA — его копия для ввода и вывода.

    // Готовит память для восстановления или читает начальное поле в нумерации решателя.
    void prepare(const RunSettings& run, const MeshPartition& part, const PeriodicPartition* seam,
                 const FieldNodeSelection& selection, MeshIndex state_nodes, const Processes& world);

    // После обмена с соседями проверяет периодические копии и устанавливает начальное решение.
    void start_from_initial(const MeshPartition& part, const PeriodicPartition* seam);
};
}
#endif
