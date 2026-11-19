#ifndef AERO_CASE_GEOMETRY_H
#define AERO_CASE_GEOMETRY_H
#include "mpi_periodic.h"
#include "../io/settings_boundary.h"

namespace aero {
struct RunSettings;

// Геометрия одной подобласти. Решатель использует её объёмы до конца расчёта.
struct CaseGeometry {
    CaseBoundaries boundaries;        // Условия на физических границах сетки.
    std::unique_ptr<MedianDual> dual; // Контрольные объёмы до объединения периодических узлов.
    PeriodicPartition seam;           // Нумерация и объёмы после объединения узлов.

    // У обычной сетки все узлы сохраняют исходную локальную нумерацию.
    bool periodic() const { return !boundaries.periodic_pairs.empty(); }

    // Передаёт операторам нумерацию периодической сетки, если она построена.
    const PeriodicPartition* periodic_partition() const { return periodic() ? &seam : nullptr; }

    // В периодическом расчёте оператор уже забрал геометрию поверхностей.
    // Нумерация и объёмы остаются доступны для шага по времени и записи полей.
    void release_stencil_geometry();
};

// Возвращает false, если пользователю нужно заполнить созданный файл граничных условий.
// В однопроцессном режиме также выбирает устройство до построения контрольных объёмов.
bool prepare_case_geometry(RunSettings& run, const MeshPartition& partition, const Processes& world,
                           bool first_attempt, CaseGeometry& geometry);
}
#endif
