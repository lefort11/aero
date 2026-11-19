#ifndef AERO_SOLVER_H
#define AERO_SOLVER_H
#include "../mesh/partition.h"
#include "../residual/residual.h"
#include "../parallel/processes.h"
#include "../io/settings.h"

namespace aero {
// Выбирает общий для всех процессов базис компонент импульса.
ResidualFrame momentum_frame(MeshView mesh, const Processes& world, const MeshPartition* partition = nullptr);
// Готовит сетку, подбирает разбиение для выбранного устройства и выполняет расчёт.
int calculate_case(RunSettings run, const ReconstructionOptions& reconstruction, const Processes& world,
                   bool archive);
}
#endif
