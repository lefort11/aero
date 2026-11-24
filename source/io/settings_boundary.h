#ifndef AERO_IO_SETTINGS_BOUNDARY_H
#define AERO_IO_SETTINGS_BOUNDARY_H

#include "../bc/bc.h"
#include <vector>

namespace aero {
struct MeshData;
// Правила граничных участков постановки и список периодических пар.
struct CaseBoundaries {
    // Номер элемента совпадает с MeshData::boundary_patch. Оба участка периодического шва имеют тип Periodic.
    BoundaryPatches patches; // Метки и отдельный массив условий free.
    std::vector<PeriodicPair> periodic_pairs;
};
// Связывает граничные секции постановки с участками загруженной сетки.
CaseBoundaries read_case_boundaries(const std::string& path, const MeshData& mesh, bool linear = false,
                                   std::vector<std::string>* warnings = nullptr, BoundaryTemperature temperature = {});
}
#endif
