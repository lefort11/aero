#ifndef AERO_IO_SETTINGS_BOUNDARY_H
#define AERO_IO_SETTINGS_BOUNDARY_H

#include "../bc/bc.h"
#include <vector>

namespace aero {
struct MeshData;
struct CaseBoundaries {
    // Indexed by MeshData::boundary_patch. Both sides of a seam are Periodic.
    std::vector<BoundaryConfig> by_patch;
    std::vector<PeriodicPair> periodic_pairs;
};
CaseBoundaries read_case_boundaries(const std::string& path, const MeshData& mesh);
}
#endif
