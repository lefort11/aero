#ifndef AERO_CONVERTER_H
#define AERO_CONVERTER_H
#include "../../source/mesh/mesh.h"
#include <istream>
namespace aero { namespace converter {
struct Options {
    // CGNS indices are one-based; zero means require exactly one base/zone.
    int base, zone;
    double length_scale_to_meters;
    std::string unmarked_name;
    Options() : base(0), zone(0), length_scale_to_meters(1), unmarked_name("unmarked") {}
};
Mesh read_gmsh(std::istream& input, const Options& options = Options());
Mesh read_gmsh(const std::string& path, const Options& options = Options());
Mesh read_cgns(const std::string& path, const Options& options = Options());
// another mesh directory or its mesh.txt; ordinary TEXT/BINARY 2D/3D meshes.
Mesh read_another(const std::string& path, const Options& options = Options());
bool cgns_available() noexcept;
} }
#endif
