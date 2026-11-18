#ifndef AERO_INITIAL_H
#define AERO_INITIAL_H
#include <iosfwd>
#include <string>
#include <vector>

namespace aero {
class Mesh;
struct PeriodicView;

// Construct [computational node][rho, momentum..., total energy density].
// settings is one initial-condition block. Relative field paths use directory.
// Supply a host periodic view to compact and check identified original nodes.
std::vector<double> read_initial(const Mesh& mesh,double gamma,std::istream& settings,
    const std::string& directory=".",const PeriodicView* periodic=nullptr);

// Resolve field paths relative to the settings file containing the block.
std::vector<double> read_initial_file(const Mesh& mesh,double gamma,const std::string& settings_path,
    const PeriodicView* periodic=nullptr);
}
#endif
