#ifndef AERO_TECPLOT_H
#define AERO_TECPLOT_H
#include "../../source/mesh/mesh.h"
#include "../../source/io/fields.h"
#include <iosfwd>
#include <stdexcept>
#include <string>

namespace aero { namespace tecplot {
class ExportError : public std::runtime_error {
public:
    explicit ExportError(const std::string& message) : std::runtime_error(message) {}
};
struct Options {
    // 0: OpenMP runtime default (OMP_NUM_THREADS); serial builds use one thread.
    int threads;
    Options() : threads(0) {}
};
bool openmp_available() noexcept;
// Tecplot 360 binary PLT v112, little endian. Coordinates in meters.
// Checks SHA-256, sizes and finite values before writing. See README.txt.
void write(std::ostream& output,const Mesh& mesh,const FieldSnapshot& fields,const Options& options=Options());
// Existing output paths are never overwritten. Failed writes are removed.
void write(const std::string& path,const Mesh& mesh,const FieldSnapshot& fields,const Options& options=Options());
void write(const std::string& path,const Mesh& mesh,const Options& options=Options());
int run_cli(int argc,const char* const* argv,std::ostream& out,std::ostream& err);
} }
#endif
