#ifndef AERO_CONVERTER_CLI_H
#define AERO_CONVERTER_CLI_H
#include <iosfwd>
namespace aero { namespace converter {
// Returns 0 on success/help, 1 on missing arguments or conversion failure.
int run_cli(int argc, const char* const* argv, std::ostream& out, std::ostream& err);
} }
#endif
