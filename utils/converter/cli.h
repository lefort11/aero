#ifndef AERO_CONVERTER_CLI_H
#define AERO_CONVERTER_CLI_H
#include <iosfwd>
namespace aero {
namespace converter {
// Возвращает 0 при успехе или показе справки, 1 при ошибке аргументов или преобразования.
int run_cli(int argc, const char* const* argv, std::ostream& out, std::ostream& err);
}
}
#endif
