#pragma once
#include <iosfwd>

namespace aero {
namespace visfield {
// Разбирает пути и параметры visfield, выбирает формат по имени результата.
int run_cli(int argc, const char* const* argv, std::ostream& out, std::ostream& err);
}
}
