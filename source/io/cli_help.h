#pragma once
#include <cstring>

namespace aero {
// Распознаёт отдельный аргумент справки решателя и утилит без выделения строки.
inline bool help_requested(int argc, const char* const* argv) {
    return argc == 2 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0);
}
}
