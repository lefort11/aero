#pragma once
#include <ostream>
#include <vector>

namespace aero {
namespace implicit {
// Порядок записей совпадает с порядком поправок Ньютона, включая последнюю неудачную попытку.
inline void write_linear_iterations(std::ostream& out, const std::vector<unsigned>& counts, unsigned limit) {
    if (counts.empty()) {
        out << "no Newton corrections";
        return;
    }
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (i) out << ", ";
        out << '#' << i + 1 << " = " << counts[i] << '/' << limit;
    }
}
}
}
