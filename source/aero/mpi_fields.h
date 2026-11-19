#ifndef AERO_MPI_FIELDS_H
#define AERO_MPI_FIELDS_H
#include "indexed_io.h"
#include "../io/output_settings.h"

namespace aero {
// Владельцы узлов совместно записывают один файл зоны. Только завершённый снимок получает окончательное имя.
void write_global_fields(const Processes& world, const std::string& directory, const OutputZone& zone,
                         const std::vector<unsigned>& columns, const FieldNodeSelection& mesh,
                         const std::vector<OwnedNode>& nodes, double gamma, std::uint64_t step, double time,
                         const std::vector<double>& state);
}
#endif
