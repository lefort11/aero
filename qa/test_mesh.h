#ifndef AERO_QA_TEST_MESH_H
#define AERO_QA_TEST_MESH_H
#include "../source/mesh/mesh.h"
#include <stdexcept>

namespace aero_test {
// Сетки входят в репозиторий вместе с задачами QA. На отрезке длины 1
// метки xmin и xmax позволяют проверить периодическое замыкание.
inline aero::Mesh test_mesh(unsigned dimension) {
    switch (dimension) {
    case 1: return aero::Mesh::read("qa/cases/pulse_1d/grid.mesh");
    case 2: return aero::Mesh::read("qa/cases/momentum_2d/grid.mesh");
    case 3: return aero::Mesh::read("qa/cases/wave_3d/grid.mesh");
    default: throw std::invalid_argument("Unsupported test mesh dimension");
    }
}
}
#endif
