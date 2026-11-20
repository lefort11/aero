#ifndef AERO_MESH_INDEX_H
#define AERO_MESH_INDEX_H
#include <cstdint>
namespace aero {
// На CPU и в вычислениях Kokkos на GPU номера элементов должны занимать одинаковые восемь байтов.
using MeshIndex = std::uint64_t;
}
#endif
