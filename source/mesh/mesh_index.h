#ifndef AERO_MESH_INDEX_H
#define AERO_MESH_INDEX_H
#ifndef __CUDACC_RTC__
#include <cstdint>
#endif
namespace aero {
// На CPU и при компиляции ядра NVRTC номера элементов должны занимать одинаковые восемь байтов.
#ifdef __CUDACC_RTC__
using MeshIndex = unsigned long long;
static_assert(sizeof(MeshIndex) == 8, "Mesh indices require 64 bits");
#else
using MeshIndex = std::uint64_t;
#endif
}
#endif
