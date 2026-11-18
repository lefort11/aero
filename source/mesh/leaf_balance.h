#ifndef AERO_LEAF_BALANCE_H
#define AERO_LEAF_BALANCE_H
#include "partition.h"
#include <functional>

namespace aero {
struct LeafCost { MeshIndex work,bytes; };
struct CpuNodeShare {
    std::vector<int> ranks;
    long double mesh_bytes;
};
struct GpuRankCapacity {
    long double work_capacity; // archive work units, refined using the measured device layout
    // Actual device reservation failed for these whole-leaf intervals on this rank.
    std::vector<std::pair<MeshIndex,MeshIndex>> rejected;
};
// Cumulative costs for [0,end). Only directory entries at queried cuts are needed.
using LeafPrefix=std::function<LeafCost(MeshIndex)>;
LeafLayout balance_leaves(MeshIndex leaves,const LeafPrefix& prefix,const std::vector<CpuNodeShare>& nodes,
                          const std::vector<long double>& speeds);
LeafLayout balance_gpu_leaves(MeshIndex leaves,const LeafPrefix& prefix,const std::vector<CpuNodeShare>& nodes,
                             const std::vector<long double>& speeds,const std::vector<GpuRankCapacity>& cards);
}
#endif
