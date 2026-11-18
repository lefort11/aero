#ifndef AERO_MPI_BALANCE_H
#define AERO_MPI_BALANCE_H
#include "../mesh/leaf_balance.h"

namespace aero {
struct RunSettings;
struct GpuMemoryFit {
    MeshIndex free_bytes=0,budget=0,first=0,end=0,required=0;
    bool failed=false;
};
struct GpuPlacement {
    GpuMemoryFit local;
    std::vector<GpuRankCapacity> cards; // planning history on rank zero
};
LeafLayout plan_partition(const RunSettings& run,const Processes& world,GpuPlacement* gpu=nullptr);
}
#endif
