#ifndef AERO_MPI_PERIODIC_H
#define AERO_MPI_PERIODIC_H
#include "../bc/periodic.h"
#include "../mesh/partition.h"
#include "../parallel/processes.h"

namespace aero {
// IDs are the smallest original vertex in each equivalence class. Edges
// remain distinct even when two different edges have the same canonical ends.
struct PeriodicPartition {
    std::vector<MeshIndex> ids,node_dof;
    std::vector<int> owners;
    std::vector<double> volumes;
    struct Edge { MeshIndex begin,end; CvVector area; std::pair<MeshIndex,MeshIndex> original; };
    std::vector<Edge> edges;
    std::vector<CvBoundary> boundary;
};
PeriodicPartition partition_periodic(const MeshPartition&,CvView,const std::vector<PeriodicPair>&,const Processes&);
}
#endif
