#ifndef AERO_MPI_STENCIL_H
#define AERO_MPI_STENCIL_H
#include "mpi_periodic.h"
#include <set>

namespace aero {
struct StencilEdge {
    MeshIndex surface,begin,end;
    std::pair<MeshIndex,MeshIndex> original;
};
struct StencilRequests {
    std::vector<StencilEdge> edges;
    std::set<MeshIndex> inlet_nodes;
};
struct PartitionStencil {
    struct Rows {
        std::vector<ReconstructionIndex> offsets,nodes;
        std::vector<double> weights;
    };
    Rows linear,slopes;
    std::vector<ReconstructionIndex> row_owner;
    std::vector<double> epsilon_scale;
    std::vector<unsigned char> weno_order;
    std::vector<unsigned char> nodal_traces;
    std::vector<std::pair<MeshIndex,int>> extra_nodes;
    MeshIndex incomplete=0;
    ReconstructionStencilView view(MeshIndex state_nodes) const;
};
// Prepare both side rows on the process that evaluates the edge flux.
PartitionStencil prepare_partition_stencil(const MeshPartition&,const PeriodicPartition*,const std::vector<PeriodicPair>&,
    PartitionInput*,const StencilRequests&,MeshIndex surfaces,const ReconstructionOptions&,const Processes&);
}
#endif
