#ifndef AERO_PERIODIC_H
#define AERO_PERIODIC_H
#include "bc.h"
#include "../cv/cv.h"
#include "../reconstruction/reconstruction.h"
#ifndef __CUDACC_RTC__
#include <array>
#endif

namespace aero {
// One entry per computational edge, oriented like its original representative.
// Parallel edges (even loops in a one-cell period) are distinct mesh entities.
struct PeriodicEdge { MeshIndex begin,end,original; CvVector area; };
struct PeriodicView {
    MeshIndex node_count,edge_count,boundary_count,original_node_count;
    const MeshIndex* node_dof;
    const double* volumes;
    const PeriodicEdge* edges;
    const CvBoundary* boundary; // physical pieces only; node is the compact DOF
};
#ifndef __CUDACC_RTC__
namespace periodic_detail {
double match_tolerance(const std::vector<PeriodicPair>& pairs,double requested=0);
bool compatible_areas(const std::array<double,3>& a,const std::array<double,3>& b);
}
// Temporary translated cells for stencil preparation. field maps every image
// vertex (including the original mesh prefix) to its computational node.
Mesh periodic_stencil_cover(const Mesh&,const std::vector<MeshIndex>& node_dof,MeshIndex count,
                            double tolerance,unsigned rings,std::vector<MeshIndex>& field);
class PeriodicDomain {
public:
    // Preparation on CPU. tol is an absolute distance in input coordinates;
    // zero selects 1e-10 of the smallest specified period length.
    PeriodicDomain(const Mesh&,const std::vector<PeriodicPair>&,double tol=0);
    PeriodicView view() const;
    // Requires the original immutable mesh (borrowed during domain lifetime).
    ReconstructionStencil prepare(const ReconstructionOptions&) const;
    PeriodicDomain(Mesh&&,const std::vector<PeriodicPair>&,double=0)=delete;
private:
    const Mesh& mesh_;
    double tolerance_;
    std::vector<MeshIndex> node_dof_;
    std::vector<double> volumes_;
    std::vector<PeriodicEdge> edges_;
    std::vector<CvBoundary> boundary_;
};
#endif
}
#endif
