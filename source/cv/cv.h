#ifndef AERO_CV_H
#define AERO_CV_H

#include "../mesh/mesh_index.h"
#ifndef __CUDACC_RTC__
#include <vector>
#endif

namespace aero {
class Mesh;
struct CvVector { double x, y, z; };
struct CvBoundary {
    MeshIndex node, face, patch;
    CvVector area; // outward from this node's control volume
};

// Borrowed arrays indexed by the ORIGINAL mesh IDs. For CUDA, copy each
// array to device-accessible storage and bind its address in this descriptor.
struct CvView {
    unsigned dimension;
    MeshIndex node_count, edge_count, boundary_count;
    const double* volumes;       // m^dimension
    const CvVector* edge_areas;  // m^(dimension-1), edge_begin -> edge_end
    const CvBoundary* boundary; // one piece per (exterior face, incident node)
};

#ifndef __CUDACC_RTC__
class MedianDual {
public:
    // Static Euclidean mesh. Centers are arithmetic means of vertices.
    // Polygon fans and face pyramids must have positive oriented measures.
    explicit MedianDual(const Mesh& mesh);
    CvView view() const noexcept;
private:
    unsigned dimension_;
    std::vector<double> volumes_;
    std::vector<CvVector> edge_areas_;
    std::vector<CvBoundary> boundary_;
};
#endif
}
#endif
