#ifndef AERO_GEOMETRY_H
#define AERO_GEOMETRY_H
#include "../mesh/partition.h"
namespace aero {
struct CvView;
void write_geometry(const std::string& output,const Mesh& mesh,const CvView& cv,
                    const MeshPartition* partition,int rank,int ranks);
}
#endif
