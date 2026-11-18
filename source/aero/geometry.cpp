#include "geometry.h"
#include "../cv/cv.h"
#include <fstream>
#include <iomanip>
#include <limits>

namespace aero {
void write_geometry(const std::string& output,const Mesh& mesh,const CvView& cv,
                    const MeshPartition* part,int rank,int ranks) {
    const auto m=mesh.view();
    const auto owned=[&](MeshIndex n) { return !part || part->node_owners[n]==rank; };
    const auto node=[&](MeshIndex n) { return part ? part->nodes[n] : n; };
    const std::string path=output+"/geometry-"+std::to_string(rank)+"-of-"+std::to_string(ranks)+".tsv";
    std::ofstream out(path.c_str());
    if (!out) throw MeshError("Cannot create "+path);
    out.exceptions(std::ios::failbit|std::ios::badbit);
    out.imbue(std::locale::classic());
    out<<std::setprecision(std::numeric_limits<double>::max_digits10);
    out<<"# Original mesh IDs, median dual in SI units\n";
    for (MeshIndex i=0;i<m.node_count;++i) if (owned(i))
        out<<"V\t"<<node(i)<<'\t'<<cv.volumes[i]<<'\n';
    const auto vector=[&](CvVector v) { out<<'\t'<<v.x<<'\t'<<v.y<<'\t'<<v.z<<'\n'; };
    for (MeshIndex e=0;e<m.edge_count;++e) if (owned(m.edge_begin[e])) {
        out<<"E\t"<<node(m.edge_begin[e])<<'\t'<<node(m.edge_end[e]);
        vector(cv.edge_areas[e]);
    }
    for (MeshIndex k=0;k<cv.boundary_count;++k) {
        const auto& b=cv.boundary[k];
        if (owned(b.node) && (!part || b.patch<part->physical_patches)) {
            out<<"B\t"<<(part?part->faces[b.face]:b.face)<<'\t'<<node(b.node)<<'\t'<<b.patch;
            vector(b.area);
        }
    }
    out.close();
}
}
