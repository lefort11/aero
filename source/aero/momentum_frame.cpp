#include "solver.h"
#include <cmath>
#include <limits>

namespace aero {
namespace {
CvVector cross(CvVector a,CvVector b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
double length(CvVector a) { return std::hypot(std::hypot(a.x,a.y),a.z); }
CvVector unit(CvVector a) {
    const double n=length(a);
    if (!(n>0 && std::isfinite(n))) throw MeshError("Cannot determine mesh momentum frame");
    return {a.x/n,a.y/n,a.z/n};
}
CvVector orient(CvVector v) {
    double major=std::abs(v.x)>=std::abs(v.y)?v.x:v.y;
    if (std::abs(v.z)>std::abs(major)) major=v.z;
    return major<0 ? CvVector{-v.x,-v.y,-v.z} : v;
}
const MeshIndex absent=std::numeric_limits<MeshIndex>::max();
struct Candidate {
    MeshIndex begin=absent,end=absent;
    double score=-1;
    CvVector vector{};
};
// Highest score, then first edge in native lexicographic order.
CvVector select(const Candidate& c,const Processes& world) {
    if (world.size==1) return c.vector;
    const double score=-world.minimum(-c.score);
    const auto begin=world.first_id(c.score==score?c.begin:absent);
    const auto end=world.first_id(c.score==score && c.begin==begin?c.end:absent);
    if (begin==absent) throw MeshError("Cannot determine mesh momentum frame");
    const auto root=world.first_id(c.begin==begin && c.end==end?static_cast<MeshIndex>(world.rank):absent);
    double v[]={c.vector.x,c.vector.y,c.vector.z};
    world.broadcast(v,3,static_cast<int>(root));
    return {v[0],v[1],v[2]};
}
}
ResidualFrame momentum_frame(MeshView mesh,const Processes& world,const MeshPartition* part) {
    ResidualFrame frame;
    if (mesh.dimension==3) return frame;
    CvVector tangent{};
    for (unsigned pass=0;pass<mesh.dimension;++pass) {
        Candidate best;
        world.local([&] {
            for (MeshIndex e=0;e<mesh.edge_count;++e) {
                const auto a=mesh.edge_begin[e],b=mesh.edge_end[e];
                if (part && part->node_owners[a]!=world.rank) continue;
                CvVector v=unit({mesh.x[b]-mesh.x[a],mesh.y[b]-mesh.y[a],mesh.z[b]-mesh.z[a]});
                if (pass) v=cross(tangent,v);
                const double score=pass?std::min(0.5,length(v)):1;
                if (score>best.score) {
                    best.begin=part?part->nodes[a]:a; best.end=part?part->nodes[b]:b;
                    best.score=score; best.vector=v;
                }
                if (!pass || score==0.5) break;
            }
        });
        const auto chosen=select(best,world);
        if (!pass) {
            tangent=chosen;
            if (mesh.dimension==1) frame.axis[0]=orient(tangent);
        } else {
            const auto normal=orient(unit(chosen));
            const CvVector reference=std::abs(normal.x)>0.99 ? CvVector{0,1,0} : CvVector{1,0,0};
            frame.axis[1]=unit(cross(normal,reference));
            frame.axis[0]=unit(cross(frame.axis[1],normal));
        }
    }
    return frame;
}
}
