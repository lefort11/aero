#ifndef AERO_MESHDEC_H
#define AERO_MESHDEC_H
#include "storage.h"

namespace aero { namespace meshdec {
struct Options {
    std::string source,target,temp;
    Id memory=64*1024*1024, leaf_bytes=4*1024*1024, target_ranks=0;
    bool compress=false;
};
struct Header {
    Id dimension,nodes,faces,face_entries,cells,cell_entries,boundaries,patches;
    double scale,low[3],high[3];
};
struct Entity {
    Id morton,kind,id,bytes;
    bool operator<(const Entity& other) const {
        if (morton!=other.morton) return morton<other.morton;
        if (kind!=other.kind) return kind<other.kind;
        return id<other.id;
    }
};
struct Address { Id leaf,offset; };
struct Boundary { Id id,patch; };
struct Box {
    double low[3],high[3];
    static Box point(double x,double y,double z) { return {{x,y,z},{x,y,z}}; }
    void include(const Box& other) {
        for (unsigned k=0;k<3;++k) {
            low[k]=std::min(low[k],other.low[k]);
            high[k]=std::max(high[k],other.high[k]);
        }
    }
};
struct Leaf {
    Id first,records,raw_bytes,nodes,faces,cells;
};
struct Chunk { Id begin,end; };
inline Chunk divide(Id count,int rank,int size) {
    const Id n=count/static_cast<unsigned>(size), extra=count%static_cast<unsigned>(size);
    const Id r=static_cast<unsigned>(rank),start=r*n+std::min(r,extra);
    return {start,start+n+(r<extra?1u:0u)};
}
inline std::string part(const std::string& base,int rank) { return base+"-"+std::to_string(rank); }
void unpack(const Options&,const std::string& dir);
void index_topology(const std::string& dir,Id memory);
void order_entities(const std::string& dir,Id memory,int rank,int size);
void choose_splitters(const std::string& dir,Id memory,int size);
void sort_range(const std::string& dir,Id memory,int rank,int size);
void plan_leaves(const std::string& dir,const Options&,int size);
void pack_leaves(const std::string& dir,bool compress,int rank,int size);
Id write_catalog(const std::string& dir,const std::string& staged,bool compress,int size);
void write_part(const std::string& dir,const std::string& staged,int rank);
} }
#endif
