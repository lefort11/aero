#ifndef AERO_CFL_DETAIL_H
#define AERO_CFL_DETAIL_H
#include "timestep.h"

namespace aero { namespace cfl_detail {
struct Entry { size_t face; double area; };
struct Grid {
    size_t nodes;
    const size_t* offsets;
    const Entry* entries;
    const double* volumes;
};
#ifdef __CUDACC__
#define AERO_CFL_HD __host__ __device__
#else
#define AERO_CFL_HD
#endif
AERO_CFL_HD inline CflStep unconstrained() {
    return {max_finite_double,static_cast<size_t>(-1)};
}
AERO_CFL_HD inline CflStep minimum(CflStep a,CflStep b) {
    return (b.dt<a.dt || (b.dt==a.dt && b.node<a.node)) ? b : a;
}
AERO_CFL_HD inline CflStep from_sum(double sum,double volume,double cfl,size_t node) {
    if (sum==0) return unconstrained();
    if (sum>max_finite_double) return {-1,node};
    // Scale exponents separately: 2*CFL*V may overflow before division.
    int ec,ev,es;
    const double mc=::frexp(cfl,&ec),mv=::frexp(volume,&ev),ms=::frexp(sum,&es);
    const double dt=::ldexp(mc*mv/ms,1+ec+ev-es);
    if (dt>max_finite_double) return unconstrained();
    return {dt>0 ? dt : -1,node};
}
struct FaceSpeed { double speed,area; };
template<class Offset,class ReadFace>
AERO_CFL_HD inline CflStep local(const Offset* offsets,double volume,double cfl,size_t node,ReadFace read_face) {
    double sum=0;
    for (Offset j=offsets[node];j<offsets[node+1];++j) {
        const FaceSpeed face=read_face(j);
        if (!(face.speed>=0 && face.speed<=max_finite_double)) return {-1,node};
        sum+=face.speed*face.area;
    }
    return from_sum(sum,volume,cfl,node);
}
struct GridSpeeds {
    const Entry* entries;
    WaveSpeeds speeds;
    AERO_CFL_HD FaceSpeed operator()(size_t j) const {
        const Entry e=entries[j];
        return {e.face<speeds.edge_count ? speeds.edge[e.face] : speeds.boundary[e.face-speeds.edge_count],e.area};
    }
};
AERO_CFL_HD inline CflStep local(Grid grid,WaveSpeeds speeds,double cfl,size_t node) {
    return local(grid.offsets,grid.volumes[node],cfl,node,GridSpeeds{grid.entries,speeds});
}
#undef AERO_CFL_HD
#ifndef __CUDACC__
template<class ReadNode>
CflStep host_min(size_t nodes,const ReadNode& read_node) {
    CflStep result=unconstrained();
    const auto count=static_cast<ptrdiff_t>(nodes);
#ifdef _OPENMP
#pragma omp parallel if(count>=4096)
#endif
    {
        CflStep best=unconstrained();
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (ptrdiff_t n=0;n<count;++n) best=minimum(best,read_node(static_cast<size_t>(n)));
#ifdef _OPENMP
#pragma omp critical(aero_cfl_minimum)
#endif
        result=minimum(result,best);
    }
    return result;
}
#endif
// Opaque persistent device copy of Grid plus reduction workspace.
void* upload(Grid grid,int device);
void release(void* storage) noexcept;
CflStep reduce(void* storage,WaveSpeeds speeds,double cfl,void* stream);
}}
#endif
