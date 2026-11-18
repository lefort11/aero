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
AERO_CFL_HD inline CflStep local(Grid grid,WaveSpeeds speeds,double cfl,size_t node) {
    double sum=0;
    for (size_t j=grid.offsets[node];j<grid.offsets[node+1];++j) {
        const Entry e=grid.entries[j];
        const double speed=e.face<speeds.edge_count ? speeds.edge[e.face] : speeds.boundary[e.face-speeds.edge_count];
        if (!(speed>=0 && speed<=max_finite_double)) return {-1,node};
        sum+=speed*e.area;
    }
    if (sum==0) return unconstrained();
    if (sum>max_finite_double) return {-1,node};
    // Scale exponents separately: 2*CFL*V may overflow before division.
    int ec,ev,es;
    const double mc=::frexp(cfl,&ec),mv=::frexp(grid.volumes[node],&ev),ms=::frexp(sum,&es);
    const double dt=::ldexp(mc*mv/ms,1+ec+ev-es);
    if (dt>max_finite_double) return unconstrained();
    return {dt>0 ? dt : -1,node};
}
#undef AERO_CFL_HD
// Opaque persistent device copy of Grid plus reduction workspace.
void* upload(Grid grid,int device);
void release(void* storage) noexcept;
CflStep reduce(void* storage,WaveSpeeds speeds,double cfl,void* stream);
}}
#endif
