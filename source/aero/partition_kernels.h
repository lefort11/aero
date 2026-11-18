#ifndef AERO_PARTITION_KERNELS_H
#define AERO_PARTITION_KERNELS_H
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#include "../mesh/mesh_index.h"

namespace aero { namespace partition_detail {
using residual_detail::Index;
// Pointers in this view belong to the rank's selected device.
struct Grid {
    residual_detail::Geometry geometry;
    const MeshIndex *computed,*owned;
    Index computed_count,owned_count;
    const double* volumes;
    double *initial,*state,*original;
    EulerFluxResult* flux;
    Index* bad;
};
#ifdef __CUDACC__
#define AERO_PARTITION_HD __host__ __device__
#else
#define AERO_PARTITION_HD
#endif
AERO_PARTITION_HD inline bool update_node(const residual_detail::Geometry& geometry,const EulerFluxResult* flux,
        const double* volumes,const MeshIndex* owned,double* state,double* original,Index k,
        explicit_detail::Stage stage,double dt,bool save_original) {
    const Index i=owned?owned[k]:k;
    const unsigned width=geometry.dim+2;
    double residual[5];
    if (!residual_detail::gather_node(geometry,i,flux,residual)) return false;
    for (unsigned v=0;v<width;++v) {
        double& u=state[i*width+v];
        if (save_original) original[k*width+v]=u;
        const double start=stage.original_weight==0?u:original[k*width+v];
        if (!explicit_detail::update_value(stage,start,u,dt,volumes[i],residual[v],u)) return false;
    }
    return true;
}
AERO_PARTITION_HD inline void transfer_record(MeshIndex i,unsigned width,double* state,EulerFluxResult* flux,
                                               double* record,bool edges,bool pack) {
    double* values=edges?flux[i].flux:state+i*width;
    for (unsigned v=0;v<width;++v) {
        if (pack) record[v]=values[v];
        else values[v]=record[v];
    }
    if (edges) {
        if (pack) record[width]=flux[i].max_wave_speed;
        else flux[i].max_wave_speed=record[width];
    }
}
#undef AERO_PARTITION_HD
void update(Grid grid,explicit_detail::Stage stage,double dt,bool save_original);
void transfer(Grid grid,const MeshIndex* ids,Index count,double* records,bool edges,bool pack);
}}
#endif
