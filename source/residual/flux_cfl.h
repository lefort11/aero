#ifndef AERO_FLUX_CFL_H
#define AERO_FLUX_CFL_H
#include "residual_detail.h"
#include "../timestep/cfl_detail.h"
#include "../mesh/mesh_index.h"

namespace aero { namespace residual_detail {
struct FaceSpeeds {
    const Geometry& geometry;
    const EulerFluxResult* flux;
#ifdef __CUDACC__
    __host__ __device__
#endif
    cfl_detail::FaceSpeed operator()(Index j) const {
        const auto face=geometry.incidence[j]/2;
        return {flux[face].max_wave_speed,geometry.surfaces[face].measure};
    }
};
// A null node list selects the whole mesh. A partition supplies its owned nodes.
struct FluxCfl {
    Geometry geometry;
    const EulerFluxResult* flux;
    const double* volumes;
    const MeshIndex* owned;
    Index count;
#ifdef __CUDACC__
    __host__ __device__
#endif
    CflStep local(Index k,double courant) const {
        const Index i=owned?owned[k]:k;
        const double volume=volumes[i];
        if (!owned && (!(volume>0 && volume<=max_finite_double) || geometry.offsets[i]==geometry.offsets[i+1])) return {-1,i};
        return cfl_detail::local(geometry.offsets,volume,courant,i,FaceSpeeds{geometry,flux});
    }
};
struct DeviceCflMinimum { Index bad,bits; };
double cpu_flux_cfl(const FluxCfl&,double courant,const MeshIndex* global_ids=nullptr);
double gpu_flux_cfl(const FluxCfl&,double courant,DeviceCflMinimum*,void* stream);
}}
#endif
