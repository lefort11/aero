#ifndef AERO_PARTITION_KERNELS_CUH
#define AERO_PARTITION_KERNELS_CUH
#include "partition_kernels.h"
#include "../cuda/runtime.h"
#ifndef AERO_MINGW_CUDA_HOST
namespace aero { namespace partition_detail {
AERO_GPU_KERNEL void partition_update(Grid g,explicit_detail::Stage stage,double dt,bool save_original) {
    for (Index k=blockIdx.x*blockDim.x+threadIdx.x;k<g.owned_count;k+=gridDim.x*blockDim.x)
        if (!update_node(g.geometry,g.flux,g.volumes,g.owned,g.state,g.original,k,stage,dt,save_original))
            atomicMin(g.bad,Index(g.owned?g.owned[k]:k));
}
AERO_GPU_KERNEL void partition_transfer(Grid g,const MeshIndex* ids,Index count,double* buffer,bool edges,bool pack) {
    const unsigned width=g.geometry.dim+2;
    for (Index k=blockIdx.x*blockDim.x+threadIdx.x;k<count;k+=gridDim.x*blockDim.x) {
        transfer_record(ids?ids[k]:k,width,g.state,g.flux,buffer+k*(width+unsigned(edges)),edges,pack);
    }
}
}}
#endif
#endif
