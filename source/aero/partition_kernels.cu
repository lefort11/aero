#define AERO_RESIDUAL_KERNEL_ONLY
#define AERO_BC_KERNEL_ONLY
#define AERO_EULERSOLVER_KERNEL_ONLY
#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "partition_kernels.cuh"
#include <stdexcept>

namespace aero { namespace partition_detail {
namespace {
unsigned blocks(Index count) {
    const Index n=(count+127)/128;
    return static_cast<unsigned>(n>65535?65535:n);
}
void launched() {
    const auto error=cudaGetLastError();
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
void update(Grid g,explicit_detail::Stage stage,double dt,bool save_original) {
    if (!g.owned_count) return;
    AERO_LAUNCH(partition_update,blocks(g.owned_count),128,nullptr,g,stage,dt,save_original);
    launched();
}
void transfer(Grid g,const MeshIndex* ids,Index count,double* records,bool edges,bool pack) {
    if (!count) return;
    AERO_LAUNCH(partition_transfer,blocks(count),128,nullptr,g,ids,count,records,edges,pack);
    launched();
}
}}
