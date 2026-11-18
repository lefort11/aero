#include "../cuda/device_scope.h"
#include "timestep_cuda.cuh"
#include <stdexcept>
#include <memory>

namespace aero { namespace cfl_detail {
namespace {
void checked(cudaError_t code) {
    if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> void allocate(T*& dst,std::size_t count) {
    checked(cudaMalloc(reinterpret_cast<void**>(&dst),count*sizeof(T)));
}
template<class T> void copy(T*& dst,const T* src,std::size_t count) {
    allocate(dst,count);
    checked(cudaMemcpyAsync(dst,src,count*sizeof(T),cudaMemcpyHostToDevice,0));
}
struct Storage {
    int device;
    std::size_t nodes;
    unsigned blocks;
    std::size_t* offsets=nullptr;
    Entry* entries=nullptr;
    double* volumes=nullptr;
    CflStep *partial=nullptr,*answer=nullptr;
    // Caller activates the owning device before destruction.
    ~Storage() { cudaFree(answer); cudaFree(partial); cudaFree(volumes); cudaFree(entries); cudaFree(offsets); }
};
using namespace aero::timestep_cuda_kernels;
}
void* upload(Grid grid,int device) {
    cuda_detail::DeviceScope scope(device);
    checked(scope.status);
    std::unique_ptr<Storage> s(new Storage);
    s->device=device;
    s->nodes=grid.nodes;
    const std::size_t blocks=(grid.nodes+block_threads-1)/block_threads;
    s->blocks=static_cast<unsigned>(blocks>1024?1024:blocks);
    copy(s->volumes,grid.volumes,grid.nodes);
    copy(s->entries,grid.entries,grid.offsets[grid.nodes]);
    copy(s->offsets,grid.offsets,grid.nodes+1);
    allocate(s->partial,s->blocks);
    allocate(s->answer,1);
    checked(cudaStreamSynchronize(0));
    return s.release();
}
void release(void* ptr) noexcept {
    Storage* s=static_cast<Storage*>(ptr);
    cuda_detail::DeviceScope scope(s->device);
    if (scope.status==cudaSuccess) delete s;
}
CflStep reduce(void* ptr,WaveSpeeds speeds,double cfl,void* stream_handle) {
    Storage& s=*static_cast<Storage*>(ptr);
    cuda_detail::DeviceScope scope(s.device);
    checked(scope.status);
    cudaStream_t stream=static_cast<cudaStream_t>(stream_handle);
    Grid grid={s.nodes,s.offsets,s.entries,s.volumes};
    AERO_LAUNCH(nodal_min, s.blocks, block_threads, stream, grid,speeds,cfl,s.partial);
    checked(cudaGetLastError());
    AERO_LAUNCH(global_min, 1, block_threads, stream, s.partial,s.blocks,s.answer);
    checked(cudaGetLastError());
    CflStep result;
    checked(cudaMemcpyAsync(&result,s.answer,sizeof(result),cudaMemcpyDeviceToHost,stream));
    checked(cudaStreamSynchronize(stream));
    return result;
}
}}
