#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "reconstruction.h"
#include "../cuda/runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

namespace {
using aero::ReconstructionIndex;
struct State {
    aero::ReconstructionStencilView view;
    int device;
    ReconstructionIndex* offsets;
    ReconstructionIndex* nodes;
    double* weights;
    ReconstructionIndex* row_owner;
    ReconstructionIndex* slope_offsets;
    ReconstructionIndex* slope_nodes;
    double* slope_weights;
    unsigned char* weno_order;
    double* weno_epsilon_scale;
};

int report(cudaError_t code, char* error, unsigned capacity) {
    if (code != cudaSuccess && error && capacity)
        snprintf(error,capacity,"CUDA: %s",cudaGetErrorString(code));
    return code == cudaSuccess ? 0 : 1;
}
struct DeviceScope {
    int previous;
    cudaError_t status;
    explicit DeviceScope(int device) : previous(-1), status(cudaGetDevice(&previous)) {
        if (status==cudaSuccess) status=cudaSetDevice(device);
    }
    ~DeviceScope() { if (previous>=0) cudaSetDevice(previous); }
};
cudaError_t copy_array(void** out, const void* src, size_t bytes) {
    if (!bytes) return cudaSuccess;
    cudaError_t code=cudaMalloc(out,bytes);
    if (code==cudaSuccess) code=cudaMemcpy(*out,src,bytes,cudaMemcpyHostToDevice);
    return code;
}
}

extern "C" int aero_reconstruction_cuda_count(int* count, char* error, unsigned capacity) {
    *count=0;
    cudaError_t code=cudaGetDeviceCount(count);
    if (code==cudaErrorNoDevice || code==cudaErrorInsufficientDriver) {
        cudaGetLastError(); *count=0; return 0;
    }
    return report(code,error,capacity);
}
extern "C" void aero_reconstruction_cuda_destroy(void* opaque) {
    State* s=static_cast<State*>(opaque);
    if (!s) return;
    DeviceScope scope(s->device);
    cudaFree(s->offsets); cudaFree(s->nodes); cudaFree(s->weights);
    cudaFree(s->slope_offsets); cudaFree(s->slope_nodes); cudaFree(s->slope_weights);
    cudaFree(s->weno_order); cudaFree(s->weno_epsilon_scale);
    cudaFree(s->row_owner);
    free(s);
}
extern "C" int aero_reconstruction_cuda_create(aero::ReconstructionStencilView view, int device, void** opaque,
                                    char* error, unsigned capacity) {
    *opaque=nullptr;
    DeviceScope scope(device);
    if (scope.status!=cudaSuccess) return report(scope.status,error,capacity);
    State* s=static_cast<State*>(calloc(1,sizeof(State)));
    if (!s) return report(cudaErrorMemoryAllocation,error,capacity);
    s->device=device; s->view=view;
    const size_t entries=static_cast<size_t>(view.offsets[2*view.edge_count]);
    cudaError_t code=copy_array(reinterpret_cast<void**>(&s->offsets),view.offsets,
                               static_cast<size_t>(2*view.edge_count+1)*sizeof(ReconstructionIndex));
    if (code==cudaSuccess) code=copy_array(reinterpret_cast<void**>(&s->nodes),view.nodes,entries*sizeof(ReconstructionIndex));
    if (code==cudaSuccess) code=copy_array(reinterpret_cast<void**>(&s->weights),view.weights,entries*sizeof(double));
    if (code==cudaSuccess) code=copy_array(reinterpret_cast<void**>(&s->row_owner),view.row_owner,
        static_cast<size_t>(2*view.edge_count)*sizeof(ReconstructionIndex));
    if (code==cudaSuccess && view.weno_order) {
        const size_t rows=static_cast<size_t>(2*view.edge_count);
        const size_t slope_entries=static_cast<size_t>(view.slope_offsets[4*rows]);
        code=copy_array(reinterpret_cast<void**>(&s->slope_offsets),view.slope_offsets,(4*rows+1)*sizeof(ReconstructionIndex));
        if (code==cudaSuccess) code=copy_array(reinterpret_cast<void**>(&s->slope_nodes),view.slope_nodes,slope_entries*sizeof(ReconstructionIndex));
        if (code==cudaSuccess) code=copy_array(reinterpret_cast<void**>(&s->slope_weights),view.slope_weights,slope_entries*sizeof(double));
        if (code==cudaSuccess) code=copy_array(reinterpret_cast<void**>(&s->weno_order),view.weno_order,rows*sizeof(unsigned char));
        if (code==cudaSuccess) code=copy_array(reinterpret_cast<void**>(&s->weno_epsilon_scale),view.weno_epsilon_scale,rows*sizeof(double));
    }
    // The caller can use the stencil on a different stream as soon as creation returns.
    if (code==cudaSuccess) code=cudaStreamSynchronize(nullptr);
    if (code!=cudaSuccess) { aero_reconstruction_cuda_destroy(s); return report(code,error,capacity); }
    s->view.row_owner=s->row_owner;
    s->view.offsets=s->offsets; s->view.nodes=s->nodes; s->view.weights=s->weights;
    s->view.slope_offsets=s->slope_offsets; s->view.slope_nodes=s->slope_nodes; s->view.slope_weights=s->slope_weights;
    s->view.weno_order=s->weno_order; s->view.weno_epsilon_scale=s->weno_epsilon_scale;
    *opaque=s;
    return 0;
}
extern "C" void aero_reconstruction_cuda_view(void* opaque,aero::ReconstructionStencilView* view) {
    *view=static_cast<State*>(opaque)->view;
}
