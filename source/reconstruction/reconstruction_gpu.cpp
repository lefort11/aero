#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "reconstruction.h"
#include "../parallel/device_scope.h"
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

int report(aero::gpu::Status code, char* error, unsigned capacity) {
    if (code != aero::gpu::Status::Success && error && capacity)
        snprintf(error, capacity, "CUDA: %s", aero::gpu::message(code));
    return code == aero::gpu::Status::Success ? 0 : 1;
}
aero::gpu::Status copy_array(void** out, const void* src, size_t bytes) {
    if (!bytes)
        return aero::gpu::Status::Success;
    aero::gpu::Status code = aero::gpu::allocate(out, bytes);
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::copy(*out, src, bytes, aero::gpu::Copy::ToDevice);
    return code;
}
}

extern "C" int aero_reconstruction_device_count(int* count, char* error, unsigned capacity) {
    *count = 0;
    aero::gpu::Status code = aero::gpu::count(count);
    if (code == aero::gpu::Status::NoDevice || code == aero::gpu::Status::Unavailable) {

        *count = 0;
        return 0;
    }
    return report(code, error, capacity);
}
extern "C" void aero_reconstruction_device_destroy(void* opaque) {
    State* s = static_cast<State*>(opaque);
    if (!s)
        return;
    aero::gpu::DeviceScope scope(s->device);
    aero::gpu::release(s->offsets);
    aero::gpu::release(s->nodes);
    aero::gpu::release(s->weights);
    aero::gpu::release(s->slope_offsets);
    aero::gpu::release(s->slope_nodes);
    aero::gpu::release(s->slope_weights);
    aero::gpu::release(s->weno_order);
    aero::gpu::release(s->weno_epsilon_scale);
    aero::gpu::release(s->row_owner);
    free(s);
}
extern "C" int aero_reconstruction_device_create(aero::ReconstructionStencilView view, int device,
                                               void** opaque, char* error, unsigned capacity) {
    *opaque = nullptr;
    aero::gpu::DeviceScope scope(device);
    if (scope.status != aero::gpu::Status::Success)
        return report(scope.status, error, capacity);
    State* s = static_cast<State*>(calloc(1, sizeof(State)));
    if (!s)
        return report(aero::gpu::Status::OutOfMemory, error, capacity);
    s->device = device;
    s->view = view;
    const size_t entries = static_cast<size_t>(view.offsets[2 * view.edge_count]);
    aero::gpu::Status code = copy_array(reinterpret_cast<void**>(&s->offsets), view.offsets,
                                  static_cast<size_t>(2 * view.edge_count + 1) * sizeof(ReconstructionIndex));
    if (code == aero::gpu::Status::Success)
        code = copy_array(reinterpret_cast<void**>(&s->nodes), view.nodes,
                          entries * sizeof(ReconstructionIndex));
    if (code == aero::gpu::Status::Success)
        code = copy_array(reinterpret_cast<void**>(&s->weights), view.weights, entries * sizeof(double));
    if (code == aero::gpu::Status::Success)
        code = copy_array(reinterpret_cast<void**>(&s->row_owner), view.row_owner,
                          static_cast<size_t>(2 * view.edge_count) * sizeof(ReconstructionIndex));
    if (code == aero::gpu::Status::Success && view.weno_order) {
        const size_t rows = static_cast<size_t>(2 * view.edge_count);
        const size_t slope_entries = static_cast<size_t>(view.slope_offsets[4 * rows]);
        code = copy_array(reinterpret_cast<void**>(&s->slope_offsets), view.slope_offsets,
                          (4 * rows + 1) * sizeof(ReconstructionIndex));
        if (code == aero::gpu::Status::Success)
            code = copy_array(reinterpret_cast<void**>(&s->slope_nodes), view.slope_nodes,
                              slope_entries * sizeof(ReconstructionIndex));
        if (code == aero::gpu::Status::Success)
            code = copy_array(reinterpret_cast<void**>(&s->slope_weights), view.slope_weights,
                              slope_entries * sizeof(double));
        if (code == aero::gpu::Status::Success)
            code = copy_array(reinterpret_cast<void**>(&s->weno_order), view.weno_order,
                              rows * sizeof(unsigned char));
        if (code == aero::gpu::Status::Success)
            code = copy_array(reinterpret_cast<void**>(&s->weno_epsilon_scale), view.weno_epsilon_scale,
                              rows * sizeof(double));
    }
    // К моменту возврата геометрия доступна читателю и в другом потоке CUDA.
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::wait(nullptr);
    if (code != aero::gpu::Status::Success) {
        aero_reconstruction_device_destroy(s);
        return report(code, error, capacity);
    }
    s->view.row_owner = s->row_owner;
    s->view.offsets = s->offsets;
    s->view.nodes = s->nodes;
    s->view.weights = s->weights;
    s->view.slope_offsets = s->slope_offsets;
    s->view.slope_nodes = s->slope_nodes;
    s->view.slope_weights = s->slope_weights;
    s->view.weno_order = s->weno_order;
    s->view.weno_epsilon_scale = s->weno_epsilon_scale;
    *opaque = s;
    return 0;
}
extern "C" void aero_reconstruction_device_view(void* opaque, aero::ReconstructionStencilView* view) {
    *view = static_cast<State*>(opaque)->view;
}
