#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "device_stencil.h"
#include "../parallel/device_arrays.h"
#include <cstdio>
#include <memory>

namespace {
struct State {
    aero::ReconstructionStencilView view{};
    aero::gpu::DeviceArrays storage; // Освобождает и частично загруженный шаблон при ошибке.
};

int report(aero::gpu::Status code, char* error, unsigned capacity) {
    if (code != aero::gpu::Status::Success && error && capacity)
        std::snprintf(error, capacity, "CUDA: %s", aero::gpu::message(code));
    return code == aero::gpu::Status::Success ? 0 : 1;
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
    delete static_cast<State*>(opaque);
}
extern "C" int aero_reconstruction_device_create(aero::ReconstructionStencilView view, int device,
                                               void** opaque, char* error, unsigned capacity) {
    *opaque = nullptr;
    aero::gpu::DeviceScope scope(device);
    if (scope.status != aero::gpu::Status::Success)
        return report(scope.status, error, capacity);
    try {
        std::unique_ptr<State> state(new State);
        state->storage.device = device;
        state->view = aero::upload_reconstruction_stencil(state->storage,view);
        // Внешний вызывающий может сразу читать шаблон в другом потоке GPU.
        const auto status = aero::gpu::wait(nullptr);
        if (status != aero::gpu::Status::Success) return report(status,error,capacity);
        *opaque = state.release();
        return 0;
    } catch (const std::bad_alloc&) {
        return report(aero::gpu::Status::OutOfMemory,error,capacity);
    } catch (const std::exception& e) {
        if (error && capacity) std::snprintf(error,capacity,"CUDA: %s",e.what());
        return 1;
    }
}
extern "C" void aero_reconstruction_device_view(void* opaque, aero::ReconstructionStencilView* view) {
    *view = static_cast<State*>(opaque)->view;
}
