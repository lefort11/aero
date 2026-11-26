#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "device_stencil.h"
#include "../parallel/device_arrays.h"
#include "../parallel/device_check.h"
#include <cstdio>
#include <memory>

namespace {
// Дескриптор GPU-шаблона и его рабочего состояния.
struct State {
    // Адреса таблиц тестового шаблона.
    aero::ReconstructionStencilView view{};
    aero::gpu::DeviceArrays storage; // Освобождает и частично загруженный шаблон при ошибке.
};
}

// Возвращает число доступных устройств через интерфейс GPU-модуля реконструкции.
extern "C" int aero_reconstruction_device_count(int* count, char* error, unsigned capacity) {
    return aero::gpu::available_devices(count, error, capacity);
}
// Освобождает GPU-копию шаблона реконструкции.
extern "C" void aero_reconstruction_device_destroy(void* opaque) {
    delete static_cast<State*>(opaque);
}
// Создаёт GPU-копию шаблона реконструкции и возвращает её дескриптор.
extern "C" int aero_reconstruction_device_create(aero::ReconstructionStencilView view, int device,
                                               void** opaque, char* error, unsigned capacity) {
    *opaque = nullptr;
    aero::gpu::DeviceScope scope(device);
    if (scope.status != aero::gpu::Status::Success)
        return aero::gpu::report_status(scope.status, error, capacity);
    try {
        std::unique_ptr<State> state(new State);
        state->storage.device = device;
        state->view = aero::upload_reconstruction_stencil(state->storage,view);
        // Внешний вызывающий может сразу читать шаблон в другом потоке GPU.
        const auto status = aero::gpu::wait(nullptr);
        if (status != aero::gpu::Status::Success) return aero::gpu::report_status(status,error,capacity);
        *opaque = state.release();
        return 0;
    } catch (const std::bad_alloc&) {
        return aero::gpu::report_status(aero::gpu::Status::OutOfMemory,error,capacity);
    } catch (const std::exception& e) {
        if (error && capacity) std::snprintf(error,capacity,"CUDA: %s",e.what());
        return 1;
    }
}
// Возвращает представление шаблона из GPU-дескриптора.
extern "C" void aero_reconstruction_device_view(void* opaque, aero::ReconstructionStencilView* view) {
    *view = static_cast<State*>(opaque)->view;
}
