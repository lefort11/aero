#ifndef AERO_GPU_DEVICE_SCOPE_H
#define AERO_GPU_DEVICE_SCOPE_H
#include "device_memory.h"

namespace aero {
namespace gpu {
// Временно делает устройство текущим для вызывающего CPU-потока и возвращает прежнее при выходе из области.
class DeviceScope {
    // Сохраняет прежний номер устройства. Отрицательное значение означает, что восстановление не требуется.
    int previous_ = -1;

public:
    aero::gpu::Status status = aero::gpu::Status::Success;
    explicit DeviceScope(int device) noexcept {
        if (device < 0)
            return;
        status = aero::gpu::current(&previous_);
        if (status == aero::gpu::Status::Success && previous_ != device)
            status = aero::gpu::select(device);
    }
    ~DeviceScope() {
        if (previous_ >= 0)
            aero::gpu::select(previous_);
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
};
}
}
#endif
