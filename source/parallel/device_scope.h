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
    // Результат установки текущего GPU.
    aero::gpu::Status status = aero::gpu::Status::Success;
    // Временно выбирает устройство, сохраняя предыдущее для восстановления.
    explicit DeviceScope(int device) noexcept {
        if (device < 0)
            return;
        status = aero::gpu::current(&previous_);
        if (status != aero::gpu::Status::Success || previous_ == device) {
            // В обычном MPI-процессе все вызовы относятся к его карте. При
            // выходе восстанавливать устройство нужно только после смены карты.
            previous_ = -1;
        } else {
            status = aero::gpu::select(device);
        }
    }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~DeviceScope() {
        if (previous_ >= 0)
            aero::gpu::select(previous_);
    }
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    DeviceScope(const DeviceScope&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    DeviceScope& operator=(const DeviceScope&) = delete;
};
}
}
#endif
