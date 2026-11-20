#ifndef AERO_QA_DEVICE_TEST_BUFFER_H
#define AERO_QA_DEVICE_TEST_BUFFER_H

#include "../source/parallel/device_check.h"
#include <cstddef>

namespace aero_test {
// Владеет массивом на текущем GPU. Длина всех операций задаётся в элементах T.
template <class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t size) {
        if (size)
            aero::gpu::checked(aero::gpu::allocate(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data)
            aero::gpu::release(data);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const T* source, std::size_t size) {
        if (size)
            aero::gpu::checked(aero::gpu::copy(data, source, size * sizeof(T), aero::gpu::Copy::ToDevice));
    }
    void download(T* destination, std::size_t size) const {
        if (size)
            aero::gpu::checked(
                aero::gpu::copy(destination, data, size * sizeof(T), aero::gpu::Copy::ToHost));
    }
    // Оставляем передачу в том же потоке, в котором следующий этап будет читать эти данные.
    void upload_async(const T* source, std::size_t size, aero::gpu::Stream stream) {
        if (size)
            aero::gpu::checked(
                aero::gpu::copy_async(data, source, size * sizeof(T), aero::gpu::Copy::ToDevice, stream));
    }
};
}
#endif
