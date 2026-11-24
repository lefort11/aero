#ifndef AERO_TESTS_DEVICE_TEST_BUFFER_H
#define AERO_TESTS_DEVICE_TEST_BUFFER_H

#include "../source/parallel/device_check.h"
#include <cstddef>

namespace aero_test {
// Владеет массивом на текущем GPU. Длина всех операций задаётся в элементах T.
template <class T> class DeviceBuffer {
public:
    // Массив устройства, которым владеет тестовый объект.
    T* data = nullptr;
    // Выделяет временный массив заданного размера в памяти GPU.
    explicit DeviceBuffer(std::size_t size) {
        if (size)
            aero::gpu::checked(aero::gpu::allocate(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~DeviceBuffer() {
        if (data)
            aero::gpu::release(data);
    }
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    DeviceBuffer(const DeviceBuffer&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Копирует исходные тестовые данные с CPU на GPU.
    void upload(const T* source, std::size_t size) {
        if (size)
            aero::gpu::checked(aero::gpu::copy(data, source, size * sizeof(T), aero::gpu::Copy::ToDevice));
    }
    // Копирует результат проверки с GPU в CPU-буфер.
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
