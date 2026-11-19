#ifndef AERO_QA_CUDA_TEST_BUFFER_H
#define AERO_QA_CUDA_TEST_BUFFER_H

#include "../source/cuda/check.h"
#include <cstddef>

namespace aero_test {
// Владеет массивом на текущем GPU. Длина всех операций задаётся в элементах T.
template <class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t size) {
        if (size)
            aero::cuda_detail::checked(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data)
            cudaFree(data);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const T* source, std::size_t size) {
        if (size)
            aero::cuda_detail::checked(cudaMemcpy(data, source, size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* destination, std::size_t size) const {
        if (size)
            aero::cuda_detail::checked(
                cudaMemcpy(destination, data, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    // Оставляем передачу в том же потоке, в котором следующий этап будет читать эти данные.
    void upload_async(const T* source, std::size_t size, cudaStream_t stream) {
        if (size)
            aero::cuda_detail::checked(
                cudaMemcpyAsync(data, source, size * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
};
}
#endif
