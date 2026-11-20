#ifndef AERO_PINNED_BUFFER_H
#define AERO_PINNED_BUFFER_H
#include "runtime.h"
#include <limits>
#include <memory>
#include <stdexcept>

namespace aero {
// Закреплённая память CPU для обмена с картой. Если устройство поддерживает
// отображение, ядро может обратиться к той же памяти по адресу device().
template <class T> class PinnedBuffer {
    struct Release {
        void operator()(T* p) const {
            cudaStreamSynchronize(nullptr);
            cudaFreeHost(p);
        }
    };
    std::unique_ptr<T, Release> host_; // Владельцем остаётся объект на CPU.
    T* device_ = nullptr;              // Адрес на GPU может отличаться от адреса на CPU.
public:
    // Возвращает ошибку выделения вызывающему коду. Для необязательного отображения
    // отсутствие поддержки штатно: буфер CPU остаётся доступен для обычного копирования.
    cudaError_t allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::length_error("Pinned buffer too large");
        void* memory = nullptr;
        auto status = cudaMallocHost(&memory, count * sizeof(T));
        if (status != cudaSuccess)
            return status;
        host_.reset(static_cast<T*>(memory));
        void* alias = nullptr;
        status = cudaHostGetDevicePointer(&alias, memory, 0);
        device_ = status == cudaSuccess ? static_cast<T*>(alias) : nullptr;
        if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported) {
            cudaGetLastError();
            return cudaSuccess;
        }
        return status;
    }
    T* host() const { return host_.get(); }
    T* device() const { return device_; }
};
}
#endif
