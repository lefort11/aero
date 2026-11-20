#ifndef AERO_PINNED_BUFFER_H
#define AERO_PINNED_BUFFER_H
#include "device_memory.h"
#include <limits>
#include <memory>
#include <stdexcept>

namespace aero {
// Закреплённая память CPU для обмена с картой. Если устройство поддерживает
// отображение, ядро может обратиться к той же памяти по адресу device().
template <class T> class PinnedBuffer {
    struct Release {
        void operator()(T* p) const {
            aero::gpu::wait(nullptr);
            aero::gpu::release_pinned(p);
        }
    };
    std::unique_ptr<T, Release> host_; // Владельцем остаётся объект на CPU.
    T* device_ = nullptr;              // Адрес на GPU может отличаться от адреса на CPU.
public:
    // Возвращает ошибку выделения вызывающему коду. Для необязательного отображения
    // отсутствие поддержки штатно: буфер CPU остаётся доступен для обычного копирования.
    aero::gpu::Status allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::length_error("Pinned buffer too large");
        void* memory = nullptr;
        auto status = aero::gpu::allocate_pinned(&memory, count * sizeof(T));
        if (status != aero::gpu::Status::Success)
            return status;
        host_.reset(static_cast<T*>(memory));
        void* alias = nullptr;
        status = aero::gpu::mapped_pointer(&alias, memory);
        device_ = status == aero::gpu::Status::Success ? static_cast<T*>(alias) : nullptr;
        if (status == aero::gpu::Status::Invalid || status == aero::gpu::Status::Unsupported) {

            return aero::gpu::Status::Success;
        }
        return status;
    }
    T* host() const { return host_.get(); }
    T* device() const { return device_; }
};
}
#endif
