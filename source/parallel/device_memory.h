#ifndef AERO_PARALLEL_DEVICE_MEMORY_H
#define AERO_PARALLEL_DEVICE_MEMORY_H
#include "device_api.h"
#include <cstddef>

namespace aero {
namespace gpu {
// Интерфейс хоста с исполнителем Kokkos. Указатели относятся к выбранной карте,
// stream задаёт порядок команд. Нулевой stream означает стандартную очередь.
using Stream = void*;
enum class Status { Success, NoDevice, Unavailable, OutOfMemory, Invalid, Unsupported, Failed };
enum class Copy { ToDevice, ToHost, OnDevice };

AERO_DEVICE_ENTRY const char* message(Status);
AERO_DEVICE_ENTRY Status count(int* devices);
AERO_DEVICE_ENTRY Status current(int* device);
AERO_DEVICE_ENTRY Status select(int device);
AERO_DEVICE_ENTRY Status bus_id(char* text, int capacity, int device);
AERO_DEVICE_ENTRY Status memory_info(std::size_t* available, std::size_t* total);

// Все размеры указаны в байтах. Между стадиями выделенный Kokkos буфер остаётся на карте.
AERO_DEVICE_ENTRY Status allocate(void** out, std::size_t bytes);
template <class T> inline Status allocate(T** out, std::size_t bytes) {
    void* buffer = nullptr;
    const auto status = allocate(&buffer, bytes);
    *out = static_cast<T*>(buffer);
    return status;
}
AERO_DEVICE_ENTRY Status release(void* data);
AERO_DEVICE_ENTRY Status allocate_pinned(void** out, std::size_t bytes);
AERO_DEVICE_ENTRY Status release_pinned(void* data);
AERO_DEVICE_ENTRY Status mapped_pointer(void** alias, void* host);
AERO_DEVICE_ENTRY Status copy_async(void* dst, const void* src, std::size_t bytes, Copy direction, Stream stream);
AERO_DEVICE_ENTRY Status fill_async(void* dst, int value, std::size_t bytes, Stream stream);
AERO_DEVICE_ENTRY Status wait(Stream stream);
AERO_DEVICE_ENTRY Status wait_all();
inline Status copy(void* dst, const void* src, std::size_t bytes, Copy direction) {
    const auto status = copy_async(dst, src, bytes, direction, nullptr);
    return status == Status::Success ? wait(nullptr) : status;
}
AERO_DEVICE_ENTRY Status create_stream(Stream* stream, bool nonblocking = false);
AERO_DEVICE_ENTRY Status destroy_stream(Stream stream);
}
}
#endif
