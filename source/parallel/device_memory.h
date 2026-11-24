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

// Возвращает текст последней ошибки GPU либо сообщение об успехе.
AERO_DEVICE_ENTRY const char* message(Status);
// Запрашивает число доступных физических CUDA-устройств.
AERO_DEVICE_ENTRY Status count(int* devices);
// Копирует имя выбранной карты в буфер указанной вместимости.
AERO_DEVICE_ENTRY Status selected_device_name(char* name, std::size_t capacity);
// Запрашивает номер текущего CUDA-устройства.
AERO_DEVICE_ENTRY Status current(int* device);
// Выбирает CUDA-устройство для последующих операций текущего потока.
AERO_DEVICE_ENTRY Status select(int device);
// Возвращает PCI-адрес карты для сопоставления физических устройств.
AERO_DEVICE_ENTRY Status bus_id(char* text, int capacity, int device);
// Запрашивает свободный и полный объём памяти текущей карты.
AERO_DEVICE_ENTRY Status memory_info(std::size_t* available, std::size_t* total);

// Все размеры указаны в байтах. Между стадиями выделенный Kokkos буфер остаётся на карте.
AERO_DEVICE_ENTRY Status allocate(void** out, std::size_t bytes);
// Все размеры указаны в байтах. Между стадиями выделенный Kokkos буфер остаётся на карте.
template <class T> inline Status allocate(T** out, std::size_t bytes) {
    void* buffer = nullptr;
    const auto status = allocate(&buffer, bytes);
    *out = static_cast<T*>(buffer);
    return status;
}
// Освобождает ранее выделенную память GPU.
AERO_DEVICE_ENTRY Status release(void* data);
// Выделяет закреплённую память CPU для обменов с GPU.
AERO_DEVICE_ENTRY Status allocate_pinned(void** out, std::size_t bytes);
// Освобождает закреплённый буфер CPU.
AERO_DEVICE_ENTRY Status release_pinned(void* data);
// Получает адрес закреплённого буфера, доступный из кода GPU.
AERO_DEVICE_ENTRY Status mapped_pointer(void** alias, void* host);
// Ставит копирование байтов в очередь указанного потока.
AERO_DEVICE_ENTRY Status copy_async(void* dst, const void* src, std::size_t bytes, Copy direction, Stream stream);
// Ставит заполнение буфера байтовым значением в очередь указанного потока.
AERO_DEVICE_ENTRY Status fill_async(void* dst, int value, std::size_t bytes, Stream stream);
// Ожидает завершения операций указанного потока.
AERO_DEVICE_ENTRY Status wait(Stream stream);
// Ожидает завершения всех операций текущего устройства.
AERO_DEVICE_ENTRY Status wait_all();
// Копирует байты между CPU и GPU и ожидает завершения операции.
inline Status copy(void* dst, const void* src, std::size_t bytes, Copy direction) {
    const auto status = copy_async(dst, src, bytes, direction, nullptr);
    return status == Status::Success ? wait(nullptr) : status;
}
// Создаёт поток CUDA с заданным режимом синхронизации.
AERO_DEVICE_ENTRY Status create_stream(Stream* stream, bool nonblocking = false);
// Удаляет ранее созданный поток CUDA.
AERO_DEVICE_ENTRY Status destroy_stream(Stream stream);
}
}
#endif
