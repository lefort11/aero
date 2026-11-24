// Вся память численных модулей проходит через пространства памяти Kokkos.
// CUDA 12 обслуживает выбор физической карты и внешних потоков исполнения:
// в публичном интерфейсе Kokkos 3.2 отсутствуют PCI-адрес, остаток памяти
// карты и создание совместимой очереди для вызывающей программы.
#define AERO_BUILD_DEVICE_WORK
#include "device_execution.h"
#include "device_memory.h"
#include <new>
#include <cstdio>

namespace aero {
namespace gpu {
namespace {
using device_work::Execution;
using Device = Execution::memory_space;
using Host = Kokkos::HostSpace;
using Pinned = Kokkos::CudaHostPinnedSpace;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
thread_local std::string last_error;

// У MinGW и CUDA могут быть разные библиотеки C++. Ошибки возвращаем на хост
// как небольшой код и текст, исключение остаётся на стороне исполнителя.
template <class Work> Status run(const Work& work) {
    try {
        work();
        return Status::Success;
    } catch (const std::bad_alloc& error) {
        last_error = error.what();
        return Status::OutOfMemory;
    } catch (const std::exception& error) {
        last_error = error.what();
        return Status::Failed;
    }
}
// Переводит код ошибки CUDA в статус GPU-интерфейса и сохраняет текст причины.
Status platform(cudaError_t result) {
    if (result == cudaSuccess)
        return Status::Success;
    last_error = cudaGetErrorString(result);
    cudaGetLastError();
    switch (result) {
    case cudaErrorNoDevice: return Status::NoDevice;
    case cudaErrorInsufficientDriver: return Status::Unavailable;
    case cudaErrorMemoryAllocation: return Status::OutOfMemory;
    case cudaErrorInvalidValue: return Status::Invalid;
    case cudaErrorNotSupported: return Status::Unsupported;
    default: return Status::Failed;
    }
}
// Выделяет буфер Kokkos в заданном пространстве памяти; нулевой размер не требует выделения.
template <class Space> Status acquire(void** out, std::size_t bytes) {
    *out = nullptr;
    if (!bytes)
        return Status::Success;
    return run([&] {
        device_work::current_execution(nullptr); // Инициализируем Kokkos на выбранной карте.
        *out = Kokkos::kokkos_malloc<Space>("Aero buffer", bytes);
    });
}
// Освобождает непустой буфер в соответствующем пространстве памяти Kokkos.
template <class Space> Status discard(void* ptr) {
    return ptr ? run([&] { Kokkos::kokkos_free<Space>(ptr); }) : Status::Success;
}
// Переносит указанный диапазон байтов между источником и приёмником.
template <class To, class From>
void copy_bytes(const Execution& exec, void* dst, const void* src, std::size_t bytes) {
    Kokkos::View<unsigned char*, To, Unmanaged> target(static_cast<unsigned char*>(dst), bytes);
    Kokkos::View<const unsigned char*, From, Unmanaged> source(static_cast<const unsigned char*>(src), bytes);
    Kokkos::deep_copy(exec, target, source);
}
}

// Возвращает текст последней ошибки GPU либо сообщение об успехе.
const char* message(Status status) { return status == Status::Success ? "Success" : last_error.c_str(); }
// Запрашивает число доступных физических CUDA-устройств.
Status count(int* devices) { return platform(cudaGetDeviceCount(devices)); }
// Копирует имя выбранной карты в буфер указанной вместимости.
Status selected_device_name(char* name, std::size_t capacity) {
    return run([&] {
        const auto exec = device_work::current_execution(nullptr);
        std::snprintf(name, capacity, "%s", exec.cuda_device_prop().name);
    });
}
// Запрашивает номер текущего CUDA-устройства.
Status current(int* device) { return platform(cudaGetDevice(device)); }
// Выбирает CUDA-устройство для последующих операций текущего потока.
Status select(int device) { return platform(cudaSetDevice(device)); }
// Возвращает PCI-адрес карты для сопоставления физических устройств.
Status bus_id(char* text, int capacity, int device) {
    return platform(cudaDeviceGetPCIBusId(text, capacity, device));
}
// Запрашивает свободный и полный объём памяти текущей карты.
Status memory_info(std::size_t* available, std::size_t* total) {
    return platform(cudaMemGetInfo(available, total));
}
// Создаёт поток CUDA с заданным режимом синхронизации.
Status create_stream(Stream* stream, bool nonblocking) {
    cudaStream_t queue = nullptr;
    const auto result = platform(cudaStreamCreateWithFlags(&queue, nonblocking ? cudaStreamNonBlocking : cudaStreamDefault));
    *stream = queue;
    return result;
}
// Удаляет ранее созданный поток CUDA.
Status destroy_stream(Stream stream) { return platform(cudaStreamDestroy(static_cast<cudaStream_t>(stream))); }
// Ожидает завершения операций указанного потока.
Status wait(Stream stream) { return run([&] { device_work::current_execution(stream).fence(); }); }
// Ожидает завершения всех операций текущего устройства.
Status wait_all() { return run([] { Execution::impl_static_fence(); }); }
// Все размеры указаны в байтах. Между стадиями выделенный Kokkos буфер остаётся на карте.
Status allocate(void** out, std::size_t bytes) { return acquire<Device>(out, bytes); }
// Освобождает ранее выделенную память GPU.
Status release(void* ptr) { return discard<Device>(ptr); }
// Выделяет закреплённую память CPU для обменов с GPU.
Status allocate_pinned(void** out, std::size_t bytes) { return acquire<Pinned>(out, bytes); }
// Освобождает закреплённый буфер CPU.
Status release_pinned(void* ptr) { return discard<Pinned>(ptr); }
// Получает адрес закреплённого буфера, доступный из кода GPU.
Status mapped_pointer(void** alias, void* host) { return platform(cudaHostGetDevicePointer(alias, host, 0)); }
// Ставит копирование байтов в очередь указанного потока.
Status copy_async(void* dst, const void* src, std::size_t bytes, Copy direction, Stream stream) {
    if (!bytes)
        return Status::Success;
    return run([&] {
        const auto exec = device_work::current_execution(stream);
        switch (direction) {
        case Copy::ToDevice: copy_bytes<Device, Host>(exec, dst, src, bytes); break;
        case Copy::ToHost: copy_bytes<Host, Device>(exec, dst, src, bytes); break;
        case Copy::OnDevice: copy_bytes<Device, Device>(exec, dst, src, bytes); break;
        }
    });
}
// Ставит заполнение буфера байтовым значением в очередь указанного потока.
Status fill_async(void* dst, int value, std::size_t bytes, Stream stream) {
    return run([&] {
        Kokkos::View<unsigned char*, Device, Unmanaged> target(static_cast<unsigned char*>(dst), bytes);
        Kokkos::deep_copy(device_work::current_execution(stream), target, static_cast<unsigned char>(value));
    });
}
}
}
