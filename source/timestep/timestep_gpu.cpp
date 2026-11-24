#include "../parallel/device_arrays.h"
#include "../parallel/device_work.h"
#include <memory>

namespace aero {
namespace cfl_detail {
namespace {
using aero::gpu::checked;
// Владеет устройством и массивами геометрии для оценки CFL на GPU.
struct Storage {
    gpu::DeviceArrays arrays;
    // Количество узлов в расчёте шага.
    std::size_t nodes;
    // Смещения строк площадок на GPU.
    std::size_t* offsets = nullptr;
    // Инцидентные площадки на GPU.
    Entry* entries = nullptr;
    // Объёмы узловых ячеек на GPU.
    double* volumes = nullptr;
};
}
// Копирует геометрию CFL на выбранную карту и возвращает её дескриптор.
void* upload(Grid grid, int device) {
    gpu::DeviceScope scope(device);
    checked(scope.status);
    std::unique_ptr<Storage> s(new Storage);
    s->arrays.device = device;
    s->nodes = grid.nodes;
    s->volumes = s->arrays.upload_async(grid.volumes, grid.nodes, nullptr);
    s->entries = s->arrays.upload_async(grid.entries, grid.offsets[grid.nodes], nullptr);
    s->offsets = s->arrays.upload_async(grid.offsets, grid.nodes + 1, nullptr);
    checked(aero::gpu::wait(0));
    return s.release();
}
// Выбирает карту-владельца и освобождает массивы геометрии CFL.
void release(void* ptr) noexcept {
    delete static_cast<Storage*>(ptr);
}
// Вычисляет минимальный шаг по узлам с помощью выбранного исполнителя.
CflStep reduce(void* ptr, WaveSpeeds speeds, double cfl, void* stream_handle) {
    Storage& s = *static_cast<Storage*>(ptr);
    gpu::DeviceScope scope(s.arrays.device);
    checked(scope.status);
    Grid grid = {s.nodes, s.offsets, s.entries, s.volumes};
    CflStep result;
    device_work::check(device_work::aero_device_wave_step(grid, speeds, cfl, &result, stream_handle));
    return result;
}
}
}
