#include "../parallel/device_check.h"
#include "../parallel/device_scope.h"
#include "../parallel/device_work.h"
#include <stdexcept>
#include <memory>

namespace aero {
namespace cfl_detail {
namespace {
using aero::gpu::checked;
template <class T> void allocate(T*& dst, std::size_t count) {
    checked(aero::gpu::allocate(reinterpret_cast<void**>(&dst), count * sizeof(T)));
}
template <class T> void copy(T*& dst, const T* src, std::size_t count) {
    allocate(dst, count);
    checked(aero::gpu::copy_async(dst, src, count * sizeof(T), aero::gpu::Copy::ToDevice, 0));
}
struct Storage {
    int device;
    std::size_t nodes;
    std::size_t* offsets = nullptr;
    Entry* entries = nullptr;
    double* volumes = nullptr;
    // Перед уничтожением вызывающий код активирует устройство, которому принадлежит память.
    ~Storage() {
        aero::gpu::release(volumes);
        aero::gpu::release(entries);
        aero::gpu::release(offsets);
    }
};
}
void* upload(Grid grid, int device) {
    gpu::DeviceScope scope(device);
    checked(scope.status);
    std::unique_ptr<Storage> s(new Storage);
    s->device = device;
    s->nodes = grid.nodes;
    copy(s->volumes, grid.volumes, grid.nodes);
    copy(s->entries, grid.entries, grid.offsets[grid.nodes]);
    copy(s->offsets, grid.offsets, grid.nodes + 1);
    checked(aero::gpu::wait(0));
    return s.release();
}
void release(void* ptr) noexcept {
    Storage* s = static_cast<Storage*>(ptr);
    gpu::DeviceScope scope(s->device);
    if (scope.status == aero::gpu::Status::Success)
        delete s;
}
CflStep reduce(void* ptr, WaveSpeeds speeds, double cfl, void* stream_handle) {
    Storage& s = *static_cast<Storage*>(ptr);
    gpu::DeviceScope scope(s.device);
    checked(scope.status);
    Grid grid = {s.nodes, s.offsets, s.entries, s.volumes};
    CflStep result;
    device_work::check(device_work::aero_device_wave_step(grid, speeds, cfl, &result, stream_handle));
    return result;
}
}
}
