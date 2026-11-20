#include "../cuda/check.h"
#include "../cuda/device_scope.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#else
#include "timestep_cuda.cuh"
#endif
#include <stdexcept>
#include <memory>

namespace aero {
namespace cfl_detail {
namespace {
using aero::cuda_detail::checked;
template <class T> void allocate(T*& dst, std::size_t count) {
    checked(cudaMalloc(reinterpret_cast<void**>(&dst), count * sizeof(T)));
}
template <class T> void copy(T*& dst, const T* src, std::size_t count) {
    allocate(dst, count);
    checked(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, 0));
}
struct Storage {
    int device;
    std::size_t nodes;
#ifndef AERO_WITH_KOKKOS_GPU
    unsigned blocks;
    CflStep *partial = nullptr, *answer = nullptr;
#endif
    std::size_t* offsets = nullptr;
    Entry* entries = nullptr;
    double* volumes = nullptr;
    // Перед уничтожением вызывающий код активирует устройство, которому принадлежит память.
    ~Storage() {
#ifndef AERO_WITH_KOKKOS_GPU
        cudaFree(answer);
        cudaFree(partial);
#endif
        cudaFree(volumes);
        cudaFree(entries);
        cudaFree(offsets);
    }
};
#ifndef AERO_WITH_KOKKOS_GPU
using namespace aero::timestep_cuda_kernels;
#endif
}
void* upload(Grid grid, int device) {
    cuda_detail::DeviceScope scope(device);
    checked(scope.status);
    std::unique_ptr<Storage> s(new Storage);
    s->device = device;
    s->nodes = grid.nodes;
#ifndef AERO_WITH_KOKKOS_GPU
    const std::size_t blocks = (grid.nodes + block_threads - 1) / block_threads;
    s->blocks = static_cast<unsigned>(blocks > 1024 ? 1024 : blocks);
#endif
    copy(s->volumes, grid.volumes, grid.nodes);
    copy(s->entries, grid.entries, grid.offsets[grid.nodes]);
    copy(s->offsets, grid.offsets, grid.nodes + 1);
#ifndef AERO_WITH_KOKKOS_GPU
    allocate(s->partial, s->blocks);
    allocate(s->answer, 1);
#endif
    checked(cudaStreamSynchronize(0));
    return s.release();
}
void release(void* ptr) noexcept {
    Storage* s = static_cast<Storage*>(ptr);
    cuda_detail::DeviceScope scope(s->device);
    if (scope.status == cudaSuccess)
        delete s;
}
CflStep reduce(void* ptr, WaveSpeeds speeds, double cfl, void* stream_handle) {
    Storage& s = *static_cast<Storage*>(ptr);
    cuda_detail::DeviceScope scope(s.device);
    checked(scope.status);
    Grid grid = {s.nodes, s.offsets, s.entries, s.volumes};
    CflStep result;
#ifdef AERO_WITH_KOKKOS_GPU
    device_work::check(device_work::aero_device_wave_step(grid, speeds, cfl, &result, stream_handle));
#else
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    AERO_LAUNCH(nodal_min, s.blocks, block_threads, stream, grid, speeds, cfl, s.partial);
    checked(cudaGetLastError());
    AERO_LAUNCH(global_min, 1, block_threads, stream, s.partial, s.blocks, s.answer);
    checked(cudaGetLastError());
    checked(cudaMemcpyAsync(&result, s.answer, sizeof(result), cudaMemcpyDeviceToHost, stream));
    checked(cudaStreamSynchronize(stream));
#endif
    return result;
}
}
}
