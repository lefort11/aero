#define AERO_RESIDUAL_KERNEL_ONLY
#define AERO_BC_KERNEL_ONLY
#define AERO_EULERSOLVER_KERNEL_ONLY
#define AERO_RECONSTRUCTION_KERNEL_ONLY
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#else
#include "partition_kernels.cuh"
#endif
#include <stdexcept>

namespace aero {
namespace partition_detail {
#ifndef AERO_WITH_KOKKOS_GPU
namespace {
// Ограничиваем размер сетки запуска: ядро с повторным шагом по индексам обработает оставшиеся узлы.
unsigned blocks(Index count) {
    const Index n = (count + 127) / 128;
    return static_cast<unsigned>(n > 65535 ? 65535 : n);
}
void launched() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
}
#endif
// Запускает обновление только собственных степеней свободы MPI-процесса.
void update(Grid g, explicit_detail::Stage stage, double dt, bool save_original) {
    if (!g.owned_count)
        return;
#ifdef AERO_WITH_KOKKOS_GPU
    device_work::check(device_work::aero_device_owned_stage(g, stage, dt, save_original));
#else
    AERO_LAUNCH(partition_update, blocks(g.owned_count), 128, nullptr, g, stage, dt, save_original);
    launched();
#endif
}
// Список индексов определяет порядок значений в сообщении соседнему процессу.
void transfer(Grid g, const MeshIndex* ids, Index count, double* records, bool edges, bool pack) {
    if (!count)
        return;
#ifdef AERO_WITH_KOKKOS_GPU
    device_work::check(device_work::aero_device_records(g, ids, count, records, edges, pack));
#else
    AERO_LAUNCH(partition_transfer, blocks(count), 128, nullptr, g, ids, count, records, edges, pack);
    launched();
#endif
}
}
}
