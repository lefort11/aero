#pragma once
#include "../parallel/parallel_for.h"
#include "partition.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_api.h"
#endif

namespace aero {
namespace implicit {
#ifdef AERO_WITH_KOKKOS_GPU
// Запускает предиктор истории через типизированную точку входа GPU.
inline const char* launch_rows(int device, MeshIndex count, const TimePrediction& work) {
    return device_work::aero_device_time_prediction(device, count, work);
}
// Запускает поправку Ньютона через типизированную точку входа GPU.
inline const char* launch_rows(int device, MeshIndex count, const NewtonUpdate& work) {
    return device_work::aero_device_newton_update(device, count, work);
}
// Запускает операцию над векторами BiCGSTAB через типизированную точку входа GPU.
inline const char* launch_rows(int device, MeshIndex count, const KrylovUpdate& work) {
    return device_work::aero_device_krylov_update(device, count, work);
}
// Совмещает построение газовой невязки с вычислением её нормы на устройстве.
inline const char* launch_norm(int device, MeshIndex count, const NewtonUpdate& work, double* result) {
    return device_work::aero_device_newton_residual(device, count, work, result);
}
// Совмещает поправку линейного решателя с вычислением нормы обновлённой невязки.
inline const char* launch_norm(int device, MeshIndex count, const KrylovUpdate& work, double* result) {
    return device_work::aero_device_krylov_update_norm(device, count, work, result);
}
#ifdef AERO_HAS_MULTIGRID
// Запускает перенос между сетками через типизированную точку входа GPU.
inline const char* launch_rows(int device, MeshIndex count, const multigrid::Transfer& work) {
    return device_work::aero_device_multigrid_transfer(device, count, work);
}
// Запускает обновление грубосеточного вектора через типизированную точку входа GPU.
inline const char* launch_rows(int device, MeshIndex count, const multigrid::VectorUpdate& work) {
    return device_work::aero_device_multigrid_update(device, count, work);
}
#endif
#endif

// Вычисляет локальную норму тем же ядром, которое обновляет строки, затем
// согласует ошибку и максимум между рангами. CPU-операция задаётся при компиляции.
template<class Work, class HostNorm>
double measure_rows(const Partition& partition, int device, MeshIndex count, const Work& work, HostNorm host_norm) {
    return partition.maximum_norm([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            double value;
            device_work::check(launch_norm(device, count, work, &value));
            return value;
        }
#else
        (void)device;
#endif
        host_parallel::ensure_runtime();
        return host_norm(host_parallel::Execution(), count, work);
    });
}

// Выбирает исполнителя до обхода строк. На CPU сохраняет последовательный путь малых задач
// и обработку исключений parallel_for. На GPU операция передаётся без промежуточного буфера.
template<class Work>
void apply_rows(int device, MeshIndex count, const Work& work) {
#ifdef AERO_WITH_KOKKOS_GPU
    if (device >= 0) {
        device_work::check(launch_rows(device, count, work));
        return;
    }
#else
    (void)device;
#endif
    parallel_for(count, work);
}
}
}
