#ifndef AERO_DEVICE_EXECUTION_H
#define AERO_DEVICE_EXECUTION_H
#include <Kokkos_Core.hpp>
#include "device_scope.h"
#include <stdexcept>
#include <string>
namespace aero {
namespace device_work {
using Execution = Kokkos::DefaultExecutionSpace;
// Возвращает контекст Kokkos для выбранного устройства и потока CUDA.
Execution current_execution(void* stream);
// Исключения библиотеки остаются внутри модуля. На хост уходит только текст ошибки.
template <class Work>
const char* on_device(void* stream, Work work, int requested_device = -1) {
    static thread_local std::string error;
    try {
        gpu::DeviceScope on_card(requested_device);
        if (on_card.status != gpu::Status::Success)
            throw std::runtime_error(gpu::message(on_card.status));
        work(current_execution(stream));
        return nullptr;
    } catch (const std::exception& e) {
        error = e.what();
        return error.c_str();
    }
}

// Одна итерация соответствует одному индексу. Kokkos выбирает сетку запуска.
// HintLightWeight помещает аргументы небольшого ядра непосредственно в его параметры.
template <unsigned min_blocks = 0, class Work>
void for_indices(const Execution& exec, const char* name, MeshIndex count, Work work) {
    using Policy = Kokkos::RangePolicy<Execution, Kokkos::IndexType<MeshIndex>, Kokkos::LaunchBounds<128, min_blocks>>;
    if (count)
        Kokkos::parallel_for(name, Kokkos::Experimental::require(Policy(exec, 0, count),
                             Kokkos::Experimental::WorkItemProperty::HintLightWeight), work);
}

}
}
#endif
