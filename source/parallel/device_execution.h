#ifndef AERO_DEVICE_EXECUTION_H
#define AERO_DEVICE_EXECUTION_H
#include <Kokkos_Core.hpp>
#include "device_scope.h"
#include "row_loop.h"
#include <stdexcept>
#include <string>
namespace aero {
namespace device_work {
using Execution = Kokkos::DefaultExecutionSpace;
// Возвращает контекст Kokkos для выбранного устройства и потока устройства.
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

// Точки входа GPU используют те же циклы, что и численные модули на хосте.
using row_loop::for_indices;
template<unsigned min_blocks = 0> using DeviceRange = row_loop::Range<Execution,min_blocks>;

// Выполняет по одной операции на строку на выбранной карте в её стандартном потоке.
// Тип операции известен при компиляции. Параметры передаются в существующее ядро Kokkos.
template<class Work>
const char* run_rows(int device, const char* name, MeshIndex count, const Work& work) {
    return on_device(nullptr, [&](const Execution& exec) {
        for_indices(exec, name, count, work);
    }, device);
}

}
}
#endif
