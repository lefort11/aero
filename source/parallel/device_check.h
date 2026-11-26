#ifndef AERO_GPU_CHECK_H
#define AERO_GPU_CHECK_H

#include "device_memory.h"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace aero {
namespace gpu {
// Переводит ошибку драйвера в статус интерфейса C. Текст пишем только при ошибке
// и учитываем вместимость вызывающей стороны, включая завершающий нулевой байт.
inline int report_status(Status code, char* error, unsigned capacity) {
    if (code != Status::Success && error && capacity)
        std::snprintf(error, capacity, "CUDA: %s", message(code));
    return code == Status::Success ? 0 : 1;
}

// Возвращает доступное число устройств. Отсутствие драйвера или карты означает
// пустой список. Остальные ошибки передаём вызывающему коду с текстом драйвера.
inline int available_devices(int* devices, char* error, unsigned capacity) {
    *devices = 0;
    const Status code = count(devices);
    if (code == Status::NoDevice || code == Status::Unavailable) {
        *devices = 0;
        return 0;
    }
    return report_status(code, error, capacity);
}

// Вызывается на CPU после обращения к CUDA. При ошибке сохраняет текст драйвера и контекст операции.
inline void checked(aero::gpu::Status code, const char* context = "") {
    if (code != aero::gpu::Status::Success)
        throw std::runtime_error(std::string(context) + aero::gpu::message(code));
}
}
}
#endif
