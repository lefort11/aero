#ifndef AERO_GPU_CHECK_H
#define AERO_GPU_CHECK_H

#include "device_memory.h"
#include <stdexcept>
#include <string>

namespace aero {
namespace gpu {
// Вызывается на CPU после обращения к CUDA. При ошибке сохраняет текст драйвера и контекст операции.
inline void checked(aero::gpu::Status code, const char* context = "") {
    if (code != aero::gpu::Status::Success)
        throw std::runtime_error(std::string(context) + aero::gpu::message(code));
}
}
}
#endif
