#ifndef AERO_CUDA_CHECK_H
#define AERO_CUDA_CHECK_H

#include "runtime.h"
#include <stdexcept>
#include <string>

namespace aero {
namespace cuda_detail {
// Вызывается на CPU после обращения к CUDA. При ошибке сохраняет текст драйвера и контекст операции.
inline void checked(cudaError_t code, const char* context = "") {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(context) + cudaGetErrorString(code));
}
}
}
#endif
