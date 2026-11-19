#include "explicit_stage.h"
#include "explicit_cuda.cuh"

namespace aero {
namespace explicit_detail {
namespace {
using namespace aero::explicit_cuda_kernels;
}
// Запускает обновление компонент поля по одним и тем же коэффициентам стадии Эйлера или SSPRK3.
void launch_update(FieldStageUpdate update, std::size_t entries, int* invalid, void* stream) {
    const std::size_t blocks = (entries + 255) / 256;
    AERO_LAUNCH(stage_kernel, static_cast<unsigned>(blocks > 65535 ? 65535 : blocks), 256,
                static_cast<cudaStream_t>(stream), update, entries, invalid);
}
}
}
