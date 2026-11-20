#include "explicit_stage.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#else
#include "explicit_cuda.cuh"
#endif

namespace aero {
namespace explicit_detail {
// Запускает обновление компонент поля по одним и тем же коэффициентам стадии Эйлера или SSPRK3.
void launch_update(FieldStageUpdate update, std::size_t entries, int* invalid, void* stream) {
#ifdef AERO_WITH_KOKKOS_GPU
    device_work::check(device_work::aero_device_stage(update, entries, invalid, stream));
#else
    using namespace aero::explicit_cuda_kernels;
    const std::size_t blocks = (entries + 255) / 256;
    AERO_LAUNCH(stage_kernel, static_cast<unsigned>(blocks > 65535 ? 65535 : blocks), 256,
                static_cast<cudaStream_t>(stream), update, entries, invalid);
#endif
}
}
}
