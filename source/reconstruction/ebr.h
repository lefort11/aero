#ifndef AERO_EBR_H
#define AERO_EBR_H
#include "../parallel/kernel_math.h"
namespace aero {
// Восстанавливает вектор для одной стороны подготовленного ребра по EBR3, EBR5 или первому порядку.
// Коэффициенты геометрии вычисляются на CPU, формула применяется и на CPU, и на CUDA.
template <class SideInput>
AERO_HOST_DEVICE inline void reconstruct_ebr_increment2(const SideInput& input, unsigned long long components,
                                                        double* output,
                                                        unsigned long long output_stride = 1) {
    for (unsigned long long c = 0; c < components; ++c)
        output[c * output_stride] = input.linear(c);
}
#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
class Mesh;
class ReconstructionStencil;
struct ReconstructionOptions;
// Готовит только EBR3 и EBR5. Общая точка выбора схемы, включая первый порядок, — prepare_reconstruction.
ReconstructionStencil prepare_ebr(const Mesh& mesh, const ReconstructionOptions& options);
#endif
} // namespace aero
#endif
