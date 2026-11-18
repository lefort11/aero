#ifndef AERO_EBR_H
#define AERO_EBR_H
#if defined(__CUDACC__)
#define AERO_EBR_HD __host__ __device__
#else
#define AERO_EBR_HD
#endif
namespace aero {
// EBR3/EBR5 and first-order vector reconstruction for ONE prepared edge side.
// All geometry coefficients are prepared on CPU; this body is shared by CPU/CUDA.
template<class SideInput>
AERO_EBR_HD inline void reconstruct_ebr_increment2(const SideInput& input,
    unsigned long long components,double* output,unsigned long long output_stride=1) {
    for (unsigned long long c=0;c<components;++c) output[c*output_stride]=input.linear(c);
}
#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
class Mesh;
class ReconstructionStencil;
struct ReconstructionOptions;
// EBR3/EBR5 only. Use prepare_reconstruction for dispatch and FirstOrder.
ReconstructionStencil prepare_ebr(const Mesh& mesh,const ReconstructionOptions& options);
#endif
} // namespace aero
#undef AERO_EBR_HD
#endif
