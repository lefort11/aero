#ifndef AERO_PARALLEL_BACKEND_H
#define AERO_PARALLEL_BACKEND_H
namespace aero {
#ifdef AERO_WITH_HIP
constexpr const char* gpu_backend_name = "hip";
constexpr const char* gpu_backend_label = "HIP";
#else
constexpr const char* gpu_backend_name = "cuda";
constexpr const char* gpu_backend_label = "CUDA";
#endif
}
#endif
