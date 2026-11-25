#ifndef AERO_DEVICE_PLATFORM_H
#define AERO_DEVICE_PLATFORM_H
#include <Kokkos_Core.hpp>

// Native operations absent from the Kokkos 3.2 public execution/memory interfaces.
namespace aero { namespace gpu { namespace native {
#ifdef KOKKOS_ENABLE_HIP
using Error = hipError_t;
constexpr auto success = hipSuccess;
constexpr auto no_device = hipErrorNoDevice;
constexpr auto no_driver = hipErrorInsufficientDriver;
constexpr auto out_of_memory = hipErrorOutOfMemory;
constexpr auto invalid = hipErrorInvalidValue;
constexpr auto unsupported = hipErrorNotSupported;
constexpr auto error_string = hipGetErrorString;
constexpr auto clear_error = hipGetLastError;
constexpr auto get_device = hipGetDevice;
constexpr auto set_device = hipSetDevice;
constexpr auto device_count = hipGetDeviceCount;
constexpr auto pci_bus_id = hipDeviceGetPCIBusId;
constexpr auto memory_info = hipMemGetInfo;
using Stream = hipStream_t;
constexpr auto create_stream = hipStreamCreateWithFlags;
constexpr auto nonblocking = hipStreamNonBlocking;
constexpr auto blocking = hipStreamDefault;
constexpr auto destroy_stream = hipStreamDestroy;
inline Error mapped_pointer(void** alias, void* host, unsigned flags) {
    return hipHostGetDevicePointer(alias, host, flags);
}
using Pinned = Kokkos::Experimental::HIPHostPinnedSpace;
inline Stream default_stream() { return nullptr; }
using Properties = hipDeviceProp_t;
inline Error properties(Properties* result, int device) { return hipGetDeviceProperties(result, device); }
#elif defined(KOKKOS_ENABLE_CUDA)
using Error = cudaError_t;
constexpr auto success = cudaSuccess;
constexpr auto no_device = cudaErrorNoDevice;
constexpr auto no_driver = cudaErrorInsufficientDriver;
constexpr auto out_of_memory = cudaErrorMemoryAllocation;
constexpr auto invalid = cudaErrorInvalidValue;
constexpr auto unsupported = cudaErrorNotSupported;
constexpr auto error_string = cudaGetErrorString;
constexpr auto clear_error = cudaGetLastError;
constexpr auto get_device = cudaGetDevice;
constexpr auto set_device = cudaSetDevice;
constexpr auto device_count = cudaGetDeviceCount;
constexpr auto pci_bus_id = cudaDeviceGetPCIBusId;
constexpr auto memory_info = cudaMemGetInfo;
using Stream = cudaStream_t;
constexpr auto create_stream = cudaStreamCreateWithFlags;
constexpr auto nonblocking = cudaStreamNonBlocking;
constexpr auto blocking = cudaStreamDefault;
constexpr auto destroy_stream = cudaStreamDestroy;
inline Error mapped_pointer(void** alias, void* host, unsigned flags) {
    return cudaHostGetDevicePointer(alias, host, flags);
}
using Pinned = Kokkos::CudaHostPinnedSpace;
inline Stream default_stream() { return cudaStreamLegacy; }
using Properties = cudaDeviceProp;
inline Error properties(Properties* result, int device) { return cudaGetDeviceProperties(result, device); }
#endif
} } }
#endif
