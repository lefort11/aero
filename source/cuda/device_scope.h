#ifndef AERO_CUDA_DEVICE_SCOPE_H
#define AERO_CUDA_DEVICE_SCOPE_H
#include "runtime.h"

namespace aero { namespace cuda_detail {
class DeviceScope {
    int previous_=-1;
public:
    cudaError_t status=cudaSuccess;
    explicit DeviceScope(int device) noexcept {
        if (device<0) return;
        status=cudaGetDevice(&previous_);
        if (status==cudaSuccess && previous_!=device) status=cudaSetDevice(device);
    }
    ~DeviceScope() { if (previous_>=0) cudaSetDevice(previous_); }
    DeviceScope(const DeviceScope&)=delete;
    DeviceScope& operator=(const DeviceScope&)=delete;
};
}}
#endif
