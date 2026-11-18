#include "runtime.h"
#include <windows.h>
#include <mutex>
#include <vector>

namespace {
// cuda.h maps versioned entry points (e.g. cuMemAlloc_v2) to their ABI names.
#define AERO_STRING_(x) #x
#define AERO_STRING(x) AERO_STRING_(x)
#define AERO_DRIVER_API(X) \
    X(cuInit) X(cuGetErrorString) X(cuDeviceGetCount) X(cuDevicePrimaryCtxRetain) \
    X(cuDevicePrimaryCtxRelease) X(cuCtxSetCurrent) X(cuCtxGetCurrent) X(cuCtxGetDevice) \
    X(cuCtxSynchronize) X(cuMemAlloc) X(cuMemFree) X(cuMemcpyHtoD) X(cuMemcpyDtoH) \
    X(cuMemcpyDtoD) X(cuMemcpyHtoDAsync) X(cuMemcpyDtoHAsync) X(cuMemcpyDtoDAsync) \
    X(cuMemsetD8Async) X(cuStreamCreate) X(cuStreamDestroy) X(cuStreamSynchronize) \
    X(cuLibraryLoadData) X(cuLibraryUnload) X(cuLibraryGetKernel) X(cuKernelGetFunction) X(cuLaunchKernel)

struct Driver {
    HMODULE dll=LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    CUresult status=CUDA_ERROR_NOT_INITIALIZED;
    std::mutex lock;
    std::vector<CUcontext> primary;
#define AERO_DECLARE(name) decltype(&::name) name=nullptr;
    AERO_DRIVER_API(AERO_DECLARE)
#undef AERO_DECLARE
    Driver() {
        if (!dll) return;
#define AERO_BIND(name) name=reinterpret_cast<decltype(name)>(reinterpret_cast<void(*)()>(GetProcAddress(dll,AERO_STRING(name)))); if (!name) return;
        AERO_DRIVER_API(AERO_BIND)
#undef AERO_BIND
        status=cuInit(0);
        if (status==CUDA_ERROR_SYSTEM_DRIVER_MISMATCH) status=CUDA_ERROR_NOT_INITIALIZED;
        int count=0;
        if (status==CUDA_SUCCESS) status=cuDeviceGetCount(&count);
        if (status==CUDA_SUCCESS) primary.resize(static_cast<std::size_t>(count));
    }
    ~Driver() {
        for (std::size_t i=0;i<primary.size();++i)
            if (primary[i]) cuDevicePrimaryCtxRelease(static_cast<CUdevice>(i));
        if (dll) FreeLibrary(dll);
    }
    CUresult select(int ordinal) {
        if (status!=CUDA_SUCCESS) return status;
        if (ordinal<0 || static_cast<std::size_t>(ordinal)>=primary.size()) return CUDA_ERROR_INVALID_DEVICE;
        std::lock_guard<std::mutex> guard(lock);
        CUcontext& ctx=primary[static_cast<std::size_t>(ordinal)];
        if (!ctx) {
            const CUresult result=cuDevicePrimaryCtxRetain(&ctx,ordinal);
            if (result!=CUDA_SUCCESS) return result;
        }
        return cuCtxSetCurrent(ctx);
    }
    CUresult current() {
        if (status!=CUDA_SUCCESS) return status;
        CUcontext ctx=nullptr;
        const CUresult result=cuCtxGetCurrent(&ctx);
        return result==CUDA_SUCCESS && !ctx ? select(0) : result;
    }
};
Driver& driver() {static Driver instance; return instance;}
thread_local CUresult last_launch=CUDA_SUCCESS;
#define AERO_CURRENT() Driver& d=driver(); const auto current=d.current(); if (current!=CUDA_SUCCESS) return current
}

cudaError_t cudaGetDeviceCount(int* count) {
    Driver& d=driver();
    *count=static_cast<int>(d.primary.size());
    return d.status;
}
cudaError_t cudaSetDevice(int device) {return driver().select(device);}
cudaError_t cudaGetDevice(int* device) {AERO_CURRENT(); return d.cuCtxGetDevice(device);}
const char* cudaGetErrorString(cudaError_t error) {
    const char* message=nullptr;
    const auto describe=driver().cuGetErrorString;
    if (describe && describe(error,&message)==CUDA_SUCCESS) return message;
    return "CUDA driver unavailable";
}
cudaError_t cudaGetLastError() {
    const auto result=last_launch;
    last_launch=CUDA_SUCCESS;
    return result;
}
cudaError_t cudaMalloc(void** p,std::size_t bytes) {
    AERO_CURRENT();
    CUdeviceptr address=0;
    const auto result=d.cuMemAlloc(&address,bytes);
    *p=reinterpret_cast<void*>(address);
    return result;
}
cudaError_t cudaFree(void* p) {
    if (!p) return CUDA_SUCCESS;
    AERO_CURRENT(); return d.cuMemFree(reinterpret_cast<CUdeviceptr>(p));
}
cudaError_t cudaMemcpy(void* dst,const void* src,std::size_t size,cudaMemcpyKind kind) {
    AERO_CURRENT();
    switch (kind) {
    case cudaMemcpyHostToDevice: return d.cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst),src,size);
    case cudaMemcpyDeviceToHost: return d.cuMemcpyDtoH(dst,reinterpret_cast<CUdeviceptr>(src),size);
    case cudaMemcpyDeviceToDevice: return d.cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst),reinterpret_cast<CUdeviceptr>(src),size);
    }
    return CUDA_ERROR_INVALID_VALUE;
}
cudaError_t cudaMemcpyAsync(void* dst,const void* src,std::size_t size,cudaMemcpyKind kind,cudaStream_t stream) {
    AERO_CURRENT();
    switch (kind) {
    case cudaMemcpyHostToDevice: return d.cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst),src,size,stream);
    case cudaMemcpyDeviceToHost: return d.cuMemcpyDtoHAsync(dst,reinterpret_cast<CUdeviceptr>(src),size,stream);
    case cudaMemcpyDeviceToDevice: return d.cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst),reinterpret_cast<CUdeviceptr>(src),size,stream);
    }
    return CUDA_ERROR_INVALID_VALUE;
}
cudaError_t cudaMemsetAsync(void* p,int value,std::size_t size,cudaStream_t stream) {
    AERO_CURRENT(); return d.cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(p),static_cast<unsigned char>(value),size,stream);
}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream,unsigned flags) {AERO_CURRENT(); return d.cuStreamCreate(stream,flags);}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {AERO_CURRENT(); return d.cuStreamDestroy(stream);}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {AERO_CURRENT(); return d.cuStreamSynchronize(stream);}
cudaError_t cudaDeviceSynchronize() {AERO_CURRENT(); return d.cuCtxSynchronize();}

namespace aero { namespace cuda_detail {
Image::Image(const unsigned char* bytes):library(nullptr),status(driver().current()) {
    if (status==CUDA_SUCCESS)
        status=driver().cuLibraryLoadData(&library,bytes,nullptr,nullptr,0,nullptr,nullptr,0);
}
Image::~Image() {if (library) driver().cuLibraryUnload(library);}
Kernel::Kernel(const Image& image,const char* name):kernel(nullptr),status(image.status) {
    if (status==CUDA_SUCCESS) status=driver().cuLibraryGetKernel(&kernel,image.library,name);
}
void Kernel::launch(unsigned blocks,unsigned threads,CUstream stream,void** args) const {
    CUfunction function=nullptr;
    CUresult result=status;
    if (result==CUDA_SUCCESS) result=driver().cuKernelGetFunction(&function,kernel);
    if (result==CUDA_SUCCESS) result=driver().cuLaunchKernel(function,blocks,1,1,threads,1,1,0,stream,args,nullptr);
    last_launch=result;
}
}}
