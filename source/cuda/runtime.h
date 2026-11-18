#ifndef AERO_CUDA_RUNTIME_H
#define AERO_CUDA_RUNTIME_H

#if defined(__CUDACC_RTC__)
#define AERO_GPU_KERNEL extern "C" __global__
#elif !defined(AERO_MINGW_CUDA_HOST)
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif
#define AERO_GPU_KERNEL __global__
#define AERO_LAUNCH(name, blocks, threads, stream, ...) name<<<blocks, threads, 0, stream>>>(__VA_ARGS__)
#else
#include <cuda.h>
#include <cstddef>

// The MinGW host uses the driver's primary contexts and the same streams as CUDA.
using cudaError_t = CUresult;
using cudaStream_t = CUstream;
constexpr auto cudaSuccess = CUDA_SUCCESS;
constexpr auto cudaErrorNoDevice = CUDA_ERROR_NO_DEVICE;
constexpr auto cudaErrorInsufficientDriver = CUDA_ERROR_NOT_INITIALIZED;
constexpr auto cudaErrorMemoryAllocation = CUDA_ERROR_OUT_OF_MEMORY;
constexpr auto cudaErrorInvalidValue = CUDA_ERROR_INVALID_VALUE;
constexpr auto cudaErrorNotSupported = CUDA_ERROR_NOT_SUPPORTED;
constexpr unsigned cudaStreamNonBlocking = CU_STREAM_NON_BLOCKING;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };

cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaMemGetInfo(std::size_t*,std::size_t*);
cudaError_t cudaDeviceGetPCIBusId(char*,int,int);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**,std::size_t);
template<class T> cudaError_t cudaMalloc(T** out,std::size_t size) {
    void* memory=nullptr;
    const auto status=cudaMalloc(&memory,size);
    *out=static_cast<T*>(memory);
    return status;
}
cudaError_t cudaFree(void*);
cudaError_t cudaMallocHost(void**,std::size_t);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaHostGetDevicePointer(void**,void*,unsigned);
cudaError_t cudaMemcpy(void*,const void*,std::size_t,cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
cudaError_t cudaMemsetAsync(void*,int,std::size_t,cudaStream_t);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*,unsigned);
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {return cudaStreamCreateWithFlags(stream,0);}
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaDeviceSynchronize();

namespace aero { namespace cuda_detail {
struct Image {
    explicit Image(const unsigned char* bytes);
    ~Image();
    Image(const Image&)=delete;
    Image& operator=(const Image&)=delete;
    CUlibrary library;
    CUresult status;
};
struct Kernel {
    Kernel(const Image&,const char* name);
    void launch(unsigned blocks,unsigned threads,CUstream stream,void** args) const;
    CUkernel kernel;
    CUresult status;
};
template<class... Args>
void launch(const Kernel& kernel,unsigned blocks,unsigned threads,CUstream stream,Args... args) {
    void* params[]={static_cast<void*>(&args)...};
    kernel.launch(blocks,threads,stream,params);
}
}}

#ifdef AERO_CUDA_IMAGE
#define AERO_IMAGE_NAME_(x) #x
#define AERO_IMAGE_NAME(x) AERO_IMAGE_NAME_(x)
asm(".section .rdata,\"dr\"\n.balign 8\n"
    AERO_IMAGE_NAME(AERO_CUDA_IMAGE) ":\n.incbin \"" AERO_CUDA_IMAGE_PATH "\"\n.text");
extern "C" const unsigned char AERO_CUDA_IMAGE[];
namespace {
inline const aero::cuda_detail::Image& cuda_image() {
    static const aero::cuda_detail::Image image(AERO_CUDA_IMAGE);
    return image;
}
}
#define AERO_LAUNCH(name, blocks, threads, stream, ...) do { \
    static const aero::cuda_detail::Kernel kernel(cuda_image(),#name); \
    aero::cuda_detail::launch(kernel,blocks,threads,stream,__VA_ARGS__); \
} while(false)
#endif
#endif
#endif
