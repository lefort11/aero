#define AERO_EULERSOLVER_KERNEL_ONLY
#include "lax_friedrichs.h"
#include "roe.h"
#include "characteristic.h"
#include "eulersolver_cuda_tests.cuh"
#include <stdio.h>

namespace {
// This is a test/example of the caller's CUDA launch, NOT a grid loop in the solver.
using namespace aero::eulersolver_cuda_tests_kernels;
int report(cudaError_t code,char* error,unsigned capacity) {
    if (code!=cudaSuccess && error && capacity) snprintf(error,capacity,"CUDA: %s",cudaGetErrorString(code));
    return code==cudaSuccess?0:1;
}
}
extern "C" int euler_test_cuda_count(int* count,char* error,unsigned capacity) {
    *count=0; cudaError_t code=cudaGetDeviceCount(count);
    if (code==cudaErrorNoDevice || code==cudaErrorInsufficientDriver) { cudaGetLastError(); *count=0; return 0; }
    return report(code,error,capacity);
}
extern "C" int euler_test_cuda_flux(int device,unsigned mode,unsigned dimension,const double* left,const double* right,
    const double* aux_left,const double* aux_right,const double* direction,double gamma,
    aero::EulerFluxResult* result,int* status,char* error,unsigned capacity) {
    int previous=-1; cudaError_t code=cudaGetDevice(&previous);
    if (code==cudaSuccess) code=cudaSetDevice(device);
    if (code!=cudaSuccess) return report(code,error,capacity);
    Case input={}; input.mode=mode; input.dimension=dimension; input.gamma=gamma;
    for (unsigned k=0;k<dimension+2;++k) {
        input.left[k]=left[k]; input.right[k]=right[k];
        if (aux_left) input.aux_left[k]=aux_left[k];
        if (aux_right) input.aux_right[k]=aux_right[k];
    }
    for (unsigned k=0;k<dimension;++k) input.direction[k]=direction[k];
    Output output={123,*result,456,-1}; Case* di=nullptr; Output* dout=nullptr; cudaStream_t stream=nullptr;
    code=cudaMalloc(reinterpret_cast<void**>(&di),sizeof(Case));
    if (code==cudaSuccess) code=cudaMalloc(reinterpret_cast<void**>(&dout),sizeof(Output));
    if (code==cudaSuccess) code=cudaStreamCreate(&stream);
    if (code==cudaSuccess) code=cudaMemcpy(di,&input,sizeof(Case),cudaMemcpyHostToDevice);
    if (code==cudaSuccess) code=cudaMemcpy(dout,&output,sizeof(Output),cudaMemcpyHostToDevice);
    if (code==cudaSuccess) { AERO_LAUNCH(one_interface, 1, 1, stream, di,dout); code=cudaGetLastError(); }
    if (code==cudaSuccess) code=cudaStreamSynchronize(stream);
    if (code==cudaSuccess) code=cudaMemcpy(&output,dout,sizeof(Output),cudaMemcpyDeviceToHost);
    if (stream) cudaStreamDestroy(stream);
    cudaFree(di); cudaFree(dout);
    if (previous>=0) cudaSetDevice(previous);
    if (code!=cudaSuccess) return report(code,error,capacity);
    if (output.before!=123 || output.after!=456) { snprintf(error,capacity,"CUDA wrote beyond one flux result"); return 1; }
    *result=output.result; *status=output.status; return 0;
}
