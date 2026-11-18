#include "bc_test_case.h"
#include "periodic.h"
#include "bc_cuda_tests.cuh"
#include <cstdio>
#include <stdexcept>
namespace aero {
namespace {
void checked(cudaError_t code) {
    if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> struct Buffer {
    T* ptr;
    explicit Buffer(size_t count):ptr(nullptr) { checked(cudaMalloc(&ptr,count*sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};

using namespace aero::bc_cuda_tests_kernels;
}
int check_bc_devices(const BcTestCase* cases,size_t count,bool required) {
    int devices=0;
    cudaError_t code=cudaGetDeviceCount(&devices);
    if (code==cudaErrorNoDevice || code==cudaErrorInsufficientDriver || (code==cudaSuccess && !devices)) {
        std::puts("No CUDA device available.");
        return required?1:0;
    }
    checked(code);
    for (int dev=0;dev<devices;++dev) {
        checked(cudaSetDevice(dev));
        Buffer<BcTestCase> input(count);
        Buffer<int> failure(1);
        int first=static_cast<int>(count);
        checked(cudaMemcpy(input.ptr,cases,count*sizeof(*cases),cudaMemcpyHostToDevice));
        checked(cudaMemcpy(failure.ptr,&first,sizeof(first),cudaMemcpyHostToDevice));
        AERO_LAUNCH(test_flux, static_cast<unsigned>((count+63)/64), 64, nullptr, input.ptr,count,failure.ptr);
        checked(cudaGetLastError());
        checked(cudaMemcpy(&first,failure.ptr,sizeof(first),cudaMemcpyDeviceToHost));
        if (first!=static_cast<int>(count)) {
            std::fprintf(stderr,"CUDA boundary case %d failed on GPU %d\n",first,dev);
            return 1;
        }
        std::printf("%u boundary flux cases passed on GPU %d.\n",static_cast<unsigned>(count),dev);
    }
    return 0;
}
}
void periodic_gpu(int device,aero::PeriodicView view,aero::ReconstructionStencilView stencil,
                  const double* field,const double* expected) {
    using namespace aero;
    checked(cudaSetDevice(device));
    Buffer<PeriodicEdge> edges(view.edge_count);
    Buffer<double> q(view.node_count),out(2*view.edge_count);
    checked(cudaMemcpy(q.ptr,field,view.node_count*sizeof(double),cudaMemcpyHostToDevice));
    checked(cudaMemcpy(edges.ptr,view.edges,view.edge_count*sizeof(PeriodicEdge),cudaMemcpyHostToDevice));
    AERO_LAUNCH(periodic_rows, static_cast<unsigned>((2*view.edge_count+63)/64), 64, nullptr, stencil,edges.ptr,q.ptr,out.ptr);
    checked(cudaGetLastError());
    std::vector<double> result(2*view.edge_count);
    checked(cudaMemcpy(result.data(),out.ptr,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
    for (size_t i=0;i<result.size();++i)
        if (!(std::fabs(result[i]-expected[i])<1e-11)) throw std::runtime_error("periodic GPU row mismatch");
}
