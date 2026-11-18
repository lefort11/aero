#include "cv.h"
#include "cv_cuda_tests.cuh"
#include <cstdio>
#include <cmath>
#include <stdexcept>

namespace {
void checked(cudaError_t s) {
    if (s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
template<class T> struct Buffer {
    T* p;
    explicit Buffer(size_t n) : p(nullptr) { checked(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    void upload(const T* src,size_t n) { checked(cudaMemcpy(p,src,n*sizeof(T),cudaMemcpyHostToDevice)); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};

// Exercise all three borrowed arrays by integrating x*n_x over the outer
// surface of the axis-aligned unit cube in the mixed mesh (nodes 4..11).
using namespace aero::cv_cuda_tests_kernels;
}
int check_cv_devices(aero::CvView host,bool required) {
    int count=0;
    cudaError_t status=cudaGetDeviceCount(&count);
    if (status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !count)) {
        std::puts("No CUDA device available.");
        return required ? 1 : 0;
    }
    checked(status);
    double reference=0;
    for (size_t i=0;i<host.edge_count;++i) reference+=std::fabs(host.edge_areas[i].x);
    for (int dev=0;dev<count;++dev) {
        checked(cudaSetDevice(dev));
        Buffer<double> volumes(host.node_count), output(3);
        Buffer<aero::CvVector> areas(host.edge_count);
        Buffer<aero::CvBoundary> boundary(host.boundary_count);
        volumes.upload(host.volumes,host.node_count);
        areas.upload(host.edge_areas,host.edge_count);
        boundary.upload(host.boundary,host.boundary_count);
        aero::CvView view=host;
        view.volumes=volumes.p;
        view.edge_areas=areas.p;
        view.boundary=boundary.p;
        AERO_LAUNCH(inspect, 1, 1, nullptr, view,output.p);
        checked(cudaGetLastError());
        double result[3];
        checked(cudaMemcpy(result,output.p,sizeof(result),cudaMemcpyDeviceToHost));
        if (!(std::fabs(result[0]-1)<1e-12 && std::fabs(result[1]-1)<1e-12 &&
              std::fabs(result[2]-reference)<1e-12)) throw std::runtime_error("CUDA median dual mismatch");
        std::printf("CUDA median dual test passed on device %d.\n",dev);
    }
    return 0;
}
