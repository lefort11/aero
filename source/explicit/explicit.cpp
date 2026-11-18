#include "explicit.h"
#include "explicit_stage.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef AERO_WITH_CUDA
#include "../cuda/runtime.h"
#endif

namespace aero {
namespace {
#ifdef AERO_WITH_CUDA
void checked(cudaError_t code) {
    if (code!=cudaSuccess) throw std::runtime_error(std::string("explicit CUDA: ")+cudaGetErrorString(code));
}
class DeviceScope {
    int previous_;
public:
    explicit DeviceScope(int device):previous_(-1) {
        if (device>=0) {
            checked(cudaGetDevice(&previous_));
            if (device!=previous_) checked(cudaSetDevice(device));
        }
    }
    ~DeviceScope() { if (previous_>=0) cudaSetDevice(previous_); }
};
#endif
}

struct ExplicitIntegrator::Impl {
    std::size_t nodes,components,size;
    int device;
    ExplicitBackend backend;
    std::vector<double> cpu;
    double *volumes,*original,*stage,*residual;
    bool stepping;
#ifdef AERO_WITH_CUDA
    double* allocation;
    int* invalid;
    cudaStream_t stream;
#endif
    Impl():nodes(0),components(0),size(0),device(-1),backend(ExplicitBackend::Cpu),
           volumes(nullptr),original(nullptr),stage(nullptr),residual(nullptr),stepping(false)
#ifdef AERO_WITH_CUDA
           ,allocation(nullptr),invalid(nullptr),stream(nullptr)
#endif
    {}
    ~Impl() {
#ifdef AERO_WITH_CUDA
        if (device<0) return;
        int previous=-1;
        cudaGetDevice(&previous);
        if (cudaSetDevice(device)==cudaSuccess) {
            if (stream) cudaStreamSynchronize(stream);
            if (allocation) cudaFree(allocation);
            if (invalid) cudaFree(invalid);
            if (stream) cudaStreamDestroy(stream);
        }
        if (previous>=0 && previous!=device) cudaSetDevice(previous);
#endif
    }
    void bind(double* storage) {
        volumes=storage;
        original=volumes+nodes;
        stage=original+size;
        residual=stage+size;
    }
    void* stream_handle() const {
#ifdef AERO_WITH_CUDA
        return stream;
#else
        return nullptr;
#endif
    }
    void update_stage(explicit_detail::Stage s,const double* current,double dt) {
        int bad=0;
#ifdef AERO_WITH_CUDA
        if (backend==ExplicitBackend::Cuda) {
            checked(cudaMemsetAsync(invalid,0,sizeof(int),stream));
            explicit_detail::launch_update(s,size,components,volumes,original,current,residual,dt,stage,invalid,stream);
            checked(cudaGetLastError());
            checked(cudaMemcpyAsync(&bad,invalid,sizeof(int),cudaMemcpyDeviceToHost,stream));
            checked(cudaStreamSynchronize(stream));
        } else
#endif
        {
            const std::ptrdiff_t count=static_cast<std::ptrdiff_t>(size);
#ifdef _OPENMP
#pragma omp parallel for if(count>=4096) reduction(|:bad)
#endif
            for (std::ptrdiff_t k=0;k<count;++k) {
                stage[k]=explicit_detail::update(s,original[k],current[k],dt,volumes[k/components],residual[k]);
                if (!explicit_detail::finite_value(stage[k])) bad=1;
            }
        }
        if (bad) throw std::domain_error("non-finite explicit stage");
    }
};

int ExplicitIntegrator::cuda_device_count() {
#ifdef AERO_WITH_CUDA
    int count=0;
    const cudaError_t code=cudaGetDeviceCount(&count);
    if (code==cudaErrorNoDevice || code==cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return 0;
    }
    checked(code);
    return count;
#else
    return 0;
#endif
}

ExplicitIntegrator::ExplicitIntegrator(std::size_t nodes,std::size_t components,const double* volumes,
    const double* initial,ExplicitBackend backend,int device):impl_(new Impl) {
    const std::size_t limit=static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
    if (!nodes || !components || !volumes || !initial || nodes>limit ||
        components>(limit-nodes)/nodes/3)
        throw std::invalid_argument("invalid explicit field dimensions or storage");
    if (device<0 || (backend!=ExplicitBackend::Auto && backend!=ExplicitBackend::Cpu && backend!=ExplicitBackend::Cuda))
        throw std::invalid_argument("invalid explicit backend or device");
    Impl& p=*impl_;
    p.nodes=nodes; p.components=components; p.size=nodes*components;
    for (std::size_t n=0;n<nodes;++n)
        if (!(volumes[n]>0) || !std::isfinite(volumes[n])) throw std::invalid_argument("invalid control volume");
    for (std::size_t k=0;k<p.size;++k)
        if (!std::isfinite(initial[k])) throw std::invalid_argument("non-finite initial state");
    const int devices=backend==ExplicitBackend::Cpu?0:cuda_device_count();
    if (backend==ExplicitBackend::Cuda || (backend==ExplicitBackend::Auto && devices>0)) {
        if (device>=devices) throw std::invalid_argument("requested CUDA device unavailable");
#ifdef AERO_WITH_CUDA
        DeviceScope scope(device);
        p.backend=ExplicitBackend::Cuda;
        p.device=device;
        checked(cudaStreamCreateWithFlags(&p.stream,cudaStreamNonBlocking));
        checked(cudaMalloc(reinterpret_cast<void**>(&p.allocation),(nodes+3*p.size)*sizeof(double)));
        checked(cudaMalloc(reinterpret_cast<void**>(&p.invalid),sizeof(int)));
        p.bind(p.allocation);
        checked(cudaMemcpyAsync(p.volumes,volumes,nodes*sizeof(double),cudaMemcpyHostToDevice,p.stream));
        checked(cudaMemcpyAsync(p.original,initial,p.size*sizeof(double),cudaMemcpyHostToDevice,p.stream));
        checked(cudaStreamSynchronize(p.stream));
#endif
    } else {
        p.cpu.resize(nodes+3*p.size);
        p.bind(p.cpu.data());
        std::copy(volumes,volumes+nodes,p.volumes);
        std::copy(initial,initial+p.size,p.original);
    }
}
ExplicitIntegrator::~ExplicitIntegrator() {}
ExplicitBackend ExplicitIntegrator::backend() const { return impl_->backend; }

void ExplicitIntegrator::inspect(const std::function<void(const ExplicitField&)>& reader) const {
    Impl& p=*impl_;
    if (!reader) throw std::invalid_argument("missing field reader");
    if (p.stepping) throw std::logic_error("field inspection requires an idle integrator");
#ifdef AERO_WITH_CUDA
    DeviceScope scope(p.device);
#endif
    p.stepping=true;
    try {
        const ExplicitField field={p.nodes,p.components,p.original,p.volumes,p.backend,p.device,p.stream_handle()};
        reader(field);
#ifdef AERO_WITH_CUDA
        if (p.stream) checked(cudaStreamSynchronize(p.stream));
#endif
    } catch (...) {
#ifdef AERO_WITH_CUDA
        if (p.stream) cudaStreamSynchronize(p.stream);
#endif
        p.stepping=false;
        throw;
    }
    p.stepping=false;
}

void ExplicitIntegrator::step(ExplicitScheme scheme,double time,double dt,const ExplicitResidual& assemble) {
    if (!assemble || !std::isfinite(time) || !(dt>0) || !std::isfinite(time+dt))
        throw std::invalid_argument("invalid explicit step time, dt or residual callback");
    explicit_detail::Stages stages;
    switch(scheme) {
    case ExplicitScheme::Euler: stages=explicit_detail::euler_stages(); break;
    case ExplicitScheme::SSPRK3: stages=explicit_detail::ssprk3_stages(); break;
    default: throw std::invalid_argument("unknown explicit scheme");
    }
    Impl& p=*impl_;
    if (p.stepping) throw std::logic_error("recursive explicit step");
#ifdef AERO_WITH_CUDA
    DeviceScope scope(p.device);
#endif
    p.stepping=true;
    try {
        const double* current=p.original;
        for (unsigned i=0;i<stages.count;++i) {
            const explicit_detail::Stage s=stages.data[i];
            ExplicitField field={p.nodes,p.components,current,p.volumes,p.backend,p.device,p.stream_handle()};
            assemble(time+s.time_fraction*dt,field,p.residual);
            p.update_stage(s,current,dt);
            current=p.stage;
        }
        std::swap(p.original,p.stage);
    } catch (...) {
#ifdef AERO_WITH_CUDA
        if (p.stream) cudaStreamSynchronize(p.stream);
#endif
        p.stepping=false;
        throw;
    }
    p.stepping=false;
}
std::vector<double> ExplicitIntegrator::state() const {
    const Impl& p=*impl_;
#ifdef AERO_WITH_CUDA
    if (p.backend==ExplicitBackend::Cuda) {
        DeviceScope scope(p.device);
        std::vector<double> out(p.size);
        checked(cudaMemcpyAsync(out.data(),p.original,p.size*sizeof(double),cudaMemcpyDeviceToHost,p.stream));
        checked(cudaStreamSynchronize(p.stream));
        return out;
    }
#endif
    return std::vector<double>(p.original,p.original+p.size);
}
}
