#ifndef AERO_EXPLICIT_H
#define AERO_EXPLICIT_H

#include "../parallel/kernel_math.h"
#ifndef __CUDACC_RTC__
#include <functional>
#include <memory>
#include <vector>
#endif

namespace aero {
enum class ExplicitScheme { Euler, SSPRK3 };
enum class ExplicitBackend { Auto, Cpu, Cuda };

// Borrowed only during the residual or inspection callback. The CUDA stream and pointers
// belong to the selected device; cpu views have device=-1 and stream=nullptr.
struct ExplicitField {
    size_t nodes, components;
    const double* values;   // [node*components + component]
    const double* volumes;  // one fixed, positive measure per computational node
    ExplicitBackend backend;
    int device;
    void* stream;
};

// Overwrite every entry of R with the sum of outward, area-integrated fluxes.
// Throw to reject a stage. CUDA callbacks enqueue their work on field.stream.
#ifndef __CUDACC_RTC__
typedef std::function<void(double time,const ExplicitField& field,double* R)> ExplicitResidual;

class ExplicitIntegrator {
public:
    // Both input arrays reside on the host and are copied at construction.
    ExplicitIntegrator(std::size_t nodes,std::size_t components,const double* volumes,
                       const double* initial,ExplicitBackend backend=ExplicitBackend::Auto,int device=0);
    ~ExplicitIntegrator();
    ExplicitIntegrator(const ExplicitIntegrator&)=delete;
    ExplicitIntegrator& operator=(const ExplicitIntegrator&)=delete;

    // The caller owns time and dt. A successful call commits the entire step.
    void step(ExplicitScheme scheme,double time,double dt,const ExplicitResidual& residual);
    ExplicitBackend backend() const;
    // Inspect the accepted field between steps, with the selected CUDA device active.
    void inspect(const std::function<void(const ExplicitField&)>& reader) const;
    std::vector<double> state() const; // host snapshot of the last committed step
    static int cuda_device_count();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif
}
#endif
