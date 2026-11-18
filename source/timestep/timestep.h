#ifndef AERO_TIMESTEP_H
#define AERO_TIMESTEP_H
#include "../parallel/kernel_math.h"
#ifndef __CUDACC_RTC__
#include <memory>
#endif

namespace aero {
struct MeshView;
struct CvView;
struct PeriodicView;

struct WaveSpeeds {
    const double* edge;
    size_t edge_count;
    const double* boundary;
    size_t boundary_count;
};
struct CflStep { double dt; size_t node; };

// Prepare once from host geometry. device=-1 uses CPU; otherwise device
// is the CUDA ordinal. During compute the speeds must be on that backend.
#ifndef __CUDACC_RTC__
class CflTimestep {
public:
    CflTimestep(const MeshView&,const CvView&,int device=-1);
    explicit CflTimestep(const PeriodicView&,int device=-1);
    ~CflTimestep();
    CflTimestep(const CflTimestep&)=delete;
    CflTimestep& operator=(const CflTimestep&)=delete;
    // One global step for a fixed mesh. stream is the CUDA producer stream.
    // Ties return the smallest computational node ID. Use one context per concurrent call.
    CflStep compute(double cfl,WaveSpeeds speeds,void* stream=nullptr) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif
}
#endif
