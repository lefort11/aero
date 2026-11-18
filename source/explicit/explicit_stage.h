#ifndef AERO_EXPLICIT_STAGE_H
#define AERO_EXPLICIT_STAGE_H

#include "../parallel/kernel_math.h"

namespace aero {
namespace explicit_detail {
struct Stage {
    double time_fraction, original_weight, forward_weight;
};
struct Stages { const Stage* data; unsigned count; };
Stages euler_stages();
Stages ssprk3_stages();

#ifdef __CUDACC__
#define AERO_EXPLICIT_HD __host__ __device__
#else
#define AERO_EXPLICIT_HD
#endif
AERO_EXPLICIT_HD inline bool finite_value(double x) { return x>=-max_finite_double && x<=max_finite_double; }
AERO_EXPLICIT_HD inline double update(Stage s,double original,double current,
                                      double dt,double volume,double residual) {
    const double forward=current-dt*(residual/volume);
    if (s.original_weight==0 || forward==original) return forward;
    return s.original_weight*original+s.forward_weight*forward;
}
#undef AERO_EXPLICIT_HD

#ifdef AERO_WITH_CUDA
void launch_update(Stage stage,size_t size,size_t components,const double* volumes,
                   const double* original,const double* current,const double* residual,
                   double dt,double* next,int* invalid,void* stream);
#endif
}
}
#endif
