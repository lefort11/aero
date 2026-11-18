#ifndef AERO_EXPLICIT_TEST_PROBLEM_H
#define AERO_EXPLICIT_TEST_PROBLEM_H
#include "explicit.h"
enum class TestProblem { Forced, Ring, Invalid };
#ifdef __CUDACC__
__host__ __device__
#endif
inline double test_residual(TestProblem problem,double t,aero::ExplicitField f,size_t k) {
    if (problem==TestProblem::Invalid) return ::nan("");
    const size_t node=k/f.components;
    if (problem==TestProblem::Forced) return -f.volumes[node]*(f.values[k]+t);
    const size_t upstream=node==0 ? f.nodes-1 : node-1;
    return f.values[k]-f.values[upstream*f.components+k%f.components];
}
#ifdef AERO_WITH_CUDA
int current_test_device();
void device_test_residual(TestProblem,double,aero::ExplicitField,double*);
#endif
#endif
