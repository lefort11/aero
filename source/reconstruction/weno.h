#ifndef AERO_WENO_H
#define AERO_WENO_H

// Shared scalar WENO3/WENO5 formula. No STL dependency in CUDA kernels.
#if defined(__CUDACC__)
#define AERO_WENO_HD __host__ __device__
#else
#define AERO_WENO_HD
#endif

namespace aero {
// Bakhvalov/Kozubskaya (2017), (9), JS weights; slopes are h times the
// divided differences. Scaling and ratio weights avoid squaring large fields
// or forming 1/(epsilon+beta)^2 explicitly. No WENO-Z modification is applied.
AERO_WENO_HD inline double weno_increment2(unsigned order,const double* slopes,double epsilon_scale) {
    const unsigned count=order==5?3:2, slope_count=order==5?4:2;
    double scale=epsilon_scale;
    for (unsigned k=0;k<slope_count;++k) {
        const double magnitude=slopes[k]<0?-slopes[k]:slopes[k];
        if (magnitude>scale) scale=magnitude;
    }
    if (scale==0) return 0;
    double d[4]={0,0,0,0};
    for (unsigned k=0;k<slope_count;++k) d[k]=slopes[k]/scale;
    double candidate[3]={d[0],d[1],0},beta[3]={d[0]*d[0],d[1]*d[1],0};
    double ideal[3]={2.0/3,1.0/3,0};
    if (order==5) {
        const double a=d[2]-d[0],b=d[2]-3*d[0];
        const double c=d[0]-d[1],e=d[0]+d[1];
        const double f=d[1]-d[3],g=3*d[1]-d[3];
        beta[0]=(13.0/12)*a*a+0.25*b*b;
        beta[1]=(13.0/12)*c*c+0.25*e*e;
        beta[2]=(13.0/12)*f*f+0.25*g*g;
        candidate[0]=(4*d[0]-d[2])/3;
        candidate[1]=(2*d[0]+d[1])/3;
        candidate[2]=(5*d[1]-2*d[3])/3;
        ideal[0]=0.3; ideal[1]=0.6; ideal[2]=0.1;
    }
    const double t=epsilon_scale/scale,epsilon=t*t;
    double denominator[3]={0,0,0};
    for (unsigned k=0;k<count;++k) denominator[k]=epsilon+beta[k];
    double minimum=denominator[0];
    for (unsigned k=1;k<count;++k) if (denominator[k]<minimum) minimum=denominator[k];
    double sum=0,result=0;
    for (unsigned k=0;k<count;++k) {
        // A zero after underflow is handled by the epsilon -> 0 limit.
        const double ratio=minimum==0?(denominator[k]==0?1.0:0.0):minimum/denominator[k];
        const double weight=ideal[k]*ratio*ratio;
        sum+=weight; result+=weight*candidate[k];
    }
    return scale*(result/sum);
}

// Actual WENO vector reconstruction of one prepared edge side. This body is
// compiled for both CPU and CUDA; it evaluates solution-dependent JS weights.
template<class SideInput>
AERO_WENO_HD inline void reconstruct_weno_increment2(const SideInput& input,
    unsigned long long components,double* output,unsigned long long output_stride=1) {
    const unsigned order=input.order();
    for (unsigned long long c=0;c<components;++c) {
        if (!order) { output[c*output_stride]=input.linear(c); continue; }
        double slopes[4]={0,0,0,0};
        for (unsigned k=0;k<(order==5?4u:2u);++k) slopes[k]=input.slope(k,c);
        output[c*output_stride]=weno_increment2(order,slopes,input.epsilon_scale());
    }
}

#if !defined(AERO_RECONSTRUCTION_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
class Mesh;
class ReconstructionStencil;
struct ReconstructionOptions;

// WENO geometry preparation; options.scheme must be Weno3 or Weno5.
// Include reconstruction.h for the shared options, stencil, result and executor types.
ReconstructionStencil prepare_weno(const Mesh& mesh, const ReconstructionOptions& options);
#endif
} // namespace aero
#undef AERO_WENO_HD
#endif

