#ifndef AERO_EULER_CHARACTERISTIC_H
#define AERO_EULER_CHARACTERISTIC_H
#include "euler.h"
#if defined(__CUDACC__)
#define AERO_CHAR_HD __host__ __device__
#else
#define AERO_CHAR_HD
#endif
namespace aero {
namespace euler_detail {
// The caller supplies a validated basis and its unit normal.
AERO_CHAR_HD inline EulerStatus project_characteristic(const EulerRoeBasis& basis,
    const double* n,double gamma,double kinetic,const double* u,double* w) {
    const unsigned d=basis.dimension;
    double momentum[3]={}, result[5]={};
    for (unsigned k=0;k<d;++k) momentum[k]=u[k+1]-basis.velocity[k]*u[0];
    const double pressure=(gamma-1)*(u[d+1]-euler_detail::dot(d,basis.velocity,u+1)+kinetic*u[0]);
    const double sum=(pressure/basis.sound_speed)/basis.sound_speed;
    const double difference=euler_detail::dot(d,momentum,n)/basis.sound_speed;
    result[0]=0.5*(sum-difference);
    result[1]=u[0]-sum;
    result[d+1]=0.5*(sum+difference);
    for (unsigned t=0;t+1<d;++t)
        for (unsigned k=0;k<d;++k) result[2+t]+=basis.right[5*(k+1)+2+t]*momentum[k];
    for (unsigned c=0;c<d+2;++c) if (!euler_detail::finite(result[c])) return EulerStatus::NonFinite;
    for (unsigned c=0;c<d+2;++c) w[c]=result[c];
    return EulerStatus::Success;
}
} // namespace euler_detail
// Apply the left eigenvectors of a frozen Roe basis to a state or increment.
// direction and gamma must be those passed to euler_roe_basis.
AERO_CHAR_HD inline EulerStatus conservative_to_characteristic(const EulerRoeBasis& basis,
    const double* direction, double gamma, const double* u, double* w) {
    const unsigned d=basis.dimension;
    if (d<1 || d>3) return EulerStatus::InvalidDimension;
    if (!u || !w) return EulerStatus::NullPointer;
    if (!(gamma>1) || !euler_detail::finite(gamma)) return EulerStatus::InvalidGamma;
    if (!(basis.sound_speed>0) || !euler_detail::finite(basis.sound_speed)) return EulerStatus::InvalidRoeState;
    double n[3];
    const EulerStatus status=euler_unit_direction(d,direction,n);
    if (status!=EulerStatus::Success) return status;
    const double kinetic=0.5*euler_detail::dot(d,basis.velocity,basis.velocity);
    return euler_detail::project_characteristic(basis,n,gamma,kinetic,u,w);
}
// Independent linear inverse transform U=R*W, one vector per call.
// Build the frozen basis separately with euler_roe_basis(U_i,U_j,...).
// This can also transform increments: physical-state checks belong to solvers.
// No allocation or kernel launch; CPU/CUDA share this exact implementation.
AERO_CHAR_HD inline EulerStatus characteristic_to_conservative(const EulerRoeBasis& basis,
    const double* characteristic,double* conservative) {
    if (basis.dimension<1 || basis.dimension>3) return EulerStatus::InvalidDimension;
    if (!characteristic || !conservative) return EulerStatus::NullPointer;
    const unsigned count=basis.dimension+2; double result[5]={0,0,0,0,0};
    for (unsigned c=0;c<count;++c) {
        if (!euler_detail::finite(characteristic[c])) return EulerStatus::NonFinite;
        for (unsigned r=0;r<count;++r) {
            if (!euler_detail::finite(basis.right[5*r+c])) return EulerStatus::NonFinite;
            result[r]+=basis.right[5*r+c]*characteristic[c];
        }
    }
    for (unsigned r=0;r<count;++r) if (!euler_detail::finite(result[r])) return EulerStatus::NonFinite;
    for (unsigned r=0;r<count;++r) conservative[r]=result[r];
    return EulerStatus::Success;
}
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
std::vector<double> characteristic_to_conservative(const EulerRoeBasis& basis,const std::vector<double>& characteristic);
#endif
} // namespace aero
#undef AERO_CHAR_HD
#endif
