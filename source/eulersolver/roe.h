#ifndef AERO_EULER_ROE_H
#define AERO_EULER_ROE_H
#include "characteristic.h"
#if defined(__CUDACC__)
#define AERO_ROE_HD __host__ __device__
#else
#define AERO_ROE_HD
#endif
namespace aero {
namespace roe_detail {
// Classical Roe: F*=(F_L+F_R)/2 - R*|Lambda|*R^{-1}*(U_R-U_L)/2.
// No entropy fix, pressure/density floors or silent LF fallback.
AERO_ROE_HD inline EulerStatus combine(unsigned dimension,const double* left,const double* right,
    const double* fl,const double* fr,const double* direction,double gamma,double speed,EulerFluxResult* result) {
    EulerRoeBasis basis;
    EulerStatus status=euler_roe_basis(dimension,left,right,direction,gamma,&basis);
    if (status!=EulerStatus::Success) return status;
    const unsigned count=dimension+2;
    double jump[5]={},strength[5]={};
    for (unsigned k=0;k<count;++k) {
        jump[k]=right[k]-left[k];
        if (!euler_detail::finite(jump[k]) || !euler_detail::finite(fl[k]) || !euler_detail::finite(fr[k])) return EulerStatus::NonFinite;
    }
    status=conservative_to_characteristic(basis,direction,gamma,jump,strength);
    if (status!=EulerStatus::Success) return status;
    EulerFluxResult output={}; output.components=count; output.max_wave_speed=speed;
    for (unsigned c=0;c<count;++c) {
        if (!euler_detail::finite(strength[c])) return EulerStatus::NonFinite;
        const double wave=euler_detail::absolute(basis.eigenvalues[c]);
        if (wave>output.max_wave_speed) output.max_wave_speed=wave;
    }
    for (unsigned r=0;r<count;++r) {
        double dissipation=0;
        for (unsigned c=0;c<count;++c)
            dissipation+=basis.right[5*r+c]*euler_detail::absolute(basis.eigenvalues[c])*strength[c];
        output.flux[r]=0.5*fl[r]+0.5*fr[r]-0.5*dissipation;
        if (!euler_detail::finite(output.flux[r])) return EulerStatus::NonFinite;
    }
    *result=output; return EulerStatus::Success;
}
}

// One reconstructed pair, physical dimension 1/2/3. Direction is normalized;
// result is per unit area. Roe averages here use U_L/U_R, not original U_i/U_j.
// max_wave_speed is max(endpoint spectral bounds, absolute Roe eigenvalues).
AERO_ROE_HD inline EulerStatus roe_conservative(unsigned dimension,const double* left,const double* right,
    const double* direction,double gamma,EulerFluxResult* result) {
    if (!result) return EulerStatus::NullPointer;
    double fl[5],fr[5],al=0,ar=0;
    EulerStatus status=euler_flux(dimension,left,direction,gamma,fl,&al);
    if (status!=EulerStatus::Success) return status;
    status=euler_flux(dimension,right,direction,gamma,fr,&ar);
    if (status!=EulerStatus::Success) return status;
    return roe_detail::combine(dimension,left,right,fl,fr,direction,gamma,al>ar?al:ar,result);
}

// Supplied fluxes are projected on the unit direction, not LF split fluxes.
// States determine the Roe matrix and jump; supplied fluxes form the mean.
AERO_ROE_HD inline EulerStatus roe_fluxes(unsigned dimension,const double* fl,const double* fr,
    const double* left,const double* right,const double* direction,double gamma,EulerFluxResult* result) {
    if (!result || !fl || !fr) return EulerStatus::NullPointer;
    EulerState l,r; EulerStatus status=euler_state(dimension,left,gamma,&l);
    if (status!=EulerStatus::Success) return status;
    status=euler_state(dimension,right,gamma,&r);
    if (status!=EulerStatus::Success) return status;
    double n[3]={0,0,0}; status=euler_unit_direction(dimension,direction,n);
    if (status!=EulerStatus::Success) return status;
    const double al=euler_detail::absolute(euler_detail::dot(dimension,l.velocity,n))+l.sound_speed;
    const double ar=euler_detail::absolute(euler_detail::dot(dimension,r.velocity,n))+r.sound_speed;
    if (!euler_detail::finite(al) || !euler_detail::finite(ar)) return EulerStatus::NonFinite;
    return roe_detail::combine(dimension,left,right,fl,fr,direction,gamma,al>ar?al:ar,result);
}
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
EulerFluxResult roe_conservative(const std::vector<double>& left,const std::vector<double>& right,
    const std::vector<double>& direction,double gamma);
EulerFluxResult roe_fluxes(const std::vector<double>& fl,const std::vector<double>& fr,
    const std::vector<double>& left,const std::vector<double>& right,const std::vector<double>& direction,double gamma);
#endif
} // namespace aero
#undef AERO_ROE_HD
#endif
