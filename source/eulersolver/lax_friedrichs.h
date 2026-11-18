#ifndef AERO_LAX_FRIEDRICHS_H
#define AERO_LAX_FRIEDRICHS_H
#include "euler.h"
#if defined(__CUDACC__)
#define AERO_LF_HD __host__ __device__
#else
#define AERO_LF_HD
#endif

namespace aero {
namespace euler_detail {
AERO_LF_HD inline EulerStatus combine_lf(unsigned dimension,const double* left,const double* right,
    const double* flux_left,const double* flux_right,double alpha,EulerFluxResult* result) {
    EulerFluxResult value={}; value.components=dimension+2; value.max_wave_speed=alpha;
    for (unsigned k=0;k<dimension+2;++k) {
        if (!finite(flux_left[k]) || !finite(flux_right[k])) return EulerStatus::NonFinite;
        value.flux[k]=0.5*flux_left[k]+0.5*flux_right[k]-(0.5*alpha)*(right[k]-left[k]);
        if (!finite(value.flux[k])) return EulerStatus::NonFinite;
    }
    *result=value; return EulerStatus::Success;
}
}

// Local Lax-Friedrichs / Rusanov. One pair of states per call; no mesh loop,
// allocation, kernel launch, boundary condition or time integration.
// On any error result is unchanged. Arrays must contain dimension+2 entries;
// direction has dimension entries. It is normalized, area is applied by caller.
AERO_LF_HD inline EulerStatus lax_friedrichs_conservative(unsigned dimension,
    const double* left,const double* right,const double* direction,double gamma,EulerFluxResult* result) {
    if (!result) return EulerStatus::NullPointer;
    double fl[5],fr[5],al=0,ar=0;
    EulerStatus status=euler_flux(dimension,left,direction,gamma,fl,&al);
    if (status!=EulerStatus::Success) return status;
    status=euler_flux(dimension,right,direction,gamma,fr,&ar);
    if (status!=EulerStatus::Success) return status;
    return euler_detail::combine_lf(dimension,left,right,fl,fr,al>ar?al:ar,result);
}

// Inputs are reconstructed physical fluxes projected on the SAME UNIT direction
// (not LF-split F+ or F-). Reconstructed U_L/U_R are also needed for dissipation
// and the local speed. The supplied fluxes are not replaced by F(U_L/U_R).
AERO_LF_HD inline EulerStatus lax_friedrichs_fluxes(unsigned dimension,
    const double* flux_left,const double* flux_right,const double* state_left,const double* state_right,
    const double* direction,double gamma,EulerFluxResult* result) {
    if (!result || !flux_left || !flux_right) return EulerStatus::NullPointer;
    EulerState left,right; EulerStatus status=euler_state(dimension,state_left,gamma,&left);
    if (status!=EulerStatus::Success) return status;
    status=euler_state(dimension,state_right,gamma,&right);
    if (status!=EulerStatus::Success) return status;
    double n[3]={0,0,0}; status=euler_unit_direction(dimension,direction,n);
    if (status!=EulerStatus::Success) return status;
    const double al=euler_detail::absolute(euler_detail::dot(dimension,left.velocity,n))+left.sound_speed;
    const double ar=euler_detail::absolute(euler_detail::dot(dimension,right.velocity,n))+right.sound_speed;
    if (!euler_detail::finite(al) || !euler_detail::finite(ar)) return EulerStatus::NonFinite;
    return euler_detail::combine_lf(dimension,state_left,state_right,flux_left,flux_right,al>ar?al:ar,result);
}

} // namespace aero

#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
namespace aero {
// Checked CPU adapters. Dimension is direction.size(); no global field input.
EulerFluxResult lax_friedrichs_conservative(const std::vector<double>& left,const std::vector<double>& right,
    const std::vector<double>& direction,double gamma);
EulerFluxResult lax_friedrichs_fluxes(const std::vector<double>& fl,const std::vector<double>& fr,
    const std::vector<double>& ul,const std::vector<double>& ur,const std::vector<double>& direction,double gamma);

}
#endif
#undef AERO_LF_HD
#endif
