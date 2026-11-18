#ifndef AERO_EULERSOLVER_EULER_H
#define AERO_EULERSOLVER_EULER_H

#include "../parallel/kernel_math.h"
#if defined(__CUDACC__)
#define AERO_EULER_HD __host__ __device__
#else
#define AERO_EULER_HD
#endif

namespace aero {
enum class EulerStatus {
    Success, InvalidDimension, NullPointer, InvalidGamma, InvalidDirection,
    InvalidDensity, InvalidPressure, NonFinite, InvalidRoeState
};

// U = [rho, rho*u_1, ..., rho*u_dimension, total energy per volume].
struct EulerFluxResult {
    unsigned components;
    double flux[5]; // active dimension+2 entries; inactive entries zero
    double max_wave_speed; // solver's signal-speed estimate; not area weighted
};

struct EulerState {
    double density, velocity[3], pressure, sound_speed, enthalpy;
};
struct EulerRoeBasis {
    unsigned dimension;
    double velocity[3], enthalpy, sound_speed;
    // Fixed row stride 5, active (dimension+2) x (dimension+2) block.
    // Columns: acoustic-, entropy, shear1[, shear2], acoustic+.
    double right[25], eigenvalues[5];
};

namespace euler_detail {
AERO_EULER_HD inline bool finite(double x) { return x==x && x<=max_finite_double && x>=-max_finite_double; }
AERO_EULER_HD inline double absolute(double x) { return x<0?-x:x; }
AERO_EULER_HD inline double dot(unsigned d,const double* a,const double* b) {
    double value=0; for (unsigned k=0;k<d;++k) value+=a[k]*b[k]; return value;
}
}

// Direction is normalized internally; its magnitude is NOT a face area.
// No output is changed on failure, throughout this API.
AERO_EULER_HD inline EulerStatus euler_unit_direction(unsigned dimension,const double* direction,double* unit) {
    if (dimension<1 || dimension>3) return EulerStatus::InvalidDimension;
    if (!direction || !unit) return EulerStatus::NullPointer;
    double scale=0,n[3]={0,0,0};
    for (unsigned k=0;k<dimension;++k) {
        if (!euler_detail::finite(direction[k])) return EulerStatus::NonFinite;
        const double a=euler_detail::absolute(direction[k]); if (a>scale) scale=a;
    }
    if (!(scale>0)) return EulerStatus::InvalidDirection;
    for (unsigned k=0;k<dimension;++k) n[k]=direction[k]/scale;
    const double length=::sqrt(euler_detail::dot(dimension,n,n));
    for (unsigned k=0;k<dimension;++k) unit[k]=n[k]/length;
    return EulerStatus::Success;
}

AERO_EULER_HD inline EulerStatus euler_state(unsigned dimension,const double* u,double gamma,EulerState* result) {
    if (dimension<1 || dimension>3) return EulerStatus::InvalidDimension;
    if (!u || !result) return EulerStatus::NullPointer;
    if (!euler_detail::finite(gamma) || !(gamma>1)) return EulerStatus::InvalidGamma;
    for (unsigned k=0;k<dimension+2;++k) if (!euler_detail::finite(u[k])) return EulerStatus::NonFinite;
    if (!(u[0]>0)) return EulerStatus::InvalidDensity;
    EulerState state={}; state.density=u[0]; double kinetic=0;
    for (unsigned k=0;k<dimension;++k) {
        state.velocity[k]=u[k+1]/u[0];
        kinetic+=0.5*u[k+1]*state.velocity[k];
        if (!euler_detail::finite(state.velocity[k])) return EulerStatus::NonFinite;
    }
    state.pressure=(gamma-1)*(u[dimension+1]-kinetic);
    if (!euler_detail::finite(state.pressure)) return EulerStatus::NonFinite;
    if (!(state.pressure>0)) return EulerStatus::InvalidPressure;
    const double sound2=gamma*(state.pressure/u[0]);
    state.sound_speed=::sqrt(sound2);
    state.enthalpy=u[dimension+1]/u[0]+state.pressure/u[0];
    if (!euler_detail::finite(state.sound_speed) || !euler_detail::finite(state.enthalpy)) return EulerStatus::NonFinite;
    if (!(state.sound_speed>0)) return EulerStatus::InvalidPressure;
    *result=state; return EulerStatus::Success;
}

// Physical Euler flux projected on the unit direction. Output has dimension+2 values.
AERO_EULER_HD inline EulerStatus euler_flux(unsigned dimension,const double* u,const double* direction,
                                     double gamma,double* flux,double* wave_speed) {
    if (!flux || !wave_speed) return EulerStatus::NullPointer;
    EulerState state; EulerStatus status=euler_state(dimension,u,gamma,&state);
    if (status!=EulerStatus::Success) return status;
    double n[3]={0,0,0}; status=euler_unit_direction(dimension,direction,n);
    if (status!=EulerStatus::Success) return status;
    const double un=euler_detail::dot(dimension,state.velocity,n);
    double f[5]={0,0,0,0,0}; f[0]=euler_detail::dot(dimension,u+1,n);
    for (unsigned k=0;k<dimension;++k) f[k+1]=u[k+1]*un+state.pressure*n[k];
    f[dimension+1]=u[dimension+1]*un+state.pressure*un;
    const double speed=euler_detail::absolute(un)+state.sound_speed;
    if (!euler_detail::finite(speed)) return EulerStatus::NonFinite;
    for (unsigned k=0;k<dimension+2;++k) if (!euler_detail::finite(f[k])) return EulerStatus::NonFinite;
    for (unsigned k=0;k<dimension+2;++k) flux[k]=f[k];
    *wave_speed=speed; return EulerStatus::Success;
}

// Common Roe basis of two supplied conservative states. For characteristic
// reconstruction use original U_i/U_j; a Roe flux uses reconstructed U_L/U_R.
// U=R*W. Tangents: 2D t=(-ny,nx); 3D choose the Cartesian axis least aligned
// with n, project/normalize it to t1, then t2=n cross t1 (ties: x,y,z).
namespace euler_detail {
AERO_EULER_HD inline EulerStatus roe_basis_with_normal(unsigned dimension,const double* original_left,
    const double* original_right,const double* direction,double gamma,EulerRoeBasis* result,double* n) {
    if (!result) return EulerStatus::NullPointer;
    EulerState left,right;
    EulerStatus status=euler_state(dimension,original_left,gamma,&left);
    if (status!=EulerStatus::Success) return status;
    status=euler_state(dimension,original_right,gamma,&right);
    if (status!=EulerStatus::Success) return status;
    status=euler_unit_direction(dimension,direction,n);
    if (status!=EulerStatus::Success) return status;
    const double root_left=::sqrt(left.density),root_right=::sqrt(right.density);
    const double a=root_left/(root_left+root_right),b=root_right/(root_left+root_right);
    EulerRoeBasis basis={}; basis.dimension=dimension;
    for (unsigned k=0;k<dimension;++k) basis.velocity[k]=a*left.velocity[k]+b*right.velocity[k];
    basis.enthalpy=a*left.enthalpy+b*right.enthalpy;
    const double kinetic=0.5*euler_detail::dot(dimension,basis.velocity,basis.velocity);
    const double sound2=(gamma-1)*(basis.enthalpy-kinetic);
    if (!euler_detail::finite(sound2) || !euler_detail::finite(basis.enthalpy)) return EulerStatus::NonFinite;
    if (!(sound2>0)) return EulerStatus::InvalidRoeState;
    basis.sound_speed=::sqrt(sound2);
    const double un=euler_detail::dot(dimension,basis.velocity,n),c=basis.sound_speed;
    double tangent[2][3]={{0,0,0},{0,0,0}};
    if (dimension==2) { tangent[0][0]=-n[1]; tangent[0][1]=n[0]; }
    if (dimension==3) {
        unsigned axis=0;
        // Choose from the raw direction so CPU/GPU normalization rounding
        // cannot change the shear basis when components are nearly tied.
        for (unsigned k=1;k<3;++k) if (euler_detail::absolute(direction[k])<euler_detail::absolute(direction[axis])) axis=k;
        for (unsigned k=0;k<3;++k) tangent[0][k]=(k==axis?1.0:0.0)-n[axis]*n[k];
        const double length=::sqrt(euler_detail::dot(3,tangent[0],tangent[0]));
        for (unsigned k=0;k<3;++k) tangent[0][k]/=length;
        tangent[1][0]=n[1]*tangent[0][2]-n[2]*tangent[0][1];
        tangent[1][1]=n[2]*tangent[0][0]-n[0]*tangent[0][2];
        tangent[1][2]=n[0]*tangent[0][1]-n[1]*tangent[0][0];
    }
    const unsigned energy=dimension+1,plus=dimension+1;
    basis.right[0]=1; basis.right[1]=1; basis.right[plus]=1;
    for (unsigned k=0;k<dimension;++k) {
        basis.right[5*(k+1)]=basis.velocity[k]-c*n[k];
        basis.right[5*(k+1)+1]=basis.velocity[k];
        basis.right[5*(k+1)+plus]=basis.velocity[k]+c*n[k];
    }
    basis.right[5*energy]=basis.enthalpy-c*un;
    basis.right[5*energy+1]=kinetic;
    basis.right[5*energy+plus]=basis.enthalpy+c*un;
    for (unsigned t=0;t+1<dimension;++t) {
        for (unsigned k=0;k<dimension;++k) basis.right[5*(k+1)+2+t]=tangent[t][k];
        basis.right[5*energy+2+t]=euler_detail::dot(dimension,basis.velocity,tangent[t]);
    }
    for (unsigned k=0;k<dimension+2;++k) basis.eigenvalues[k]=un;
    basis.eigenvalues[0]=un-c; basis.eigenvalues[plus]=un+c;
    for (unsigned k=0;k<25;++k) if (!euler_detail::finite(basis.right[k])) return EulerStatus::NonFinite;
    for (unsigned k=0;k<dimension+2;++k) if (!euler_detail::finite(basis.eigenvalues[k])) return EulerStatus::NonFinite;
    *result=basis; return EulerStatus::Success;
}
} // namespace euler_detail
AERO_EULER_HD inline EulerStatus euler_roe_basis(unsigned dimension,const double* original_left,
    const double* original_right,const double* direction,double gamma,EulerRoeBasis* result) {
    double normal[3];
    return euler_detail::roe_basis_with_normal(dimension,original_left,original_right,direction,gamma,result,normal);
}
} // namespace aero
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
#include <vector>
#include <stdexcept>
namespace aero {
const char* euler_status_message(EulerStatus status);
class EulerError : public std::runtime_error {
public:
    explicit EulerError(EulerStatus status) : std::runtime_error(euler_status_message(status)),status_(status) {}
    EulerStatus status() const { return status_; }
private:
    EulerStatus status_;
};
} // namespace aero
#endif
#undef AERO_EULER_HD
#endif
