#ifndef AERO_BC_H
#define AERO_BC_H

#include "../eulersolver/euler.h"

#if defined(__CUDACC__)
#define AERO_BC_HD __host__ __device__
#else
#define AERO_BC_HD
#endif

namespace aero {
enum class BoundaryType { Wall, Free, SuperIn, SuperOut, Periodic };
enum class BoundaryMode { Pressure, NormalVelocity, Density, SuperIn, SuperOut };
enum class BoundarySource { Current, Initial, Constant };
struct BoundaryRule {
    BoundaryMode mode;
    BoundarySource source;
    double value;
    // Explicit superin primitive overrides, bits 0..4 = rho, p, ux, uy, uz.
    unsigned mask;
    double primitive[5];
};
struct BoundaryConfig {
    BoundaryType type;
    BoundaryRule inflow, outflow;
};
enum class BoundaryStatus {
    Success, InvalidArgument, InvalidState, MissingInitial, IncompatibleRegime, InvalidTarget
};

namespace bc_detail {
AERO_BC_HD inline double scalar(BoundaryMode mode,const EulerState& q,double un) {
    if (mode==BoundaryMode::Pressure) return q.pressure;
    return mode==BoundaryMode::Density ? q.density : un;
}

AERO_BC_HD inline BoundaryStatus trace(unsigned d,const BoundaryRule& rule,
    const EulerState& inside,const double* n,double un,double gamma,
    const double* initial,EulerState& boundary) {
    const bool entering=un<0;
    const bool sonic=::fabs(un)>=inside.sound_speed;
    if (rule.mode==BoundaryMode::SuperOut)
        return sonic && !entering ? BoundaryStatus::Success : BoundaryStatus::IncompatibleRegime;
    const bool full=rule.mode==BoundaryMode::SuperIn;
    if (full && !(sonic && entering)) return BoundaryStatus::IncompatibleRegime;
    if (sonic && !entering) return BoundaryStatus::Success;

    EulerState reference=inside;
    if (rule.source==BoundarySource::Initial) {
        if (!initial) return BoundaryStatus::MissingInitial;
        if (euler_state(d,initial,gamma,&reference)!=EulerStatus::Success)
            return BoundaryStatus::InvalidState;
    }
    if (full) {
        boundary=reference;
        if (rule.mask&1u) boundary.density=rule.primitive[0];
        if (rule.mask&2u) boundary.pressure=rule.primitive[1];
        for (unsigned k=0;k<d;++k)
            if (rule.mask&(4u<<k)) boundary.velocity[k]=rule.primitive[2+k];
        if (rule.mask>>(2+d)) return BoundaryStatus::InvalidTarget;
    } else {
        if (rule.source==BoundarySource::Current) return BoundaryStatus::Success;
        const double target=rule.source==BoundarySource::Constant ? rule.value :
            scalar(rule.mode,reference,euler_detail::dot(d,reference.velocity,n));
        if (!euler_detail::finite(target) || (rule.mode!=BoundaryMode::NormalVelocity && target<=0))
            return BoundaryStatus::InvalidTarget;
        double ub=un;
        if (sonic) {
            // All characteristics enter. Override just the requested primitive.
            if (rule.mode==BoundaryMode::Pressure) boundary.pressure=target;
            if (rule.mode==BoundaryMode::Density) boundary.density=target;
            if (rule.mode==BoundaryMode::NormalVelocity) ub=target;
        } else {
            // Isentropic normal characteristic closure with interior tangential
            // velocity. Ratios avoid explicitly forming p/rho^gamma.
            const double c=inside.sound_speed;
            if (rule.mode==BoundaryMode::NormalVelocity) {
                ub=target;
                const double cb=c+0.5*(gamma-1)*(un-ub);
                if (!(cb>0)) return BoundaryStatus::InvalidTarget;
                boundary.density*=::pow(cb/c,2/(gamma-1));
                boundary.pressure*=::pow(cb/c,2*gamma/(gamma-1));
            } else {
                if (rule.mode==BoundaryMode::Pressure) {
                    boundary.pressure=target;
                    boundary.density*=::pow(target/inside.pressure,1/gamma);
                } else {
                    boundary.density=target;
                    boundary.pressure*=::pow(target/inside.density,gamma);
                }
                const double cb=::sqrt(gamma*(boundary.pressure/boundary.density));
                ub+=2*(c-cb)/(gamma-1);
            }
        }
        for (unsigned k=0;k<d;++k) boundary.velocity[k]+=(ub-un)*n[k];
    }
    if (!(boundary.pressure>0 && boundary.density>0) ||
        !euler_detail::finite(boundary.pressure) || !euler_detail::finite(boundary.density))
        return BoundaryStatus::InvalidTarget;
    const double cb=::sqrt(gamma*(boundary.pressure/boundary.density));
    const double ub=euler_detail::dot(d,boundary.velocity,n);
    if (!(cb>0 && euler_detail::finite(cb) && euler_detail::finite(ub)))
        return BoundaryStatus::InvalidTarget;
    if ((sonic && ub>-cb) || (!sonic && (::fabs(ub)>=cb || (entering ? ub>0 : ub<0))))
        return BoundaryStatus::IncompatibleRegime;
    return BoundaryStatus::Success;
}
}

// One stationary boundary piece. area points outward in the solver frame;
// the returned Euler flux already includes the area. initial is the immutable
// conservative state at this piece at t=0, required only by the active rule.
// At zero normal speed select outflow. At |un|=c use the supersonic branch.
// Invalid data or inconsistent prescribed regime leave result untouched.
AERO_BC_HD inline BoundaryStatus boundary_flux(unsigned d,const BoundaryConfig& config,
    const double* current,const double* initial,const double* area,double gamma,EulerFluxResult* result) {
    double n[3]={};
    if (!result || !current || euler_unit_direction(d,area,n)!=EulerStatus::Success)
        return BoundaryStatus::InvalidArgument;
    EulerState inside;
    if (euler_state(d,current,gamma,&inside)!=EulerStatus::Success) return BoundaryStatus::InvalidState;
    const double un=euler_detail::dot(d,inside.velocity,n);
    EulerFluxResult flux={};
    flux.components=d+2;
    flux.max_wave_speed=::fabs(un)+inside.sound_speed;
    if (config.type==BoundaryType::Wall) {
        for (unsigned k=0;k<d;++k) flux.flux[k+1]=inside.pressure*area[k];
    } else {
        if (config.type!=BoundaryType::Free && config.type!=BoundaryType::SuperIn &&
            config.type!=BoundaryType::SuperOut) return BoundaryStatus::InvalidArgument;
        const bool inflow=config.type==BoundaryType::SuperIn || (config.type==BoundaryType::Free && un<0);
        const BoundaryRule& rule=inflow ? config.inflow : config.outflow;
        if ((config.type==BoundaryType::SuperIn && rule.mode!=BoundaryMode::SuperIn) ||
            (config.type==BoundaryType::SuperOut && rule.mode!=BoundaryMode::SuperOut) ||
            (inflow && rule.mode==BoundaryMode::SuperOut) || (!inflow && rule.mode==BoundaryMode::SuperIn))
            return BoundaryStatus::InvalidArgument;
        EulerState b=inside;
        const BoundaryStatus status=bc_detail::trace(d,rule,inside,n,un,gamma,initial,b);
        if (status!=BoundaryStatus::Success) return status;
        const double normal=euler_detail::dot(d,b.velocity,n);
        const double volume_flux=euler_detail::dot(d,b.velocity,area);
        flux.flux[0]=b.density*volume_flux;
        for (unsigned k=0;k<d;++k)
            flux.flux[k+1]=flux.flux[0]*b.velocity[k]+b.pressure*area[k];
        const double enthalpy_density=gamma/(gamma-1)*b.pressure+
            0.5*b.density*euler_detail::dot(d,b.velocity,b.velocity);
        flux.flux[d+1]=enthalpy_density*volume_flux;
        flux.max_wave_speed=::fmax(flux.max_wave_speed,::fabs(normal)+::sqrt(gamma*(b.pressure/b.density)));
    }
    if (!euler_detail::finite(flux.max_wave_speed)) return BoundaryStatus::InvalidState;
    for (unsigned k=0;k<d+2;++k)
        if (!euler_detail::finite(flux.flux[k])) return BoundaryStatus::InvalidState;
    *result=flux;
    return BoundaryStatus::Success;
}
}
#undef AERO_BC_HD
#if !defined(AERO_BC_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
#include <iosfwd>
#include <string>
namespace aero {
struct PeriodicPair {
    std::string patch, pair;
    double translation[3]; // patch -> pair, in raw mesh coordinate units
};
// Read the settings block for one mesh patch. Throws std::invalid_argument on malformed input.
// bc_type wall
// bc_type free    (requires separate inflow and outflow lines)
// inflow/outflow base_p|base_u|base_rho [initial|number]
// inflow superin [initial] [rho=...] [p=...] [ux=...] [uy=...] [uz=...]
// outflow superout
// Dedicated supersonic patches accept the same rules after bc_type.
BoundaryConfig read_boundary(std::istream& input,unsigned dimension);
BoundaryConfig read_boundary(std::istream& input,unsigned dimension,PeriodicPair& periodic);
}
#endif
#endif
