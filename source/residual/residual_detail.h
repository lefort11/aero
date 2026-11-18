#ifndef AERO_RESIDUAL_DETAIL_H
#define AERO_RESIDUAL_DETAIL_H
#include "residual.h"
#include "../bc/bc.h"
#include "../eulersolver/roe.h"
#include "../eulersolver/lax_friedrichs.h"
#include "../reconstruction/reconstruction.h"
#ifdef __CUDACC__
#define AERO_RES_HD __host__ __device__
#else
#define AERO_RES_HD
#endif
namespace aero { namespace residual_detail {
using Index=ReconstructionIndex;
struct Surface {
    Index begin,end,patch;
    double area[3], measure;
    bool use_nodal_traces;
};
struct Geometry {
    unsigned dim;
    Index nodes,edges,faces;
    const Surface* surfaces;
    const Index *offsets,*incidence;
    const BoundaryConfig* patches;
    ReconstructionStencilView stencil;
    ResidualOptions options;
};
struct Reader {
    const Geometry& g;
    const double* field;
    const Surface& surface;
    const EulerRoeBasis& basis;
    const double* normal;
    ResidualVariables variables;
    AERO_RES_HD EulerStatus project(const double* u,double* q) const {
        // The energy entry of the entropy eigenvector is |v_Roe|^2/2.
        return euler_detail::project_characteristic(basis,normal,g.options.gamma,
            basis.right[5*(g.dim+1)+1],u,q);
    }
    AERO_RES_HD EulerStatus transform(const double* u,double* q) const {
        if (variables==ResidualVariables::Characteristic)
            return project(u,q);
        if (variables==ResidualVariables::Flux) {
            double speed;
            return euler_flux(g.dim,u,surface.area,g.options.gamma,q,&speed);
        }
        for (unsigned k=0;k<g.dim+2;++k) q[k]=u[k];
        return EulerStatus::Success;
    }
    // NaNs propagate through the reconstruction and are rejected before use.
    AERO_RES_HD double operator()(Index,Index node,Index component) const {
        const double* u=field+node*(g.dim+2);
        if (variables==ResidualVariables::Conservative) return u[component];
        double q[5];
        return transform(u,q)==EulerStatus::Success ? q[component] : ::nan("");
    }
};

// Accumulate all characteristic components in the original sparse-row order.
// The first node is the reference in the zero-sum difference formula.
AERO_RES_HD inline bool characteristic_row(const Reader& reader,Index owner,const double* own,
    const Index* offsets,const Index* nodes,const double* weights,Index row,double* sum,unsigned stride) {
    const unsigned width=reader.g.dim+2;
    for (unsigned c=0;c<width;++c) sum[c*stride]=0;
    const Index begin=offsets[row],end=offsets[row+1];
    if (begin==end) return true;
    double first[5];
    const double* reference=own;
    if (nodes[begin]!=owner) {
        if (reader.project(reader.field+nodes[begin]*width,first)!=EulerStatus::Success) return false;
        reference=first;
    }
    for (Index i=begin+1;i<end;++i) {
        double projected[5];
        const double* q=own;
        if (nodes[i]!=owner) {
            if (reader.project(reader.field+nodes[i]*width,projected)!=EulerStatus::Success) return false;
            q=projected;
        }
        for (unsigned c=0;c<width;++c) sum[c*stride]+=weights[i]*(q[c]-reference[c]);
    }
    return true;
}

AERO_RES_HD inline bool characteristic_weno_increment(const Reader& reader,Index row,const double* own,double* out) {
    const auto& s=reader.g.stencil;
    const Index owner=s.row_owner[row];
    const unsigned order=s.weno_order[row];
    if (!order) return characteristic_row(reader,owner,own,s.offsets,s.nodes,s.weights,row,out,1);
    double slopes[5*4];
    for (unsigned k=0;k<(order==5?4u:2u);++k)
        if (!characteristic_row(reader,owner,own,s.slope_offsets,s.slope_nodes,s.slope_weights,
                                4*row+k,slopes+k,4)) return false;
    for (unsigned c=0;c<reader.g.dim+2;++c)
        out[c]=weno_increment2(order,slopes+4*c,s.weno_epsilon_scale[row]);
    return true;
}

AERO_RES_HD inline bool traces(const Reader& reader,Index edge,double* left,double* right) {
    const unsigned width=reader.g.dim+2;
    for (unsigned side=0;side<2;++side) {
        const Index node=side ? reader.surface.end : reader.surface.begin;
        const double* u=reader.field+node*width;
        double* dst=side ? right : left;
        if (reader.surface.use_nodal_traces) {
            if (reader.variables==ResidualVariables::Flux) {
                if (reader.transform(u,dst)!=EulerStatus::Success) return false;
            } else {
                for (unsigned k=0;k<width;++k) dst[k]=u[k];
            }
            continue;
        }
        double own[5],increment[5];
        if (reader.transform(u,own)!=EulerStatus::Success) return false;
        const auto which=side ? ReconstructionSide::Right : ReconstructionSide::Left;
        if (reader.variables==ResidualVariables::Characteristic && reader.g.stencil.weno_order) {
            if (!characteristic_weno_increment(reader,2*edge+side,own,increment)) return false;
        } else if (reconstruct_increment2(reader.g.stencil,edge,which,own,width,reader,increment)!=ReconstructionStatus::Success)
            return false;
        if (reader.variables==ResidualVariables::Characteristic) {
            // The basis is linear: apply R only to the increment. A constant
            // stencil therefore preserves the endpoint U exactly.
            if (characteristic_to_conservative(reader.basis,increment,increment)!=EulerStatus::Success) return false;
            for (unsigned k=0;k<width;++k) own[k]=u[k];
        }
        for (unsigned k=0;k<width;++k) dst[k]=own[k]+(side ? -0.5 : 0.5)*increment[k];
    }
    return true;
}

AERO_RES_HD inline double limit_ratio(double base,double value,const ResidualOptions& o) {
    const double lo=base*o.trace_ratio_min,hi=base*o.trace_ratio_max;
    if (value<lo) return (lo-base)/(value-base);
    if (value>hi) return (hi-base)/(value-base);
    return 1;
}

// Use the nodal state outside the absolute density and pressure bounds.
// Inside those bounds, limit all primitive variables with one blending weight.
AERO_RES_HD inline bool limit_trace(unsigned d,const ResidualOptions& o,const double* node,double* trace,
                                     const double* area,double* reconstructed_flux) {
    EulerState base;
    if (euler_state(d,node,o.gamma,&base)!=EulerStatus::Success) return false;
    for (unsigned k=0;k<d+2;++k) if (!euler_detail::finite(trace[k])) return false;
    EulerState candidate;
    double weight=0;
    if (trace[0]>=1e-16 && trace[0]<=1e16) {
        const auto status=euler_state(d,trace,o.gamma,&candidate);
        if (status!=EulerStatus::Success && status!=EulerStatus::InvalidPressure) return false;
        if (status==EulerStatus::Success && candidate.pressure>=1e-16 && candidate.pressure<=1e16)
            weight=::fmin(limit_ratio(base.density,candidate.density,o),
                          limit_ratio(base.pressure,candidate.pressure,o));
    }
    if (weight==1) return true;
    if (weight==0) {
        for (unsigned k=0;k<d+2;++k) trace[k]=node[k];
    } else {
        const double keep=1-weight;
        trace[0]=keep*base.density+weight*candidate.density;
        const double p=keep*base.pressure+weight*candidate.pressure;
        double kinetic=0;
        for (unsigned k=0;k<d;++k) {
            const double v=keep*base.velocity[k]+weight*candidate.velocity[k];
            trace[k+1]=trace[0]*v;
            kinetic+=0.5*trace[k+1]*v;
        }
        trace[d+1]=p/(o.gamma-1)+kinetic;
    }
    if (reconstructed_flux) {
        double nodal_flux[5],speed;
        if (euler_flux(d,node,area,o.gamma,nodal_flux,&speed)!=EulerStatus::Success) return false;
        for (unsigned k=0;k<d+2;++k)
            reconstructed_flux[k]=(1-weight)*nodal_flux[k]+weight*reconstructed_flux[k];
    }
    return true;
}

AERO_RES_HD inline bool surface_flux(const Geometry& g,Index id,const double* u,const double* u0,EulerFluxResult& out) {
    const Surface& s=g.surfaces[id];
    const unsigned width=g.dim+2;
    const double gamma=g.options.gamma;
    if (id>=g.edges)
        return boundary_flux(g.dim,g.patches[s.patch],u+s.begin*width,u0 ? u0+s.begin*width : nullptr,
                             s.area,gamma,&out)==BoundaryStatus::Success;
    EulerRoeBasis basis{};
    double normal[3];
    if (g.options.variables==ResidualVariables::Characteristic &&
        euler_detail::roe_basis_with_normal(g.dim,u+s.begin*width,u+s.end*width,s.area,gamma,&basis,normal)!=EulerStatus::Success)
        return false;
    const bool roe=g.options.solver==ResidualSolver::Roe;
    Reader reader={g,u,s,basis,normal,g.options.variables};
    double left[5],right[5];
    if (!traces(reader,id,left,right)) return false;
    EulerStatus status;
    if (reader.variables==ResidualVariables::Flux) {
        double ul[5],ur[5];
        reader.variables=ResidualVariables::Conservative;
        if (!traces(reader,id,ul,ur)) return false;
        if (!limit_trace(g.dim,g.options,u+s.begin*width,ul,s.area,left) ||
            !limit_trace(g.dim,g.options,u+s.end*width,ur,s.area,right)) return false;
        status=roe ? roe_fluxes(g.dim,left,right,ul,ur,s.area,gamma,&out)
                   : lax_friedrichs_fluxes(g.dim,left,right,ul,ur,s.area,gamma,&out);
    } else {
        if (!limit_trace(g.dim,g.options,u+s.begin*width,left,s.area,nullptr) ||
            !limit_trace(g.dim,g.options,u+s.end*width,right,s.area,nullptr)) return false;
        status=roe ? roe_conservative(g.dim,left,right,s.area,gamma,&out)
                   : lax_friedrichs_conservative(g.dim,left,right,s.area,gamma,&out);
    }
    if (status!=EulerStatus::Success) return false;
    for (unsigned k=0;k<width;++k) {
        out.flux[k]*=s.measure;
        if (!euler_detail::finite(out.flux[k])) return false;
    }
    return true;
}

AERO_RES_HD inline bool gather(const Geometry& g,Index node,const EulerFluxResult* flux,double* r) {
    double sum[5]={};
    for (Index i=g.offsets[node];i<g.offsets[node+1];++i) {
        const Index entry=g.incidence[i];
        for (unsigned k=0;k<g.dim+2;++k)
            sum[k]+=(entry&1) ? -flux[entry/2].flux[k] : flux[entry/2].flux[k];
    }
    for (unsigned k=0;k<g.dim+2;++k) {
        if (!euler_detail::finite(sum[k])) return false;
        r[node*(g.dim+2)+k]=sum[k];
    }
    return true;
}
} }
#undef AERO_RES_HD
#endif
