#pragma once
#include "residual_detail.h"

namespace aero {
namespace residual_detail {

// Границы и старт первого порядка используют те же потоки, что в основном операторе.
AERO_HOST_DEVICE inline bool uses_split_flux(const Geometry& g) {
    return g.options.variables == ResidualVariables::CharacteristicFlux &&
           !g.options.background.active() && g.stencil.has_owners();
}

// Для каждой строки один раз суммируем разности U и F_n до проекции в общий базис.
// Скорость LF берём по всем реально использованным опорам обеих сторон.
template<unsigned W>
AERO_HOST_DEVICE inline bool split_row(const Reader& r, StencilRow row, const double* fl, const double* fr,
        double* slope, double& speed) {
    const auto read = [&](Index node, double* values) -> const double* {
        const double* u = r.field + W*node;
        if (node == r.surface.begin || node == r.surface.end) {
            const double* flux = node == r.surface.begin ? fl : fr;
            for (unsigned k=0;k<W;++k) { values[k]=u[k]; values[W+k]=flux[k]; }
            return values;
        }
        EulerState tmp;
        const EulerState* gas = r.g.packed.gas ? r.g.packed.gas+node : &tmp;
        if (!r.g.packed.gas && euler_state(W-2,u,r.g.options.gamma,&tmp)!=EulerStatus::Success) return nullptr;
        double local;
        if (!(gas->density>0.) || euler_detail::project_flux(W-2,u,*gas,r.normal,values+W,&local)!=EulerStatus::Success) return nullptr;
        speed=::fmax(speed,local);
        for (unsigned k=0;k<W;++k) values[k]=u[k];
        return values;
    };
    const auto values = [&](Index j,double* buffer) { return read(row.node(j),buffer); };
    const auto weight = [&](Index j) { return row.weight(j); };
    return reconstruction_detail::sum_slope<2*W>(2*W,row.count,values,weight,slope);
}

// Ограничителю нужны только плотность и давление. Для экстремальных значений
// полный декодер дополнительно проверяет представимость скорости звука и энтальпии.
template<unsigned W>
KOKKOS_FORCEINLINE_FUNCTION bool limiter_pressure(const double* state,double gamma,double& pressure) {
    if (state[0]>1.e-50 && state[0]<1.e50 && state[W-1]>0. && state[W-1]<1.e50 && gamma<1.e5) {
        const double inv=inverse_value(state[0]);
        double kinetic=0.;
        for (unsigned k=1;k+1<W;++k) kinetic+=.5*state[k]*(state[k]*inv);
        pressure=(gamma-1.)*(state[W-1]-kinetic);
        if (pressure>1.e-100) return true;
    }
    EulerState gas;
    if (euler_state(W-2,state,gamma,&gas)!=EulerStatus::Success) return false;
    pressure=gas.pressure;
    return true;
}

// Консервативное ограничение готового потока. При необходимости приближаем
// его к потоку Русанова до допустимости обоих промежуточных состояний.
template<unsigned W>
AERO_HOST_DEVICE inline bool limit_split_flux(const Geometry& g,const double* l,const double* r,
        const double* fl,const double* fr,double a,double* flux) {
    double bar[W], shift[W];
    for (unsigned k=0;k<W;++k) {
        bar[k]=.5*(l[k]+r[k]-(fr[k]-fl[k])/a);
        const double low=.5*(fl[k]+fr[k]-a*(r[k]-l[k]));
        shift[k]=(flux[k]-low)/a;
        if (!euler_detail::finite(shift[k])) return false;
    }
    double base_pressure;
    if (!limiter_pressure<W>(bar,g.options.gamma,base_pressure)) return false;
    for (double theta=1.;theta>0.;theta*=.5) {
        bool ok=true;
        for (int sign=-1;sign<=1;sign+=2) {
            double state[W],pressure;
            for (unsigned k=0;k<W;++k) state[k]=bar[k]+sign*theta*shift[k];
            ok=ok && limiter_pressure<W>(state,g.options.gamma,pressure) &&
                limit_ratio(bar[0],state[0],g.options)==1. && limit_ratio(base_pressure,pressure,g.options)==1.;
        }
        if (ok) {
            if (theta<1.)
                for (unsigned k=0;k<W;++k) flux[k]=.5*(fl[k]+fr[k]-a*(r[k]-l[k]))+theta*a*shift[k];
            return true;
        }
    }
    return false;
}

// Сливаем обход двух линейных строк. Общая опора требует одного чтения газа
// и одного физического потока. У каждой строки остаётся её собственная точка отсчёта.
template<unsigned W>
KOKKOS_FORCEINLINE_FUNCTION bool linear_split_slopes(const Geometry& g,Index edge,const Surface& face,
        const double* field,const double* normal,double* du,double* df,double& speed) {
    if (g.stencil.tiled_linear() && g.packed.gas) {
        const auto& st=g.stencil;
        const Index tile=edge/32;
        const unsigned* header=st.linear_indices32+3*tile;
        const auto* terms=reinterpret_cast<const EbrTerm*>(st.nodes)+header[0]+edge%32;
        const auto* masks=st.linear_indices32+3*((st.edge_count+31)/32)+2*edge;
        const auto* weights=st.weights+header[1];
        unsigned left=(face.use_nodal_traces&2) ? 0u : masks[0];
        unsigned right=(face.use_nodal_traces&4) ? 0u : masks[1];
        const double *ul=nullptr,*ur=nullptr;
        double fl[W],fr[W];
        for (unsigned j=0;left || right;++j,left>>=1,right>>=1) {
            if (!((left|right)&1u)) continue;
            const auto t=terms[32*j];
            const double* u=field+W*Index(t.node);
            const auto& gas=g.packed.gas[t.node];
            double flux[W],bound;
            if (!(gas.density>0.) || euler_detail::project_flux(W-2,u,gas,normal,flux,&bound)!=EulerStatus::Success) return false;
            speed=::fmax(speed,bound);
            if (left&1u) {
                if (!ul) { ul=u; for (unsigned k=0;k<W;++k) fl[k]=flux[k]; }
                else for (unsigned k=0;k<W;++k) {
                    du[k]+=weights[t.left]*(u[k]-ul[k]);
                    df[k]+=weights[t.left]*(flux[k]-fl[k]);
                }
            }
            if (right&1u) {
                if (!ur) { ur=u; for (unsigned k=0;k<W;++k) fr[k]=flux[k]; }
                else for (unsigned k=0;k<W;++k) {
                    du[k]+=weights[t.right]*(u[k]-ur[k]);
                    df[k]-=weights[t.right]*(flux[k]-fr[k]);
                }
            }
        }
        return true;
    }
    const auto l=(face.use_nodal_traces&2) ? StencilRow{} : g.stencil.linear_row(2*edge);
    const auto r=(face.use_nodal_traces&4) ? StencilRow{} : g.stencil.linear_row(2*edge+1);
    const auto fetch=[&](Index node,double* flux) {
        const double* u=field+W*node;
        EulerState local;
        const EulerState* gas=g.packed.gas ? g.packed.gas+node : &local;
        if (!g.packed.gas && euler_state(W-2,u,g.options.gamma,&local)!=EulerStatus::Success) return false;
        double bound;
        if (!(gas->density>0.) || euler_detail::project_flux(W-2,u,*gas,normal,flux,&bound)!=EulerStatus::Success) return false;
        speed=::fmax(speed,bound);
        return true;
    };
    const Index l0=l.count ? l.node(0) : face.begin, r0=r.count ? r.node(0) : face.end;
    double fl[W]={},fr[W]={};
    if (l.count && !fetch(l0,fl)) return false;
    if (r.count && !fetch(r0,fr)) return false;
    const Index end=~Index(0);
    Index i=1,j=1, left=i<l.count ? l.node(i) : end, right=j<r.count ? r.node(j) : end;
    while (left!=end || right!=end) {
        const auto node=left<right ? left : right;
        double flux[W];
        if (!fetch(node,flux)) return false;
        const double* u=field+W*node;
        if (left==node) {
            const double weight=l.weight(i);
            for (unsigned k=0;k<W;++k) {
                du[k]+=weight*(u[k]-field[W*l0+k]);
                df[k]+=weight*(flux[k]-fl[k]);
            }
            left=++i<l.count ? l.node(i) : end;
        }
        if (right==node) {
            const double weight=r.weight(j);
            for (unsigned k=0;k<W;++k) {
                du[k]+=weight*(u[k]-field[W*r0+k]);
                df[k]-=weight*(flux[k]-fr[k]);
            }
            right=++j<r.count ? r.node(j) : end;
        }
    }
    return true;
}

// Добавляет диссипацию LF или Роу к линейной реконструкции нормального потока.
template<unsigned W>
KOKKOS_FORCEINLINE_FUNCTION bool finish_linear_split(const Geometry& g,const Surface& s,const euler_detail::InterfaceGas& gas,
        const double* ul,const double* ur,const double* fl,const double* fr,double speed,
        double* diss,const double* df,EulerFluxResult& out) {
    for (unsigned k=0;k<W;++k) diss[k]=.25*diss[k]-.5*(ur[k]-ul[k]);
    if (g.options.solver==ResidualSolver::Roe) {
        euler_detail::RoeAverage avg;
        if (euler_detail::average_roe_state(W-2,gas.left,gas.right,gas.normal,g.options.gamma,avg)!=EulerStatus::Success) return false;
        double momentum[3],waves[W];
        euler_detail::project_normal_waves(W-2,avg.velocity,gas.normal,g.options.gamma,avg.kinetic,
                                           avg.sound_speed,1./avg.sound_speed,diss,momentum,waves);
        const double transport=::fabs(avg.normal_speed),width=g.options.entropy_fix*avg.sound_speed;
        const double minus=(euler_detail::acoustic_modulus(avg.normal_speed-avg.sound_speed,width)-transport)*waves[0];
        const double plus=(euler_detail::acoustic_modulus(avg.normal_speed+avg.sound_speed,width)-transport)*waves[W-1];
        for (unsigned k=0;k<W;++k) {
            double lm,rp;
            avg.acoustic_row(k,W-2,gas.normal,lm,rp);
            diss[k]=transport*diss[k]+lm*minus+rp*plus;
        }
    } else {
        for (unsigned k=0;k<W;++k) diss[k]*=speed;
    }
    for (unsigned k=0;k<W;++k)
        out.flux[k]=.5*(fl[k]+fr[k])+.25*df[k]+g.options.upw_coeff*diss[k];
    out.components=W;
    out.max_wave_speed=speed;
    return limit_split_flux<W>(g,ul,ur,fl,fr,speed,out.flux) && integrate_flux(W,s.measure,out);
}

// Расщеплённые потоки обеих сторон строятся в одном базисе Роу.
// Полученное значение уже включает диссипацию, повторная задача Римана не нужна.
template<unsigned W,bool linear_only=false,bool full_mp=false>
AERO_HOST_DEVICE inline bool characteristic_flux(const Geometry& g,Index edge,const double* field,EulerFluxResult& out) {
    const auto s=g.surfaces[edge];
    const double* ul=field+W*s.begin;
    const double* ur=field+W*s.end;
    euler_detail::InterfaceGas gas;
    if (!edge_gas(g,edge,s,ul,ur,gas)) return false;
    if (s.use_nodal_traces==1) return solve_edge_flux(g,s,ul,ur,gas,nullptr,out);
    double fl[W],fr[W],sl,sr;
    if (euler_detail::project_flux(W-2,ul,gas.left,gas.normal,fl,&sl)!=EulerStatus::Success ||
        euler_detail::project_flux(W-2,ur,gas.right,gas.normal,fr,&sr)!=EulerStatus::Success) return false;
    double speed=::fmax(sl,sr);
    const auto& st=g.stencil;
    unsigned order[2],rows[2];
    // Для линейных строк сложение коммутирует с проекцией. В базисе Роу
    // достаточно обработать единственный вектор диссипации на всё ребро.
    if (linear_only || (!full_mp && !st.weno_order)) {
        double diss[W]={},df[W]={};
        if (!linear_split_slopes<W>(g,edge,s,field,gas.normal,diss,df,speed)) return false;
        return finish_linear_split<W>(g,s,gas,ul,ur,fl,fr,speed,diss,df,out);
    }
    EulerRoeBasis basis;
    if (euler_detail::build_roe_basis<true,W-2>(W-2,gas.left,gas.right,s.area,gas.normal,g.options.gamma,&basis)!=EulerStatus::Success) return false;
    Reader reader={g,field,s,basis,gas.normal,ResidualVariables::Characteristic};
    const double inverse_sound=1./basis.sound_speed;
    const auto project = [&](const double* u,double* w) {
        return euler_detail::project_characteristic<W-2>(basis,gas.normal,g.options.gamma,basis.kinetic_energy(),u,w,inverse_sound)==EulerStatus::Success;
    };
    // На полном общем шаблоне хранятся пять различных наклонов. Постоянные индексы
    // дают компилятору возможность оставить проекции в регистрах потока CUDA.
    if (full_mp) {
        double dw[5][W],df[5][W],waves[W];
        for (unsigned j=0;j<5;++j) {
            double slope[2*W];
            if (j==0) {
                for (unsigned k=0;k<W;++k) { slope[k]=ur[k]-ul[k]; slope[k+W]=fr[k]-fl[k]; }
            } else if (!split_row<W>(reader,st.slope_row(8*edge+(j==4 ? 7 : j)),fl,fr,slope,speed)) return false;
            if (!project(slope,dw[j]) || !project(slope+W,df[j])) return false;
        }
        for (unsigned k=0;k<W;++k) {
            const double width=(k==0 || k==W-1) ? g.options.entropy_fix*basis.sound_speed : 0.;
            const double a=g.options.upw_coeff*(g.options.solver==ResidualSolver::Roe ?
                euler_detail::acoustic_modulus(basis.eigenvalues[k],width) : speed);
            const auto plus=[&](unsigned j){return .5*(df[j][k]+a*dw[j][k]);};
            const auto minus=[&](unsigned j){return .5*(df[j][k]-a*dw[j][k]);};
            double left[]={plus(0),plus(1),plus(2),plus(3)};
            double right[]={minus(0),minus(2),minus(1),minus(4)};
            waves[k]=-.5*a*dw[0][k]+.5*nonlinear_increment(5,left,st.epsilon(2*edge),st.mp5)
                                    -.5*nonlinear_increment(5,right,st.epsilon(2*edge+1),st.mp5);
        }
        if (euler_detail::expand_characteristic<W>(W,basis,waves,out.flux)!=EulerStatus::Success) return false;
        for (unsigned k=0;k<W;++k) out.flux[k]+=.5*(fl[k]+fr[k]);
        out.components=W;
        out.max_wave_speed=speed;
        return limit_split_flux<W>(g,ul,ur,fl,fr,speed,out.flux) && integrate_flux(W,s.measure,out);
    }
    double dw[2][4][W],dg[2][4][W];
    for (unsigned side=0;side<2;++side) {
        const Index row=2*edge+side;
        order[side]=st.weno_order ? st.weno_order[row] : 0;
        rows[side]=s.use_nodal_traces & (2u<<side) ? 0 : order[side] ? weno_slope_count(order[side]) : 1;
        for (unsigned j=0;j<rows[side];++j) {
            const unsigned shared = j==1 ? 2u : j==2 ? 1u : j;
            if (side && st.shared_edge_slopes && order[0] && order[1] && j<3 && shared<rows[0]) {
                for (unsigned k=0;k<W;++k) {
                    dw[1][j][k]=dw[0][shared][k];
                    dg[1][j][k]=dg[0][shared][k];
                }
            } else {
                double slope[2*W];
                if (!split_row<W>(reader,order[side] ? st.slope_row(4*row+j) : st.linear_row(row),fl,fr,slope,speed) ||
                    !project(slope,dw[side][j]) || !project(slope+W,dg[side][j])) return false;
            }
        }
    }
    double jump[W],wjump[W],waves[W],a[W];
    for (unsigned k=0;k<W;++k) jump[k]=ur[k]-ul[k];
    if (!project(jump,wjump)) return false;
    for (unsigned k=0;k<W;++k) {
        double lambda=speed;
        if (g.options.solver==ResidualSolver::Roe) {
            const int acoustic=k==0 ? -1 : k==W-1 ? 1 : 0;
            lambda=euler_detail::acoustic_modulus(basis.eigenvalues[k],acoustic ? g.options.entropy_fix*basis.sound_speed : 0.);
        }
        a[k]=lambda*g.options.upw_coeff;
        waves[k]=-.5*a[k]*wjump[k];
    }
    for (unsigned side=0;side<2;++side) {
        const double sign=side ? -1. : 1.;
        if (!rows[side]) continue;
        double delta[W];
        for (unsigned k=0;k<W;++k) {
            double v[4]={};
            for (unsigned j=0;j<rows[side];++j) v[j]=.5*(dg[side][j][k]+sign*a[k]*dw[side][j][k]);
            delta[k]=order[side] ? nonlinear_increment(order[side],v,st.epsilon(2*edge+side),st.mp5) : v[0];
        }
        if ((st.mp5.enabled || st.mp5.weno) && g.options.mp6_hybrid>0. && rows[0]==4 && rows[1]==4) {
            FixedMatrix<double,5,W> v;
            for (unsigned k=0;k<W;++k) {
                for (unsigned j=0;j<4;++j) v(j,k)=.5*(dg[0][j][k]+sign*a[k]*dw[0][j][k]);
                v(4,k)=.5*(dg[1][3][k]+sign*a[k]*dw[1][3][k]);
            }
            double l[W],r[W];
            if (st.mp5.weno) mp6::reconstruct<true>(v,W,st.mp5,g.options.mp6_hybrid,l,r);
            else mp6::reconstruct<false>(v,W,st.mp5,g.options.mp6_hybrid,l,r);
            for (unsigned k=0;k<W;++k) delta[k]=side ? r[k] : l[k];
        }
        for (unsigned k=0;k<W;++k) waves[k]+=.5*sign*delta[k];
    }
    if (euler_detail::expand_characteristic<W>(W,basis,waves,out.flux)!=EulerStatus::Success) return false;
    for (unsigned k=0;k<W;++k) out.flux[k]+=.5*(fl[k]+fr[k]);
    out.components=W;
    out.max_wave_speed=speed;
    return limit_split_flux<W>(g,ul,ur,fl,fr,speed,out.flux) && integrate_flux(W,s.measure,out);
}

// Общая точка входа для CPU, CUDA и списков площадок MPI.
AERO_HOST_DEVICE inline bool split_face(const FaceBatch& batch,Index face) {
    if (batch.is_boundary(face)) return batch.compute_boundary(face);
    const auto& g=batch.geometry;
    const bool ok=g.dim==2 ? characteristic_flux<4>(g,face,batch.state,batch.flux[face]) :
                  g.dim==3 ? characteristic_flux<5>(g,face,batch.state,batch.flux[face]) :
                             characteristic_flux<3>(g,face,batch.state,batch.flux[face]);
    if (ok) batch.save_speed(face);
    return ok;
}
}}
