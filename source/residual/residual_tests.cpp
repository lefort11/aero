#include "residual_detail.h"
#include "../eulersolver/roe.h"
#include "../eulersolver/lax_friedrichs.h"
#ifndef AERO_RESIDUAL_NO_TIME_TEST
#include "../timestep/timestep.h"
#include "../explicit/explicit.h"
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef AERO_WITH_CUDA
#include "../cuda/runtime.h"
#endif

using namespace aero;
namespace {
using Values=std::vector<double>;
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void near(double a,double b) {
    if (!(std::abs(a-b)<=3e-11*(1+std::abs(b)))) throw std::runtime_error("numerical mismatch: "+
        std::to_string(a)+" versus "+std::to_string(b));
}
struct Result {
    Values r,edge,bc;
    Result(size_t n,size_t e,size_t b):r(n),edge(e),bc(b){}
};
#ifdef AERO_WITH_CUDA
void checked(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct DeviceArray {
    double* p=nullptr;
    explicit DeviceArray(size_t n) { if(n) checked(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(double))); }
    ~DeviceArray() { cudaFree(p); }
    void read(Values& v) { if (!v.empty()) checked(cudaMemcpy(v.data(),p,v.size()*sizeof(double),cudaMemcpyDeviceToHost)); }
};
#endif
void evaluate(int device,SpatialResidual& op,const Values& u,Result& output) {
    if (device<0) {
        op.assemble(u.data(),u.data(),output.r.data(),output.edge.data(),output.bc.data());
        return;
    }
#ifdef AERO_WITH_CUDA
    checked(cudaSetDevice(device));
    DeviceArray state(u.size()),r(u.size()),edge(output.edge.size()),bc(output.bc.size());
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    try {
        checked(cudaMemcpyAsync(state.p,u.data(),u.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        op.assemble(state.p,state.p,r.p,edge.p,bc.p,stream);
        r.read(output.r); edge.read(output.edge); bc.read(output.bc);
    } catch (...) { cudaStreamDestroy(stream); throw; }
    checked(cudaStreamDestroy(stream));
#endif
}
void compare(const Values& a,const Values& b) {
    require(a.size()==b.size(),"array extent");
    for (size_t i=0;i<a.size();++i) near(a[i],b[i]);
}
void compare(const Result& a,const Result& b) {compare(a.r,b.r);compare(a.edge,b.edge);compare(a.bc,b.bc);}
std::vector<BoundaryConfig> free_patches(const Mesh& mesh) {
    BoundaryConfig bc={};
    bc.type=BoundaryType::Free;
    return std::vector<BoundaryConfig>(mesh.data().patch_names.size(),bc);
}
constexpr double gamma_gas=1.57;
void trace_limiter_test() {
    // gamma=2: nodal rho=2, u=(1,0), p=4. Candidate rho=4,
    // u=(3,2), p=12. The pressure cap gives weight 1/4 (density gives 1/2).
    const double node[]={2,2,0,5},normal[]={1,0};
    auto options=residual_options("conservative","roe",2,.5,1.5);
    double state[]={4,12,8,38};
    double flux[]={6,10,8,17};
    require(residual_detail::limit_trace(2,options,node,state,normal,flux),"limited trace");
    compare(Values(state,state+4),{2.5,3.75,1.25,9.125});
    compare(Values(flux,flux+4),{3,7,2,11});

    double low[]={.5,.5,0,4.25};
    require(residual_detail::limit_trace(2,options,node,low,normal,nullptr),"low density trace");
    compare(Values(low,low+4),{1,1,0,4.5});
    for (const auto& nonphysical : {Values{2,4,0,1},Values{-1,1,0,5}}) {
        Values trace=nonphysical;
        require(residual_detail::limit_trace(2,options,node,trace.data(),normal,flux),"zero-order fallback");
        compare(trace,{2,2,0,5});
        compare(Values(flux,flux+4),{2,6,0,9});
    }
    double nan_trace[]={2,2,NAN,5};
    require(!residual_detail::limit_trace(2,options,node,nan_trace,normal,nullptr),"nonfinite trace accepted");
}
void basis_test() {
    const double a[]={1,.21,.32,.43,4}, b[]={1.5,.2,.3,.4,5}, dir[]={-.4,.8,.32};
    for (unsigned d=1;d<=3;++d) {
        double left[5]={1,.21,.32,.43,4},right[5]={1.5,.2,.3,.4,5};
        left[d+1]=a[4]; right[d+1]=b[4];
        EulerRoeBasis basis;
        require(euler_roe_basis(d,left,right,dir,gamma_gas,&basis)==EulerStatus::Success,"basis");
        for (unsigned c=0;c<d+2;++c) {
            double u[5],w[5];
            for (unsigned k=0;k<d+2;++k) u[k]=basis.right[5*k+c];
            require(conservative_to_characteristic(basis,dir,gamma_gas,u,w)==EulerStatus::Success,"projection");
            for (unsigned k=0;k<d+2;++k) near(w[k],c==k ? 1 : 0);
        }
    }
}

// Independently assemble in scatter order, using full transformed fields on
// the whole stencil of each edge (only small 1D tests use this reference).
Result reference(const Mesh& mesh,CvView cv,const ReconstructionStencil& stencil,
                 ResidualOptions options,const Values& u,const std::vector<BoundaryConfig>& bc) {
    const auto m=mesh.view();
    Result out(u.size(),cv.edge_count,cv.boundary_count);
    std::vector<bool> open_node(m.node_count,false);
    for (MeshIndex k=0;k<cv.boundary_count;++k) {
        const auto& b=cv.boundary[k];
        if (bc[b.patch].type!=BoundaryType::Wall) open_node[b.node]=true;
    }
    for (MeshIndex e=0;e<m.edge_count;++e) {
        const auto i=m.edge_begin[e],j=m.edge_end[e];
        const double n=cv.edge_areas[e].x;
        EulerRoeBasis basis;
        require(euler_roe_basis(1,&u[i*3],&u[j*3],&n,options.gamma,&basis)==EulerStatus::Success,"reference basis");
        Values q=u;
        for (MeshIndex k=0;k<m.node_count;++k) {
            double speed;
            if (options.variables==ResidualVariables::Characteristic)
                require(conservative_to_characteristic(basis,&n,options.gamma,&u[3*k],&q[3*k])==EulerStatus::Success,"project");
            if (options.variables==ResidualVariables::Flux)
                require(euler_flux(1,&u[3*k],&n,options.gamma,&q[3*k],&speed)==EulerStatus::Success,"project flux");
        }
        auto side=[&](const Values& field,MeshIndex node,ReconstructionSide which,double* value) {
            if (open_node[i] || open_node[j]) {
                std::copy_n(field.data()+3*node,3,value);
                return;
            }
            const ReconstructionFieldView read={field.data(),3,1};
            double delta[3];
            reconstruct_increment2(stencil.view(),e,which,&field[3*node],3,read,delta);
            for (unsigned k=0;k<3;++k) value[k]=field[3*node+k]+(which==ReconstructionSide::Left ? .5 : -.5)*delta[k];
        };
        double l[3],r[3],ul[3],ur[3];
        side(q,i,ReconstructionSide::Left,l);side(q,j,ReconstructionSide::Right,r);
        EulerFluxResult f{};
        const bool roe=options.solver==ResidualSolver::Roe;
        if (options.variables==ResidualVariables::Flux) {
            side(u,i,ReconstructionSide::Left,ul);side(u,j,ReconstructionSide::Right,ur);
            require((roe ? roe_fluxes(1,l,r,ul,ur,&n,options.gamma,&f) :
                lax_friedrichs_fluxes(1,l,r,ul,ur,&n,options.gamma,&f))==EulerStatus::Success,"reference fluxes");
        } else {
            if (options.variables==ResidualVariables::Characteristic) {
                characteristic_to_conservative(basis,l,l);characteristic_to_conservative(basis,r,r);
            }
            require((roe ? roe_conservative(1,l,r,&n,options.gamma,&f) :
                lax_friedrichs_conservative(1,l,r,&n,options.gamma,&f))==EulerStatus::Success,"reference states");
        }
        for (unsigned k=0;k<3;++k) {
            out.r[3*i+k]+=std::abs(n)*f.flux[k]; out.r[3*j+k]-=std::abs(n)*f.flux[k];
        }
        out.edge[e]=f.max_wave_speed;
    }
    for (MeshIndex k=0;k<cv.boundary_count;++k) {
        const auto& piece=cv.boundary[k];
        EulerFluxResult f{};
        const double* state=&u[3*piece.node];
        require(boundary_flux(1,bc[piece.patch],state,state,&piece.area.x,options.gamma,&f)==BoundaryStatus::Success,"reference BC");
        for (unsigned c=0;c<3;++c) out.r[3*piece.node+c]+=f.flux[c];
        out.bc[k]=f.max_wave_speed;
    }
    return out;
}
void edge_modes(int device) {
    const Mesh mesh=Mesh::read("examples/tecplot/demo_1d.mesh");
    const auto m=mesh.view();
    MedianDual dual(mesh);
    const auto cv=dual.view();
    auto bc=free_patches(mesh);
    bc[0].type=BoundaryType::Wall;
    Values u(3*m.node_count);
    for (MeshIndex k=0;k<m.node_count;++k) {
        u[k*3]=1+.15*std::sin(9*m.x[k]);
        u[k*3+1]=.15*std::cos(18*m.x[k]);
        u[k*3+2]=2.8+.26*std::sin(15*m.x[k]);
    }
    for (auto scheme:{ReconstructionScheme::FirstOrder,ReconstructionScheme::Ebr3,ReconstructionScheme::Ebr5,
                      ReconstructionScheme::Weno3,ReconstructionScheme::Weno5}) {
        ReconstructionOptions opts;opts.scheme=scheme;
        auto stencil=prepare_reconstruction(mesh,opts);
        for (const char* vars:{"conservative","characteristic","flux"}) for (const char* solver:{"lf","roe"}) {
            const auto o=residual_options(vars,solver,gamma_gas);
            SpatialResidual op(m,cv,stencil,bc,o,device);
            Result result(u.size(),cv.edge_count,cv.boundary_count);
            evaluate(device,op,u,result);
            compare(result,reference(mesh,cv,stencil,o,u,bc));
        }
    }
}
void constant_test(unsigned d,int device) {
    const Mesh mesh=Mesh::read("examples/tecplot/demo_"+std::to_string(d)+"d.mesh");
    const auto m=mesh.view();
    MedianDual dual(mesh);
    const auto cv=dual.view();
    auto bc=free_patches(mesh);
    ReconstructionOptions opts; opts.scheme=ReconstructionScheme::Ebr3;
    auto stencil=prepare_reconstruction(mesh,opts);
    Values u((d+2)*m.node_count);
    for (MeshIndex n=0;n<m.node_count;++n) {
        u[(d+2)*n]=1; u[(d+2)*n+d+1]=3;
        for (unsigned k=0;k<d;++k) u[(d+2)*n+1+k]=.1*(k+1);
    }
    for (const char* vars:{"conservative","characteristic","flux"}) {
        SpatialResidual op(m,cv,stencil,bc,residual_options(vars,"roe",gamma_gas),device);
        Result out(u.size(),cv.edge_count,cv.boundary_count);
        evaluate(device,op,u,out);
        for (double v:out.r) near(v,0);
        require(*std::min_element(out.edge.begin(),out.edge.end())>0,"edge speeds");
        require(*std::min_element(out.bc.begin(),out.bc.end())>0,"boundary speeds");
        Values varying=u;
        for (size_t i=0;i<m.node_count;++i) {
            varying[(d+2)*i]+=0.15*std::cos(4*m.x[i]+3*m.y[i]+m.z[i]);
            varying[(d+2)*i+1]+=0.2*std::sin(7*m.y[i]-3*m.x[i]);
        }
        evaluate(device,op,varying,out);
        if (device>=0) {
            SpatialResidual host(m,cv,stencil,bc,residual_options(vars,"roe",gamma_gas));
            Result expected=out;
            evaluate(-1,host,varying,expected);
            compare(out,expected);
        }
#ifdef _OPENMP
        if (device<0) {
            const int threads=omp_get_max_threads();
            omp_set_num_threads(1);
            Result serial=out;
            evaluate(-1,op,varying,serial);
            omp_set_num_threads(threads);
            require(out.r==serial.r && out.edge==serial.edge && out.bc==serial.bc,"OpenMP summation order");
        }
#endif
        Values invalid(varying.size(),0);
        bool rejected=false;
        try {evaluate(device,op,invalid,out);} catch(const std::domain_error&) {rejected=true;}
        require(rejected,"invalid state accepted");
        evaluate(device,op,u,out);
        for (double v:out.r) near(v,0);
    }
}
void inlet_outlet_edges(int device) {
    const Mesh mesh=Mesh::read("examples/tecplot/demo_2d.mesh");
    const auto m=mesh.view();
    MedianDual dual(mesh);
    const auto cv=dual.view();
    auto bc=free_patches(mesh);
    ReconstructionOptions opts;
    opts.scheme=ReconstructionScheme::Weno5;
    auto stencil=prepare_reconstruction(mesh,opts);
    for (size_t patch=0;patch<bc.size();++patch) {
        const auto& name=mesh.data().patch_names[patch];
        if (name=="xmin") {
            bc[patch].type=BoundaryType::SuperIn;
            bc[patch].inflow.mode=BoundaryMode::SuperIn;
            bc[patch].inflow.source=BoundarySource::Initial;
        } else if (name=="xmax") {
            bc[patch].type=BoundaryType::SuperOut;
            bc[patch].outflow.mode=BoundaryMode::SuperOut;
        }
    }
    Values u(4*m.node_count);
    for (MeshIndex n=0;n<m.node_count;++n) {
        u[4*n]=1+.1*m.x[n];
        u[4*n+1]=u[4*n]*(3+.1*m.y[n]);
        u[4*n+2]=.1*m.x[n];
        u[4*n+3]=2.5+.3*m.x[n]+.2*m.y[n]+.5*u[4*n+1]*u[4*n+1]/u[4*n];
    }
    SpatialResidual op(m,cv,stencil,bc,residual_options("characteristic","roe",gamma_gas),device);
    Result out(u.size(),cv.edge_count,cv.boundary_count);
    evaluate(device,op,u,out);
    std::vector<bool> boundary_node(m.node_count,false);
    for (MeshIndex i=0;i<cv.boundary_count;++i) boundary_node[cv.boundary[i].node]=true;
    size_t tangent_edges=0,interior_to_boundary_edges=0;
    for (MeshIndex e=0;e<m.edge_count;++e) {
        const auto i=m.edge_begin[e],j=m.edge_end[e];
        if (!boundary_node[i] && !boundary_node[j]) continue;
        if (!stencil.data().weno_order[2*e] && !stencil.data().weno_order[2*e+1]) continue;
        const double area[]={cv.edge_areas[e].x,cv.edge_areas[e].y};
        EulerFluxResult expected{};
        require(roe_conservative(2,&u[4*i],&u[4*j],area,gamma_gas,&expected)==EulerStatus::Success,
                "inlet/outlet reference flux");
        near(out.edge[e],expected.max_wave_speed);
        if (boundary_node[i] && boundary_node[j]) ++tangent_edges;
        else ++interior_to_boundary_edges;
    }
    require(tangent_edges && interior_to_boundary_edges,"missing inlet/outlet edge coverage");
}
void ring_test(int device) {
    const Mesh mesh=Mesh::read("examples/tecplot/demo_1d.mesh");
    PeriodicDomain ring(mesh,{{"inlet","outlet",{1,0,0}}});
    const auto g=ring.view();
    ReconstructionOptions opts;opts.scheme=ReconstructionScheme::Weno3;
    SpatialResidual op(1,g,ring.prepare(opts),{},residual_options("characteristic","roe",gamma_gas),device);
    Values u(3*g.node_count);
    for (size_t n=0;n<g.node_count;++n) { u[3*n]=1+.1*std::sin(6.28*mesh.view().x[n]); u[3*n+1]=.4;u[3*n+2]=3; }
    Result out(u.size(),g.edge_count,0);
    evaluate(device,op,u,out);
    for (unsigned k=0;k<3;++k) {
        double total=0;
        for (size_t n=0;n<g.node_count;++n) total+=out.r[3*n+k];
        near(total,0);
    }
}
#ifndef AERO_RESIDUAL_NO_TIME_TEST
void sod_step(int device) {
    Mesh mesh=Mesh::read("examples/tecplot/demo_1d.mesh");
    MedianDual dual(mesh);
    const auto cv=dual.view();
    const auto m=mesh.view();
    ReconstructionOptions opts;opts.scheme=ReconstructionScheme::FirstOrder;
    SpatialResidual op(m,cv,prepare_reconstruction(mesh,opts),free_patches(mesh),residual_options("conservative","lf",1.4),device);
    Values u(3*m.node_count);
    for (size_t n=0;n<m.node_count;++n) {u[3*n]=m.x[n]<.5 ? 1 : .125;u[3*n+2]=m.x[n]<.5 ? 2.5 : .25;}
    Result out(u.size(),cv.edge_count,cv.boundary_count);
    evaluate(device,op,u,out);
    CflTimestep cfl(m,cv);
    const auto dt=cfl.compute(.4,{out.edge.data(),out.edge.size(),out.bc.data(),out.bc.size()});
    ExplicitIntegrator step(m.node_count,3,cv.volumes,u.data(),device<0?ExplicitBackend::Cpu:ExplicitBackend::Cuda,device<0?0:device);
    step.step(ExplicitScheme::Euler,0,dt.dt,[&](double,const ExplicitField& f,double* r){
        const double actual=op.assemble_cfl(f.values,f.values,r,f.volumes,.4,f.stream);
        require(actual==dt.dt,"Sod CFL differs from reduction of the exported speeds");
    });
    const auto next=step.state();
    require(next!=u,"Sod jump must evolve");
    for (size_t i=0;i<next.size();++i) near(next[i],u[i]-dt.dt*out.r[i]/cv.volumes[i/3]);
}
#endif
}
int main(int argc,char** argv) {
    try {
        basis_test();
        trace_limiter_test();
        const int devices=ReconstructionContext::cuda_device_count();
        if (argc>1 && std::string(argv[1])=="--require-cuda") require(devices>0,"CUDA device required");
        for (int dev=-1;dev<devices;++dev) {
#ifndef AERO_RESIDUAL_NO_TIME_TEST
            sod_step(dev);
#endif
            edge_modes(dev); ring_test(dev); inlet_outlet_edges(dev); constant_test(2,dev); constant_test(3,dev);
            std::cout<<"Residual tests passed: "<<(dev<0 ? "CPU" : "GPU "+std::to_string(dev))<<"\n";
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
