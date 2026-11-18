#include "explicit_test_problem.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace aero;
void check(bool passed,const char* message) {
    if (!passed) throw std::runtime_error(message);
}
void near(double value,double expected) {
    check(std::fabs(value-expected)<2e-12*(1+std::fabs(expected)),"incorrect stage, update or conserved integral");
}
void evaluate(TestProblem problem,double time,const ExplicitField& field,double* residual) {
#ifdef AERO_WITH_CUDA
    if (field.backend==ExplicitBackend::Cuda) {
        device_test_residual(problem,time,field,residual);
        return;
    }
#endif
    for (std::size_t k=0;k<field.nodes*field.components;++k) residual[k]=test_residual(problem,time,field,k);
}
double convergence_error(ExplicitBackend backend,int device,ExplicitScheme scheme,unsigned n) {
    const double volume=.23,initial=1,dt=1./n;
    ExplicitIntegrator solver(1,1,&volume,&initial,backend,device);
    for (unsigned i=0;i<n;++i) {
        unsigned calls=0;
        double start=i*dt;
        solver.step(scheme,start,dt,[&](double time,const ExplicitField& f,double* r) {
            const double fractions[]={0,1,.5};
            check(calls<3,"too many residual evaluations");
            near(time,start+dt*fractions[calls++]);
            evaluate(TestProblem::Forced,time,f,r);
        });
        check(calls==(scheme==ExplicitScheme::Euler?1u:3u),"incorrect residual count");
    }
    return std::fabs(solver.state()[0]-(2*std::exp(1.)-2));
}

// Four computational nodes on a periodic line of interval lengths [.5,1.5,1,1].
// Check every component against the exact RK stability polynomial of its
// semidiscrete upwind operator, as well as the volume-weighted conservation law.
void periodic(ExplicitBackend backend,int device,ExplicitScheme scheme) {
    const double v[]={.75,1,1.25,1},dt=.1;
    const std::vector<double> q={1,.1,3,2,-.3,4,.8,.9,2,1.5,.4,5};
    const auto apply=[&](const std::vector<double>& u) {
        std::vector<double> du(u.size());
        for (unsigned i=0;i<4;++i) for (unsigned c=0;c<3;++c)
            du[3*i+c]=(u[3*((i+3)%4)+c]-u[3*i+c])/v[i];
        return du;
    };
    const auto a=apply(q), aa=apply(a), aaa=apply(aa);
    ExplicitIntegrator solver(4,3,v,q.data(),backend,device);
    solver.step(scheme,0,dt,[](double t,const ExplicitField& f,double* r){evaluate(TestProblem::Ring,t,f,r);});
    const auto out=solver.state();
    for (unsigned c=0;c<3;++c) {
        double balance=0;
        for (unsigned i=0;i<4;++i) {
            const unsigned k=3*i+c;
            double ref=q[k]+dt*a[k];
            if (scheme==ExplicitScheme::SSPRK3) ref+=dt*dt*(.5*aa[k]+dt*aaa[k]/6);
            near(out[k],ref);
            balance+=v[i]*(out[k]-q[k]);
        }
        near(balance,0);
    }
}
void exercise(ExplicitBackend backend,int device) {
    for (auto scheme:{ExplicitScheme::Euler,ExplicitScheme::SSPRK3}) {
        const double coarse=convergence_error(backend,device,scheme,16);
        const double fine=convergence_error(backend,device,scheme,32);
        const double p=std::log(coarse/fine)/std::log(2.);
        const double order=scheme==ExplicitScheme::Euler?1.:3.;
        check(std::fabs(p-order)<.1,"incorrect temporal order");
        std::cout<<(scheme==ExplicitScheme::Euler?"Euler":"SSPRK3")<<" observed order "<<p<<"\n";
        periodic(backend,device,scheme);
    }
    const double v=2,initial=1;
    ExplicitIntegrator solver(1,1,&v,&initial,backend,device);
    for (bool throw_in_callback:{true,false}) {
        unsigned stage=0;
        bool failed=false;
        try {
            solver.step(ExplicitScheme::SSPRK3,0,.1,[&](double t,const ExplicitField& f,double* r) {
                if (++stage==2 && throw_in_callback) throw std::domain_error("rejected by assembler");
                evaluate(stage==2?TestProblem::Invalid:TestProblem::Forced,t,f,r);
            });
        } catch(const std::domain_error&) { failed=true; }
        check(failed && stage==2 && solver.state()[0]==initial,"failed RK stage committed");
    }
    solver.step(ExplicitScheme::Euler,0,.125,[](double t,const ExplicitField& f,double* r){evaluate(TestProblem::Forced,t,f,r);});
    near(solver.state()[0],1.125);

    // Cross both the OpenMP threshold and several CUDA blocks with a tail.
    const std::size_t nodes=2049,components=3;
    std::vector<double> volumes(nodes),u(nodes*components);
    for (std::size_t i=0;i<nodes;++i) volumes[i]=.5+.125*static_cast<double>(i%5);
    for (std::size_t k=0;k<u.size();++k) u[k]=static_cast<double>(k%127)-32;
    ExplicitIntegrator large(nodes,components,volumes.data(),u.data(),backend,device);
    large.step(ExplicitScheme::Euler,.5,.25,[](double t,const ExplicitField& f,double* r){evaluate(TestProblem::Forced,t,f,r);});
    const auto result=large.state();
    for (std::size_t k=0;k<u.size();++k) near(result[k],u[k]+.25*(u[k]+.5));
}
}
int main(int argc,char** argv) {
    try {
        exercise(aero::ExplicitBackend::Cpu,0);
        std::cout<<"CPU explicit tests passed.\n";
        const int count=aero::ExplicitIntegrator::cuda_device_count();
        if (argc==2 && std::string(argv[1])=="--require-cuda") check(count>0,"CUDA device required");
        for (int i=0;i<count;++i) {
            exercise(aero::ExplicitBackend::Cuda,i);
            std::cout<<"CUDA explicit tests passed on GPU "<<i<<".\n";
        }
        const double one=1;
        aero::ExplicitIntegrator automatic(1,1,&one,&one);
        check(automatic.backend()==(count?aero::ExplicitBackend::Cuda:aero::ExplicitBackend::Cpu),"incorrect auto backend");
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
