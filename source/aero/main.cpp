#include "startup.h"
#include "../explicit/explicit.h"
#include "../initial/initial.h"
#include "../io/flowreport.h"
#include "../io/scheduled_output.h"
#include "../io/settings_boundary.h"
#include "../residual/residual.h"
#include "../timestep/timestep.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#ifdef AERO_WITH_CUDA
#include "../cuda/runtime.h"
#endif

namespace {
using namespace aero;
#ifdef AERO_WITH_CUDA
void checked(cudaError_t code) {
    if (code!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(code));
}
#endif
class Buffer {
    std::vector<double> host_;
    double* device_=nullptr;
public:
    Buffer(std::size_t count,int device) {
        if (device<0) host_.resize(count);
#ifdef AERO_WITH_CUDA
        else if (count) checked(cudaMalloc(reinterpret_cast<void**>(&device_),count*sizeof(double)));
#endif
    }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer() {
#ifdef AERO_WITH_CUDA
        if (device_) cudaFree(device_);
#endif
    }
    double* data() { return device_ ? device_ : host_.data(); }
};
int select_device(const RunSettings& run) {
    if (!run.use_cuda) return -1;
    const int count=ExplicitIntegrator::cuda_device_count();
    if (!count) {
        std::cout<<"CUDA unavailable; running on CPU.\n";
        return -1;
    }
    if (run.devices.size()!=1)
        throw std::invalid_argument("One calculation uses one GPU. Select one device in cuda.txt.");
    const int id=run.devices.front();
    if (id>=count) throw std::invalid_argument("Unavailable CUDA device "+std::to_string(id));
#ifdef AERO_WITH_CUDA
    checked(cudaSetDevice(id));
#endif
    return id;
}
CvVector cross(CvVector a,CvVector b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
double length(CvVector a) { return std::hypot(std::hypot(a.x,a.y),a.z); }
CvVector unit(CvVector a) {
    const double n=length(a);
    if (!(n>0 && std::isfinite(n))) throw MeshError("Cannot determine mesh momentum frame");
    return {a.x/n,a.y/n,a.z/n};
}
CvVector orient(CvVector v) {
    double major=std::abs(v.x)>=std::abs(v.y)?v.x:v.y;
    if (std::abs(v.z)>std::abs(major)) major=v.z;
    return major<0 ? CvVector{-v.x,-v.y,-v.z} : v;
}
ResidualFrame momentum_frame(MeshView m) {
    ResidualFrame f;
    if (m.dimension==3) return f;
    const auto edge=[&](MeshIndex i) {
        const auto a=m.edge_begin[i],b=m.edge_end[i];
        return unit({m.x[b]-m.x[a],m.y[b]-m.y[a],m.z[b]-m.z[a]});
    };
    const auto tangent=edge(0);
    if (m.dimension==1) f.axis[0]=orient(tangent);
    else {
        CvVector normal={};
        for (MeshIndex i=1;i<m.edge_count && length(normal)<0.5;++i) {
            const auto candidate=cross(tangent,edge(i));
            if (length(candidate)>length(normal)) normal=candidate;
        }
        normal=orient(unit(normal));
        const CvVector reference=std::abs(normal.x)>0.99 ? CvVector{0,1,0} : CvVector{1,0,0};
        f.axis[1]=unit(cross(normal,reference));
        f.axis[0]=unit(cross(f.axis[1],normal));
    }
    return f;
}
ReconstructionOptions reconstruction(const std::string& name) {
    ReconstructionOptions r;
    const std::map<std::string,ReconstructionScheme> schemes={
        {"first_order",ReconstructionScheme::FirstOrder},{"ebr3",ReconstructionScheme::Ebr3},
        {"ebr5",ReconstructionScheme::Ebr5},{"weno3",ReconstructionScheme::Weno3},{"weno5",ReconstructionScheme::Weno5}};
    r.scheme=schemes.at(name);
    return r;
}
int calculate(const RunSettings& run) {
    show_settings(run);
    // Delay initial-condition validation until after listing the boundary labels.
    const bool have_initial=bool(std::ifstream(run.initial_path.c_str()));
    const auto initial_text=have_initial?settings_text(run.initial_path):std::string();
    if (have_initial) std::cout<<"Initial settings\n"<<initial_text;
    if (std::ifstream(run.boundary_path.c_str())) std::cout<<"Boundary settings\n"<<settings_text(run.boundary_path);
    stage("Loading mesh");
    const Mesh mesh=Mesh::read(run.mesh_path);
    done();
    const auto m=mesh.view();
    std::cout<<"Dimension = "<<m.dimension<<"  nodes = "<<m.node_count<<"  cells = "<<m.cell_count<<'\n';
    show_boundary_labels(mesh);
    if (!prepare_boundary_file(run,mesh)) return 2;
    if (!m.cell_count || !m.edge_count) throw MeshError("The calculation requires a nonempty mesh");
    stage("Reading boundary conditions");
    const auto bc=read_case_boundaries(run.boundary_path,mesh.data());
    done();
    if (run.directories.interactive) std::cout<<"Accepted boundary settings\n"<<settings_text(run.boundary_path);
    const int device=select_device(run);
    std::cout<<"Active backend = "<<(device<0?"CPU":"CUDA device "+std::to_string(device))<<'\n';
    const bool periodic=!bc.periodic_pairs.empty();
    std::unique_ptr<PeriodicDomain> domain;
    std::unique_ptr<MedianDual> dual;
    PeriodicView grid={};
    CvView cv={};
    stage(periodic?"Building median control volumes and periodic links":"Building median control volumes");
    if (periodic) {
        domain.reset(new PeriodicDomain(mesh,bc.periodic_pairs));
        grid=domain->view();
    } else {
        dual.reset(new MedianDual(mesh));
        cv=dual->view();
    }
    done();
    const auto nodes=periodic?grid.node_count:cv.node_count;
    const auto edges=periodic?grid.edge_count:cv.edge_count;
    const auto pieces=periodic?grid.boundary_count:cv.boundary_count;
    const auto* volumes=periodic?grid.volumes:cv.volumes;
    stage("Building reconstruction stencils");
    const auto options=reconstruction(run.reconstruction);
    auto stencil=periodic?domain->prepare(options):prepare_reconstruction(mesh,options);
    done();
    std::cout<<"Computational nodes = "<<nodes<<"  edges = "<<edges
             <<"  downgraded reconstruction sides = "<<stencil.incomplete_sides()<<'\n';
    stage("Loading initial state");
    std::istringstream init(have_initial?initial_text:settings_text(run.initial_path));
    const auto initial=read_initial(mesh,run.gamma,init,run.directories.input,periodic?&grid:nullptr);
    done();
    stage("Preparing residual and CFL calculation");
    const auto physics=residual_options(run.reconstruction_variables,run.euler_solver,run.gamma,
                                        run.trace_ratio_min,run.trace_ratio_max);
    const auto frame=momentum_frame(m);
    std::unique_ptr<SpatialResidual> residual(periodic
        ? new SpatialResidual(m.dimension,grid,std::move(stencil),bc.by_patch,physics,device,frame)
        : new SpatialResidual(m,cv,std::move(stencil),bc.by_patch,physics,device,frame));
    std::unique_ptr<CflTimestep> cfl(periodic?new CflTimestep(grid,device):new CflTimestep(m,cv,device));
    Buffer edge_speed(edges,device),boundary_speed(pieces,device),first_residual(initial.size(),device);
    Buffer device_initial(device<0?0:initial.size(),device);
    const double* u0=initial.data();
#ifdef AERO_WITH_CUDA
    if (device>=0) {
        checked(cudaMemcpy(device_initial.data(),u0,initial.size()*sizeof(double),cudaMemcpyHostToDevice));
        u0=device_initial.data();
    }
#endif
    const WaveSpeeds speeds={edge_speed.data(),static_cast<std::size_t>(edges),
                             boundary_speed.data(),static_cast<std::size_t>(pieces)};
    ExplicitIntegrator solver(nodes,m.dimension+2,volumes,initial.data(),
        device<0?ExplicitBackend::Cpu:ExplicitBackend::Cuda,device<0?0:device);
    const auto scheme=run.time_scheme=="euler"?ExplicitScheme::Euler:ExplicitScheme::SSPRK3;
    done();
    stage("Preparing field output and flow reports");
    std::vector<MeshIndex> map;
    if (periodic && !run.output_zones.empty()) map.assign(grid.node_dof,grid.node_dof+grid.original_node_count);
    ScheduledOutput output(run.output_zones,run.directories.output,mesh,run.gamma,map);
    prepare_output_directories(run);
    FlowReport report(run.flowreport_every,run.gamma,m.dimension,nodes,volumes);
    done();
    std::cout<<"Starting time integration\n";
    std::uint64_t step=0;
    double time=0;
    while (time<run.final_time) {
        if (step==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Step number overflow");
        double cfl_dt=0;
        solver.inspect([&](const ExplicitField& f) {
            residual->assemble(f.values,u0,first_residual.data(),edge_speed.data(),boundary_speed.data(),f.stream);
            cfl_dt=cfl->compute(run.cfl,speeds,f.stream).dt;
        });
        const double remaining=run.final_time-time;
        const double dt=output.limit_dt(time,std::min(cfl_dt,remaining));
        const double next=dt==remaining?run.final_time:time+dt;
        if (!(next>time)) throw std::runtime_error("Time step too small to advance time");
        report.capture(step+1,solver);
        bool first=true;
        solver.step(scheme,time,dt,[&](double,const ExplicitField& f,double* r) {
            if (first) {
                first=false;
#ifdef AERO_WITH_CUDA
                if (device>=0) {
                    checked(cudaMemcpyAsync(r,first_residual.data(),initial.size()*sizeof(double),
                                            cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(f.stream)));
                    return;
                }
#endif
                std::copy_n(first_residual.data(),initial.size(),r);
            } else residual->assemble(f.values,u0,r,edge_speed.data(),boundary_speed.data(),f.stream);
        });
        time=next;
        ++step;
        if (output.due(step,time)) {
            const auto state=solver.state();
            report.write(step,time,run.cfl,dt,state);
            output.write(step,time,state);
        } else report.write(step,time,run.cfl,dt,solver);
    }
    std::cout<<"Completed "<<step<<" steps. Final time = "<<time<<'\n';
    return 0;
}
}
int main(int argc,char** argv) {
    if (argc==2 && std::string(argv[1])=="--help") {
        std::cout<<"Usage: aero --input=<directory> --output=<directory> [--interactive]\n";
        return 0;
    }
    try { return calculate(aero::read_run_settings(aero::parse_run_arguments(argc,argv))); }
    catch (const std::exception& e) { std::cerr<<"\naero: "<<e.what()<<'\n'; return 1; }
}
