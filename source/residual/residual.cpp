#include "residual_detail.h"
#include "../parallel/parallel_for.h"
#include <cmath>
#include <limits>
#include <numeric>
#ifdef AERO_WITH_CUDA
#include "../cuda/runtime.h"
#endif

namespace aero {
using namespace residual_detail;
namespace {
void validate(ResidualOptions o) {
    if (!(o.gamma>1 && std::isfinite(o.gamma))) throw std::invalid_argument("residual: invalid gamma");
    if (!(o.trace_ratio_min>=0 && o.trace_ratio_min<1 && o.trace_ratio_max>1 && std::isfinite(o.trace_ratio_max)))
        throw std::invalid_argument("residual: invalid trace ratio bounds");
    if (o.variables!=ResidualVariables::Conservative && o.variables!=ResidualVariables::Characteristic &&
        o.variables!=ResidualVariables::Flux) throw std::invalid_argument("residual: invalid reconstruction variables");
    if (o.solver!=ResidualSolver::Roe && o.solver!=ResidualSolver::LocalLF)
        throw std::invalid_argument("residual: invalid solver");
}
double dot(CvVector a,CvVector b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
double length(CvVector a) { return std::hypot(std::hypot(a.x,a.y),a.z); }
void project(Surface& s,CvVector world,unsigned d,const ResidualFrame& frame) {
    s.measure=length(world);
    if (!(s.measure>0 && std::isfinite(s.measure))) throw std::invalid_argument("residual: invalid area");
    CvVector remainder=world;
    for (unsigned k=0;k<d;++k) {
        const auto axis=frame.axis[k];
        const double a=dot(world,axis);
        s.area[k]=a;
        remainder.x-=a*axis.x; remainder.y-=a*axis.y; remainder.z-=a*axis.z;
    }
    if (!(length(remainder)<=1.e-10*s.measure))
        throw std::invalid_argument("residual: area outside the supplied momentum frame");
}
void failure(Index bad, const Geometry& g) {
    if (bad<g.edges) throw std::domain_error("residual: invalid flux at edge "+std::to_string(bad));
    if (bad<g.faces) throw std::domain_error("residual: invalid boundary flux at piece "+std::to_string(bad-g.edges));
    throw std::domain_error("residual: nonfinite sum at node "+std::to_string(bad-g.faces));
}
#ifdef AERO_WITH_CUDA
void check(cudaError_t e) { if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct DeviceScope {
    int previous;
    explicit DeviceScope(int device) { check(cudaGetDevice(&previous)); check(cudaSetDevice(device)); }
    ~DeviceScope() { cudaSetDevice(previous); }
};
struct DeviceStorage {
    int device=-1;
    std::vector<void*> allocations;
    template<class T> T* allocate(std::size_t n,const T* src=nullptr) {
        if (!n) return nullptr;
        allocations.push_back(nullptr);
        check(cudaMalloc(&allocations.back(),n*sizeof(T)));
        auto* p=static_cast<T*>(allocations.back());
        if (src) check(cudaMemcpy(p,src,n*sizeof(T),cudaMemcpyHostToDevice));
        return p;
    }
    ~DeviceStorage() {
        if (allocations.empty()) return;
        int previous=-1;
        cudaGetDevice(&previous); cudaSetDevice(device);
        for (auto p:allocations) cudaFree(p);
        if (previous>=0) cudaSetDevice(previous);
    }
};
#endif
}
#ifdef AERO_WITH_CUDA
void launch_residual(Geometry,const double*,const double*,EulerFluxResult*,double*,double*,double*,Index*,cudaStream_t);
#endif

ResidualOptions residual_options(const std::string& variables,const std::string& solver,double gamma,
                                 double trace_ratio_min,double trace_ratio_max) {
    ResidualOptions o{};
    if (variables=="conservative") o.variables=ResidualVariables::Conservative;
    else if (variables=="characteristic") o.variables=ResidualVariables::Characteristic;
    else if (variables=="flux") o.variables=ResidualVariables::Flux;
    else throw std::invalid_argument("residual: unknown reconstruction variables: "+variables);
    if (solver=="roe") o.solver=ResidualSolver::Roe;
    else if (solver=="lf") o.solver=ResidualSolver::LocalLF;
    else throw std::invalid_argument("residual: unknown solver: "+solver);
    o.gamma=gamma;
    o.trace_ratio_min=trace_ratio_min;
    o.trace_ratio_max=trace_ratio_max;
    validate(o);
    return o;
}

struct SpatialResidual::Impl {
    int device;
    ReconstructionContext context;
    std::vector<Surface> surfaces;
    std::vector<BoundaryConfig> patches;
    std::vector<Index> offsets,incidence;
    std::vector<EulerFluxResult> buffer;
    Geometry g{};
#ifdef AERO_WITH_CUDA
    DeviceStorage storage;
    EulerFluxResult* device_flux=nullptr;
    Index* device_error=nullptr;
#endif
    Impl(ReconstructionStencil stencil,int id) : device(id),context(std::move(stencil),
        id<0 ? ReconstructionBackend::Cpu : ReconstructionBackend::Cuda,id<0 ? 0 : id) {
        if (id< -1) throw std::invalid_argument("residual: invalid device");
#ifndef AERO_WITH_CUDA
        if (id>=0) throw std::invalid_argument("residual: CUDA is not enabled in this build");
#endif
    }
    template<class EdgeAt> void prepare(unsigned d,Index nodes,Index edges,Index pieces,
        const CvBoundary* boundary,EdgeAt edge_at,const std::vector<BoundaryConfig>& bc,
        ResidualOptions options,const ResidualFrame& frame) {
        validate(options);
        const auto stencil=context.host_view();
        if (d<1 || d>3 || !nodes || nodes!=stencil.node_count || edges!=stencil.edge_count ||
            (pieces && !boundary) || edges>surfaces.max_size() || pieces>surfaces.max_size()-edges ||
            nodes>=offsets.max_size() || nodes>std::vector<double>().max_size()/(d+2))
            throw std::invalid_argument("residual: incompatible grid or stencil");
        for (unsigned i=0;i<d;++i) for (unsigned j=0;j<=i;++j)
            if (!(std::abs(dot(frame.axis[i],frame.axis[j])-(i==j ? 1 : 0))<1.e-12))
                throw std::invalid_argument("residual: invalid momentum frame");
        surfaces.resize(edges+pieces);
        offsets.resize(nodes+1);
        std::vector<unsigned char> inlet_outlet_node(nodes,0);
        for (Index f=0;f<surfaces.size();++f) {
            Surface& s=surfaces[f];
            if (f<edges) {
                const auto e=edge_at(f);
                s.begin=e.begin; s.end=e.end;
                if (s.begin>=nodes || s.end>=nodes || stencil.row_owner[2*f]!=s.begin || stencil.row_owner[2*f+1]!=s.end)
                    throw std::invalid_argument("residual: edge/stencil endpoint mismatch");
                project(s,e.area,d,frame);
                ++offsets[s.end+1];
            } else {
                const auto& b=boundary[f-edges];
                s.begin=b.node; s.patch=b.patch;
                if (b.node>=nodes || b.patch>=bc.size() || bc[b.patch].type==BoundaryType::Periodic)
                    throw std::invalid_argument("residual: invalid physical boundary");
                const auto type=bc[b.patch].type;
                if (type==BoundaryType::Free || type==BoundaryType::SuperIn || type==BoundaryType::SuperOut)
                    inlet_outlet_node[b.node]=1;
                project(s,b.area,d,frame);
            }
            ++offsets[s.begin+1];
        }
        std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
        incidence.resize(offsets.back());
        auto cursor=offsets;
        for (Index f=0;f<surfaces.size();++f) {
            auto& s=surfaces[f];
            incidence[cursor[s.begin]++]=2*f;
            if (f<edges) {
                s.use_nodal_traces=inlet_outlet_node[s.begin] || inlet_outlet_node[s.end];
                incidence[cursor[s.end]++]=2*f+1;
            }
        }
        patches=bc;
        g={d,nodes,edges,surfaces.size(),surfaces.data(),offsets.data(),incidence.data(),patches.data(),stencil,options};
        if (device<0) { buffer.resize(surfaces.size()); return; }
#ifdef AERO_WITH_CUDA
        DeviceScope scope(device);
        storage.device=device;
        g.surfaces=storage.allocate(surfaces.size(),surfaces.data());
        g.offsets=storage.allocate(offsets.size(),offsets.data());
        g.incidence=storage.allocate(incidence.size(),incidence.data());
        g.patches=storage.allocate(patches.size(),patches.data());
        g.stencil=context.device_view();
        device_flux=storage.allocate<EulerFluxResult>(surfaces.size());
        device_error=storage.allocate<Index>(1);
        // Assembly may immediately consume the uploaded geometry on a nonblocking stream.
        check(cudaStreamSynchronize(nullptr));
#endif
    }
};

SpatialResidual::SpatialResidual(MeshView mesh,CvView cv,ReconstructionStencil stencil,
    const std::vector<BoundaryConfig>& patches,ResidualOptions options,int device,const ResidualFrame& frame)
    :p_(new Impl(std::move(stencil),device)) {
    if (cv.dimension!=mesh.dimension || cv.edge_count!=mesh.edge_count || cv.node_count!=mesh.node_count ||
        (cv.edge_count && (!mesh.edge_begin || !mesh.edge_end || !cv.edge_areas)))
        throw std::invalid_argument("residual: incompatible mesh and control volumes");
    p_->prepare(cv.dimension,cv.node_count,cv.edge_count,cv.boundary_count,cv.boundary,
        [&](Index e)->PeriodicEdge {return {mesh.edge_begin[e],mesh.edge_end[e],e,cv.edge_areas[e]};},
        patches,options,frame);
}
SpatialResidual::SpatialResidual(unsigned d,PeriodicView grid,ReconstructionStencil stencil,
    const std::vector<BoundaryConfig>& patches,ResidualOptions options,int device,const ResidualFrame& frame)
    :p_(new Impl(std::move(stencil),device)) {
    if (grid.edge_count && !grid.edges) throw std::invalid_argument("residual: missing periodic edges");
    p_->prepare(d,grid.node_count,grid.edge_count,grid.boundary_count,grid.boundary,
        [&](Index e){return grid.edges[e];},patches,options,frame);
}
SpatialResidual::~SpatialResidual() {}

void SpatialResidual::assemble(const double* u,const double* u0,double* r,double* edge,double* boundary,void* stream) {
    const auto& g=p_->g;
    if (!u || !r || (g.edges && !edge) || (g.faces>g.edges && !boundary))
        throw std::invalid_argument("residual: missing input or output");
#ifdef AERO_WITH_CUDA
    if (p_->device>=0) {
        DeviceScope scope(p_->device);
        const auto queue=static_cast<cudaStream_t>(stream);
        check(cudaMemsetAsync(p_->device_error,0xff,sizeof(Index),queue));
        launch_residual(g,u,u0,p_->device_flux,r,edge,boundary,p_->device_error,queue);
        Index bad;
        check(cudaMemcpyAsync(&bad,p_->device_error,sizeof(bad),cudaMemcpyDeviceToHost,queue));
        check(cudaStreamSynchronize(queue));
        if (bad!=~Index(0)) failure(bad,g);
        return;
    }
#endif
    if (stream) throw std::invalid_argument("residual: CUDA stream on CPU");
    // Each edge owns one flux. Gathering the two signed incidences requires
    // no atomic updates and uses the same summation order on both backends.
    parallel_for(g.faces,[&](Index f) {
        auto& flux=p_->buffer[f];
        if (!surface_flux(g,f,u,u0,flux)) failure(f,g);
        if (f<g.edges) edge[f]=flux.max_wave_speed;
        else boundary[f-g.edges]=flux.max_wave_speed;
    },256);
    parallel_for(g.nodes,[&](Index i) {
        if (!gather(g,i,p_->buffer.data(),r)) failure(g.faces+i,g);
    },256);
}
}
