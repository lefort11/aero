#include "solver.h"
#include "mpi_balance.h"
#include "partition_kernels.h"
#ifdef AERO_WITH_CUDA
#include "partition_gpu.h"
#endif
#include "time_loop.h"
#include "mpi_fields.h"
#include "recovery.h"
#include "mpi_periodic.h"
#include "mpi_stencil.h"
#include "partition_report.h"
#include "startup.h"
#include "../explicit/stage_sequence.h"
#include "../initial/initial.h"
#include "../io/flowreport.h"
#include "../io/scheduled_output.h"
#include "../io/settings_boundary.h"
#include "../mesh/mdec.h"
#include "../parallel/parallel_for.h"
#include "../residual/flux_cfl.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <sstream>

namespace aero {
class PartitionGpu;
namespace {
using Id=MeshIndex;
using residual_detail::Surface;
using residual_detail::Index;

class Transfers {
public:
    std::vector<std::vector<Id>> exports,imports;
    Transfers(int size,unsigned stride) : exports(size),imports(size),buffers_(size),stride_(stride) {}
    void allocate(bool cuda) {
        std::size_t total=0;
        const auto size_buffer=[&](DoubleExchange::Buffer& b,std::size_t items) {
            if (items>(host_.max_size()-total)/stride_) throw std::length_error("MPI buffer too large");
            b.count=items*stride_;
            total+=b.count;
        };
        for (std::size_t p=0;p<exports.size();++p) {
            if (exports[p].empty() && imports[p].empty()) continue;
            peers_.push_back(static_cast<int>(p));
            size_buffer(buffers_.send[p],exports[p].size());
            size_buffer(buffers_.receive[p],imports[p].size());
            sends_=sends_ || !exports[p].empty();
            receives_=receives_ || !imports[p].empty();
        }
        if (!total) return;
        double* next;
#ifdef AERO_WITH_CUDA
        if (cuda) {
            void* data=nullptr;
            const auto status=cudaMallocHost(&data,total*sizeof(double));
            if (status!=cudaSuccess) throw std::runtime_error(std::string("MPI pinned buffer: ")+cudaGetErrorString(status));
            pinned_.reset(static_cast<double*>(data));
            next=pinned_.get();
            void* alias=nullptr;
            const auto mapping=cudaHostGetDevicePointer(&alias,data,0);
            if (mapping==cudaSuccess) mapped_=static_cast<double*>(alias);
            else if (mapping==cudaErrorInvalidValue || mapping==cudaErrorNotSupported) cudaGetLastError();
            else throw std::runtime_error(cudaGetErrorString(mapping));
        } else
#else
        (void)cuda;
#endif
        {
            host_.resize(total);
            next=host_.data();
        }
        for (int p:peers_) {
            buffers_.send[p].data=next;
            next+=buffers_.send[p].count;
            buffers_.receive[p].data=next;
            next+=buffers_.receive[p].count;
        }
    }
#ifdef AERO_WITH_CUDA
    void prepare_gpu(PartitionGpu& gpu) {
        device_peers_.clear();
        for (int p:peers_) device_peers_.push_back(gpu.prepare(exports[p],imports[p],stride_));
    }
#endif
    ExchangeBytes bytes() const {
        ExchangeBytes bytes{};
        for (int p:peers_) {
            bytes.sent+=buffers_.send[p].count*sizeof(double);
            bytes.received+=buffers_.receive[p].count*sizeof(double);
        }
        return bytes;
    }
    void transfer(const Processes& world,unsigned width,double* state,EulerFluxResult* flux,bool edges,PartitionGpu* gpu=nullptr,double* seconds=nullptr) {
        WallTimer timer(seconds);
        const auto visit=[&](bool pack) {
            for (std::size_t j=0;j<peers_.size();++j) {
                const int p=peers_[j];
                auto& records=pack?buffers_.send[p]:buffers_.receive[p];
#ifdef AERO_WITH_CUDA
                if (gpu) {
                    double* alias=mapped_?mapped_+(records.data-pinned_.get()):nullptr;
                    gpu->exchange(device_peers_[j],records,alias,edges,pack);
                    continue;
                }
#endif
                const auto& ids=pack?exports[p]:imports[p];
                for (std::size_t k=0;k<ids.size();++k)
                    partition_detail::transfer_record(ids[k],width,state,flux,records.data+k*stride_,edges,pack);
            }
        };
        // A device copy can fail. Agree on success before any peer posts receives.
        if (gpu) world.local([&] {
            visit(true);
#ifdef AERO_WITH_CUDA
            // Finish all outgoing copies and any preceding reads of the receive slab before MPI uses it.
            if (sends_) gpu->synchronize();
#endif
        }); else visit(true);
        buffers_.transfer(peers_);
        if (gpu) world.local([&] {
            visit(false);
#ifdef AERO_WITH_CUDA
            // Receive-only ranks have no outgoing batch to fence reuse on the next call.
            if (receives_ && (seconds || !sends_)) gpu->synchronize();
#endif
        }); else visit(false);
    }
private:
    std::vector<double> host_;
#ifdef AERO_WITH_CUDA
    struct FreePinned {
        void operator()(double* p) const { cudaStreamSynchronize(nullptr); cudaFreeHost(p); }
    };
    std::unique_ptr<double,FreePinned> pinned_;
    double* mapped_=nullptr;
    std::vector<PartitionGpu::Peer> device_peers_;
#endif
    DoubleExchange buffers_;
    std::vector<int> peers_;
    bool sends_=false,receives_=false;
    unsigned stride_;
};

class PartitionOperator {
public:
    PartitionOperator(const MeshPartition& part,CvView cv,const PeriodicPartition* periodic,const std::vector<BoundaryConfig>& patches,
                        ResidualOptions options,const ResidualFrame& frame,const Processes& world,StencilRequests* requests,bool cuda)
        : nodes_(world.size,cv.dimension+2),edges_(world.size,cv.dimension+3),width_(cv.dimension+2),
          volumes_(periodic?periodic->volumes.data():cv.volumes),ids_(periodic?periodic->ids:part.nodes),
          owners_(periodic?periodic->owners:part.node_owners),world_(world) {
        const auto m=part.mesh.view();
        for (Id i=0;i<ids_.size();++i) {
            const int p=owners_[i];
            if (p==world.rank) owned.push_back(i);
            else nodes_.imports[p].push_back(i);
        }
        const Id edge_count=periodic?periodic->edges.size():m.edge_count;
        for (Id e=0;e<edge_count;++e) {
            const Id a=periodic?periodic->edges[e].begin:m.edge_begin[e];
            const Id b=periodic?periodic->edges[e].end:m.edge_end[e];
            const int left=owners_[a],right=owners_[b];
            if (left!=world.rank && right!=world.rank) continue;
            const auto f=surfaces_.size();
            Surface s{};
            s.begin=a; s.end=b; s.use_nodal_traces=true;
            residual_detail::project_surface(s,periodic?periodic->edges[e].area:cv.edge_areas[e],m.dimension,frame);
            surfaces_.push_back(s);
            if (left==world.rank) {
                compute_.push_back(f);
                if (requests) requests->edges.push_back({f,a,b,periodic?periodic->edges[e].original:std::make_pair(ids_[a],ids_[b])});
                if (right!=left) edges_.exports[right].push_back(f);
            } else edges_.imports[left].push_back(f);
        }
        const auto edges=surfaces_.size();
        const Id pieces=periodic?periodic->boundary.size():cv.boundary_count;
        for (Id k=0;k<pieces;++k) {
            const auto& b=periodic?periodic->boundary[k]:cv.boundary[k];
            if (b.patch>=part.physical_patches || owners_[b.node]!=world.rank) continue;
            const auto type=patches[b.patch].type;
            if (requests && residual_detail::needs_nodal_traces(type))
                requests->inlet_nodes.insert(ids_[b.node]);
            Surface s{};
            s.begin=b.node; s.patch=b.patch;
            residual_detail::project_surface(s,b.area,m.dimension,frame);
            compute_.push_back(surfaces_.size());
            surfaces_.push_back(s);
        }
        // Native edges (or their periodic representatives) come in original
        // edge order, followed by physical boundary pieces.
        residual_detail::build_incidence(ids_.size(),edges,surfaces_,offsets_,incidence_);
        grid_={m.dimension,ids_.size(),edges,surfaces_.size(),surfaces_.data(),offsets_.data(),incidence_.data(),
               patches.data(),{},options};
        if (!cuda) flux_.resize(surfaces_.size());
    }
    Id state_nodes() const { return ids_.size()+stencil_.extra_nodes.size(); }
    PartitionWork work(Id leaves) const {
        // Every physical boundary piece is in compute_, after the owned edges.
        const Id edges=compute_.size()-(surfaces_.size()-grid_.edges);
        return {leaves,owned.size(),state_nodes()-owned.size(),edges,nodes_.bytes(),edges_.bytes()};
    }
    Id prepare_stencil(const MeshPartition& part,const PeriodicPartition* seam,const std::vector<PeriodicPair>& pairs,
                         PartitionInput* input,const StencilRequests& requests,const ReconstructionOptions& options) {
        stencil_=prepare_partition_stencil(part,seam,pairs,input,requests,grid_.edges,options,world_);
        world_.local([&] {
            for (const auto& edge:requests.edges) surfaces_[edge.surface].use_nodal_traces=stencil_.nodal_traces[edge.surface]!=0;
            for (Id j=0;j<stencil_.extra_nodes.size();++j)
                nodes_.imports[stencil_.extra_nodes[j].second].push_back(ids_.size()+j);
            grid_.stencil=stencil_.view(state_nodes());
        });
        return world_.sum(stencil_.incomplete);
    }
    void connect(bool cuda) {
        Processes::Messages node_requests,edge_keys;
        world_.local([&] {
            node_requests.resize(world_.size); edge_keys.resize(world_.size);
            for (int p=0;p<world_.size;++p) {
                for (Id i:nodes_.imports[p]) {
                    const Id global=i<ids_.size()?ids_[i]:stencil_.extra_nodes[i-ids_.size()].first;
                    mdec::append_word(node_requests[p],global);
                }
                for (Id f:edges_.exports[p]) {
                    mdec::append_word(edge_keys[p],ids_[surfaces_[f].begin]);
                    mdec::append_word(edge_keys[p],ids_[surfaces_[f].end]);
                }
            }
        });
        const auto requests=world_.exchange(node_requests);
        const auto keys=world_.exchange(edge_keys);
        world_.local([&] {
            for (int p=0;p<world_.size;++p) {
                for (std::size_t k=0;k<requests[p].size();k+=8) {
                    const Id global=mdec::read_word(requests[p],k);
                    const auto it=std::lower_bound(ids_.begin(),ids_.end(),global);
                    if (it==ids_.end() || *it!=global) throw MeshError("Missing requested state");
                    const auto local=static_cast<Id>(it-ids_.begin());
                    if (owners_[local]!=world_.rank) throw MeshError("Invalid state owner");
                    nodes_.exports[p].push_back(local);
                }
                if (keys[p].size()%16 || keys[p].size()/16!=edges_.imports[p].size())
                    throw MeshError("Interface edge count mismatch");
                std::size_t k=0;
                for (Id f:edges_.imports[p]) {
                    if (mdec::read_word(keys[p],k)!=ids_[surfaces_[f].begin] ||
                        mdec::read_word(keys[p],k+8)!=ids_[surfaces_[f].end])
                        throw MeshError("Interface edge order mismatch");
                    k+=16;
                }
            }
            nodes_.allocate(cuda); edges_.allocate(cuda);
        });
    }
#ifdef AERO_WITH_CUDA
    void layout_gpu(std::size_t patches,bool compact_buffer) {
        gpu_->layout(grid_,patches,stencil_,compute_,owned,volumes_,state_nodes(),compact_buffer,world_.size==1);
        nodes_.prepare_gpu(*gpu_);
        edges_.prepare_gpu(*gpu_);
    }
    void reserve_gpu(std::size_t patches,bool compact_buffer,GpuMemoryFit& fit) {
        gpu_.reset(new PartitionGpu);
        layout_gpu(patches,compact_buffer);
        fit.failed=!gpu_->reserve(fit);
    }
    void upload_state(const std::vector<double>& state,const std::vector<double>& initial) {
        gpu_->upload_state(state,initial);
    }
#endif
    void copy_owned(const std::vector<double>& state,std::vector<double>& compact) const {
#ifdef AERO_WITH_CUDA
        if (gpu_) { gpu_->download_owned(compact); return; }
#endif
        parallel_for(owned.size(),[&](std::size_t k) {
            std::copy_n(state.data()+width_*owned[k],width_,compact.data()+width_*k);
        });
    }
    void download(std::vector<double>& state) const {
#ifdef AERO_WITH_CUDA
        if (gpu_) gpu_->download(state);
#else
        (void)state;
#endif
    }
    void halo(std::vector<double>& u,double* seconds) {
        nodes_.transfer(world_,width_,u.data(),nullptr,false,device(),seconds);
    }
    void halo_host(std::vector<double>& u) {
        nodes_.transfer(world_,width_,u.data(),nullptr,false);
    }
    void assemble_fluxes(const std::vector<double>& u,const std::vector<double>& initial,SolverTimings& timing) {
        world_.local([&] {
            WallTimer timer(timing.counter(SolverTimings::Flux));
#ifdef AERO_WITH_CUDA
            if (gpu_) { gpu_->fluxes(); return; }
#endif
            const Id* faces=world_.size==1?nullptr:compute_.data();
            residual_detail::cpu_face_fluxes({grid_,faces,compute_.size(),u.data(),initial.data(),flux_.data(),nullptr,nullptr},ids_.data());
        });
        if (world_.size==1) return;
        edges_.transfer(world_,width_,nullptr,flux_.data(),true,device(),timing.counter(SolverTimings::FluxExchange));
    }
    double cfl_step(double cfl) {
        double local=std::numeric_limits<double>::infinity();
        world_.local([&] {
#ifdef AERO_WITH_CUDA
            if (gpu_) { local=gpu_->cfl(cfl); return; }
#endif
            const Id* nodes=world_.size==1?nullptr:owned.data();
            local=residual_detail::cpu_flux_cfl({grid_,flux_.data(),volumes_,nodes,owned.size()},cfl,ids_.data());
        });
        const double dt=world_.minimum(local);
        if (!std::isfinite(dt)) throw std::domain_error("No finite global CFL step");
        return dt;
    }
    void update_owned(explicit_detail::Stage stage,std::vector<double>& start,double dt,std::vector<double>& state,bool save_original) const {
#ifdef AERO_WITH_CUDA
        if (gpu_) { gpu_->update(stage,dt,save_original); return; }
#endif
        const Id* nodes=world_.size==1?nullptr:owned.data();
        parallel_for(owned.size(),[&](std::size_t k) {
            if (!partition_detail::update_node(grid_,flux_.data(),volumes_,nodes,state.data(),start.data(),k,stage,dt,save_original))
                throw std::domain_error("Nonfinite stage at global node "+std::to_string(ids_[owned[k]]));
        },256);
    }
    std::vector<Id> owned;
private:
    PartitionGpu* device() const {
#ifdef AERO_WITH_CUDA
        return gpu_.get();
#else
        return nullptr;
#endif
    }
#ifdef AERO_WITH_CUDA
    std::unique_ptr<PartitionGpu> gpu_;
#endif
    PartitionStencil stencil_;
    Transfers nodes_,edges_;
    unsigned width_;
    const double* volumes_;
    const std::vector<Id>& ids_;
    const std::vector<int>& owners_;
    const Processes& world_;
    std::vector<Surface> surfaces_;
    std::vector<EulerFluxResult> flux_;
    std::vector<Id> compute_;
    std::vector<Index> offsets_,incidence_;
    residual_detail::Geometry grid_{};
};


void report_step(const Processes& world,const FlowSummary& local,unsigned dimension,
                 std::uint64_t step,double time,double cfl,double dt) {
    std::string message;
    if (world.size==1) {
        write_flow_summary(local,step,time,cfl,dt,dimension,std::cout);
        return;
    }
    world.local([&] {
        if (!local.volume) return;
        // Some MPI runtimes use a different ABI for long double than the host compiler.
        std::ostringstream s;
        s.imbue(std::locale::classic());
        s<<std::setprecision(std::numeric_limits<long double>::max_digits10);
        s<<local.volume<<' '<<local.rate_scale<<' '<<local.scaled_squares;
        for (unsigned k=0;k<dimension+3;++k)
            s<<' '<<local.field[k].low<<' '<<local.field[k].high<<' '<<local.field[k].integral;
        message=s.str();
    });
    const auto incoming=world.gather_text(std::move(message));
    world.local([&] {
        if (world.rank) return;
        FlowSummary total;
        for (const auto& record:incoming) {
            if (record.empty()) continue;
            std::istringstream s(record);
            s.imbue(std::locale::classic());
            FlowSummary part;
            s>>part.volume>>part.rate_scale>>part.scaled_squares;
            for (unsigned k=0;k<dimension+3;++k)
                s>>part.field[k].low>>part.field[k].high>>part.field[k].integral;
            if (!s) throw std::runtime_error("Invalid MPI flow report");
            total.merge(part,dimension+3);
        }
        write_flow_summary(total,step,time,cfl,dt,dimension,std::cout);
    });
}

void write_local_mesh(const MeshPartition& part,const std::string& directory) {
    part.mesh.write(directory+"/grid.mesh");
    std::ofstream out(directory+"/nodes.tsv");
    out<<"# local\tglobal\towner\n";
    for (Id i=0;i<part.nodes.size();++i) out<<i<<'\t'<<part.nodes[i]<<'\t'<<part.node_owners[i]<<'\n';
    out.close();
    if (!out) throw std::runtime_error("Cannot write "+directory+"/nodes.tsv");
}

// A native mesh is a single partition in original node and boundary order.
MeshPartition whole_mesh(const std::string& path) {
    MeshPartition part;
    part.mesh=Mesh::read(path);
    const auto m=part.mesh.view();
    part.nodes.resize(m.node_count);
    std::iota(part.nodes.begin(),part.nodes.end(),Id(0));
    part.node_owners.resize(m.node_count,0);
    part.boundary_order.resize(m.boundary_face_count);
    std::iota(part.boundary_order.begin(),part.boundary_order.end(),Id(0));
    part.global_counts={{m.node_count,m.face_count,m.cell_count}};
    part.physical_patches=part.mesh.data().patch_names.size();
    return part;
}

int run_partition(RunSettings run,const ReconstructionOptions& reconstruction,const Processes& world,
                  const LeafLayout& layout,GpuMemoryFit* gpu,bool first_attempt,bool archive) {
#ifndef AERO_WITH_CUDA
    (void)gpu;
#endif
    const bool reconstruct=reconstruction.scheme!=ReconstructionScheme::FirstOrder;
    std::unique_ptr<PartitionInput> input;
    const bool single=world.size==1;
    auto part=archive?load_partition(run.mesh_path,world,reconstruct && !single?&input:nullptr,layout):whole_mesh(run.mesh_path);
    const auto mesh=part.mesh.view();
    const unsigned width=mesh.dimension+2;
    MeshData physical;
    bool ready=true;
    world.local([&] {
        physical.dimension=mesh.dimension;
        const auto& names=part.mesh.data().patch_names;
        physical.patch_names.assign(names.begin(),names.begin()+part.physical_patches);
        if (!world.rank && first_attempt) {
            show_settings(run);
            if (single) {
                if (!run.directories.recover && std::ifstream(run.initial_path))
                    std::cout<<"Initial settings\n"<<settings_text(run.initial_path);
                if (std::ifstream(run.boundary_path)) std::cout<<"Boundary settings\n"<<settings_text(run.boundary_path);
                std::cout<<"Dimension = "<<mesh.dimension<<"  nodes = "<<mesh.node_count<<"  cells = "<<mesh.cell_count<<'\n';
                show_boundary_labels(part.mesh);
            }
            ready=prepare_boundary_file(run,physical.patch_names);
        }
    });
    if (!world.minimum(ready?1.0:0.0)) return 2;
    if (single && (!mesh.cell_count || !mesh.edge_count)) throw MeshError("The calculation requires a nonempty mesh");
    if (single && run.use_cuda) {
        const int devices=ExplicitIntegrator::cuda_device_count();
        if (!devices) {
            std::cout<<"CUDA unavailable; running on CPU.\n";
            run.use_cuda=false;
        } else {
            if (run.devices.size()!=1) throw std::invalid_argument("One calculation uses one GPU. Select one device in cuda.txt.");
            if (run.devices.front()>=devices) throw std::invalid_argument("Unavailable CUDA device "+std::to_string(run.devices.front()));
#ifdef AERO_WITH_CUDA
            const auto status=cudaSetDevice(run.devices.front());
            if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        }
    }
    CaseBoundaries bc;
    std::unique_ptr<MedianDual> dual;
    world.local([&] {
        bc=read_case_boundaries(run.boundary_path,physical);
        if (single && run.directories.interactive) std::cout<<"Accepted boundary settings\n"<<settings_text(run.boundary_path);
        dual.reset(new MedianDual(part.mesh));
    });
    const auto cv=dual->view();
    const bool periodic=!bc.periodic_pairs.empty();
    PeriodicPartition seam;
    if (periodic) seam=partition_periodic(part,cv,bc.periodic_pairs,world);
    const auto frame=momentum_frame(mesh,world,&part);
    std::unique_ptr<PartitionOperator> op;
    std::unique_ptr<OutputSchedule> output;
    std::unique_ptr<LocalFieldOutput> local_output;
    std::unique_ptr<FlowReport> report;
    const auto stages=explicit_detail::stages_for(explicit_detail::scheme_from_name(run.time_scheme));
    // The stage blend needs the start-of-step values only at owned nodes.
    // The same buffer supplies the reporter before and after the complete step.
    std::vector<double> owned_state,initial,state;
    std::uint64_t stencil_peak=0;
    {
        if (single) stage("Building reconstruction stencils");
        StencilRequests requests;
        world.local([&] {
            const auto physics=residual_options(run.reconstruction_variables,run.euler_solver,run.gamma,
                                                run.trace_ratio_min,run.trace_ratio_max);
            op.reset(new PartitionOperator(part,cv,periodic?&seam:nullptr,bc.by_patch,physics,frame,world,reconstruct?&requests:nullptr,run.use_cuda));
        });
        const auto downgraded=reconstruct?op->prepare_stencil(part,periodic?&seam:nullptr,bc.periodic_pairs,input.get(),requests,reconstruction):0;
        if (single) {
            done();
            std::cout<<"Computational nodes = "<<op->state_nodes()<<"  edges = "<<(periodic?seam.edges.size():mesh.edge_count)<<"  ";
        }
        if (!world.rank) std::cout<<"downgraded reconstruction sides = "<<downgraded<<'\n';
        input.reset();
    }
    if (!single) op->connect(run.use_cuda);
#ifdef AERO_WITH_CUDA
    GpuMemoryFit single_fit;
    if (single && run.use_cuda) gpu=&single_fit;
    if (gpu) {
        const bool compact=stages.count>1 || run.flowreport_every;
        world.local([&] {
            if (single) gpu->budget=std::numeric_limits<Id>::max();
            else query_gpu_memory(*gpu,run.mpi_gpu_memory_mib,world);
            gpu->first=part.first_leaf;
            gpu->end=part.end_leaf;
            op->reserve_gpu(bc.by_patch.size(),compact,*gpu);
        });
        if (single) {
            if (gpu->failed) throw std::runtime_error("Insufficient GPU memory for the mesh");
        } else {
            const bool retry=world.sum(gpu->failed?1:0)!=0;
            const auto lines=world.gather_text("rank "+std::to_string(world.rank)+" leaves ["+
                std::to_string(gpu->first)+","+std::to_string(gpu->end)+") required_bytes "+std::to_string(gpu->required)+
                " budget_bytes "+std::to_string(gpu->budget)+(gpu->failed?" rejected":" reserved"));
            if (!world.rank) for (const auto& line:lines) std::cout<<"MPI GPU fit: "<<line<<'\n';
            if (retry) return -1; // Release every trial allocation before querying devices again.
        }
        world.local([&] { op->layout_gpu(bc.by_patch.size(),compact); });
    }
#endif
    Id local_edges=0;
    for (Id e=0;e<mesh.edge_count;++e) if (part.node_owners[mesh.edge_begin[e]]==world.rank) ++local_edges;
    const FieldNodeSelection selection={part.nodes,part.mesh_sha256,mesh.dimension,
        {{part.global_counts[0],world.sum(local_edges),part.global_counts[1],part.global_counts[2]}}};
    const bool global=!single && std::any_of(run.output_zones.begin(),run.output_zones.end(),[](const OutputZone& z) { return z.global; });
    if (global && part.mesh_sha256.empty()) throw FieldError("Global field output needs the original mesh SHA-256; regenerate the .mdec with meshdec");
    const bool local_fields=single?!run.output_zones.empty():std::any_of(run.output_zones.begin(),run.output_zones.end(),[](const OutputZone& z) { return !z.global; });
    const std::string rank_directory="rank-"+std::to_string(world.rank)+"-of-"+std::to_string(world.size);
    if (global && local_fields) world.local([&] {
        for (const auto& z:run.output_zones) if (z.global) {
            const auto path=output_path_key(output_filename(z,1));
            if (path.substr(0,path.find('/'))==rank_directory)
                throw FieldError("Global field output overlaps rank directory: "+z.file);
        }
    });
    std::vector<OwnedNode> rows;
    std::unique_ptr<Recovery> recovery;
    if (global || run.recovery_every_steps || run.directories.recover) {
        world.local([&] {
            for (Id n=0;n<part.nodes.size();++n) if (part.node_owners[n]==world.rank) {
                const Id i=periodic?seam.node_dof[n]:n;
                rows.push_back({part.nodes[n],periodic?seam.ids[i]:part.nodes[n],i});
            }
        });
    }
    if (run.recovery_every_steps || run.directories.recover) {
        const auto hash=archive?part.mesh_sha256:FieldSnapshot::mesh_fingerprint(part.mesh);
        recovery.reset(new Recovery(run,world,part.global_counts[0],mesh.dimension,hash,rows));
    }
    world.local([&] {
        if (!single) stencil_peak=process_peak_kib();
        if (run.flowreport_every && !op->owned.empty()) {
            std::vector<double> volumes;
            for (Id i:op->owned) volumes.push_back(periodic?seam.volumes[i]:cv.volumes[i]);
            report.reset(new FlowReport(run.flowreport_every,run.gamma,mesh.dimension,volumes.size(),volumes.data()));
        }
        if (periodic) {
            // Only the compact numbering and volumes are used during stepping.
            std::vector<PeriodicPartition::Edge>().swap(seam.edges);
            std::vector<CvBoundary>().swap(seam.boundary);
            dual.reset();
        }
        if ((!run.use_cuda && stages.count>1) || report) owned_state.resize(op->owned.size()*width);
        if (op->state_nodes()>initial.max_size()/width) throw std::length_error("Distributed state too large");
        if (run.directories.recover) {
            initial.resize(op->state_nodes()*width);
            state.resize(initial.size());
            return;
        }
        std::ifstream in(run.initial_path);
        if (!in) throw std::runtime_error("Cannot open "+run.initial_path);
        state=read_initial(part.mesh,run.gamma,in,run.directories.input,nullptr,single?nullptr:&selection);
        if (!periodic) initial.swap(state);
        initial.resize(op->state_nodes()*width);
        if (periodic) {
            for (Id n=0;n<part.nodes.size();++n) {
                const Id i=seam.node_dof[n];
                if (part.nodes[n]==seam.ids[i] && part.node_owners[n]==world.rank)
                    std::copy_n(state.data()+n*width,width,initial.data()+i*width);
            }
        }
    });
    if (run.directories.recover) {
        recovery->restore(state,initial);
        if (!single) op->halo_host(state);
    }
    if (!single) op->halo_host(initial);
    const auto& base=recovery?recovery->fields_directory():run.directories.output;
    world.local([&] {
        if (!run.directories.recover && periodic) for (Id n=0;n<part.nodes.size();++n) {
            const Id i=seam.node_dof[n];
            check_periodic_initial(initial.data()+i*width,state.data()+n*width,width,seam.ids[i],part.nodes[n]);
        }
        if (!run.directories.recover) state=initial;
#ifdef AERO_WITH_CUDA
        if (gpu) {
            op->upload_state(state,initial);
            if (run.mpi_timing) PartitionGpu::synchronize();
        }
#endif
        const std::string directory=base+"/"+rank_directory;
        if (!periodic && local_fields && op->state_nodes()!=mesh.node_count) {
            seam.node_dof.resize(mesh.node_count);
            std::iota(seam.node_dof.begin(),seam.node_dof.end(),Id(0));
        }
        output.reset(new OutputSchedule(run.output_zones,mesh.dimension));
        if (local_fields) {
            local_output.reset(new LocalFieldOutput(single?base:directory+"/fields",part.mesh,run.gamma,
                                                   std::move(seam.node_dof),op->state_nodes()));
            if (!single) {
                prepare_zone_directories(directory+"/fields",run.output_zones,false);
                write_local_mesh(part,directory);
            }
        }
        if (single) {
            auto fields=run;
            fields.directories.output=base;
            prepare_output_directories(fields);
        }
        if (!world.rank && global) prepare_zone_directories(base,run.output_zones,true);
    });
    if (!single) write_partition_report(world,op->work(part.end_leaf-part.first_leaf),stencil_peak,std::cout);
    if (single) std::cout<<"Active backend = "<<(run.use_cuda?"CUDA device "+std::to_string(run.devices.front()):"CPU")<<'\n';
    else if (!world.rank) std::cout<<"Active backend = "<<(run.use_cuda?"CUDA":"CPU")<<" (MPI), processes = "<<world.size<<'\n';
    StepOperations operations;
    operations.cfl_step=[&](SolverTimings& timing) {
        op->assemble_fluxes(state,initial,timing);
        WallTimer timer(timing.counter(SolverTimings::Cfl));
        return op->cfl_step(run.cfl);
    };
    FlowSummary summary;
    operations.advance=[&](const TimeStep& step,SolverTimings& timing) {
        const auto assemble=[&](double) { op->assemble_fluxes(state,initial,timing); };
        const auto update=[&](explicit_detail::Stage stage,bool first,bool last) {
            world.local([&] {
                if (first && step.reporting && report) {
                    op->copy_owned(state,owned_state);
                    report->capture(step.number,owned_state);
                }
                {
                    WallTimer timer(timing.counter(SolverTimings::Update));
                    op->update_owned(stage,owned_state,step.dt,state,first && stages.count>1);
                }
                if (last && step.reporting && report) {
                    op->copy_owned(state,owned_state);
                    summary=report->measure(step.number,owned_state);
                }
            });
        };
        const auto exchange=[&] {
            if (single) return;
            op->halo(state,timing.counter(SolverTimings::Halo));
        };
        explicit_detail::advance_stages(stages,step.start,step.dt,true,assemble,update,exchange);
    };
    operations.output_state=[&]() -> const std::vector<double>& {
        if (run.use_cuda) world.local([&] { op->download(state); });
        return state;
    };
    operations.report=[&](const TimeStep& step) {
        report_step(world,summary,mesh.dimension,step.number,step.end,run.cfl,step.dt);
    };
    operations.write_fields=[&](const TimeStep& step,const std::vector<double>& snapshot) {
        output->write_due(step.number,step.end,[&](const OutputZone& zone,const std::vector<unsigned>& columns) {
            if (!single && zone.global) write_global_fields(world,base,zone,columns,selection,rows,run.gamma,step.number,step.end,snapshot);
            else world.local([&] { local_output->write(zone,columns,step.number,step.end,snapshot); });
        });
    };
    integrate_time(run,world,*output,recovery.get(),initial,operations);
    return 0;
}
}

int calculate_case(RunSettings run,const ReconstructionOptions& reconstruction,const Processes& world,bool archive) {
#ifdef AERO_WITH_CUDA
    if (world.size>1 && run.use_cuda) {
        select_partition_gpu(world,run.devices);
        GpuPlacement gpu;
        int automatic=run.mpi_rank_weights.empty()?1:0;
        world.broadcast(&automatic,1,0);
        if (automatic) {
            // A common reference gas keeps the calibration independent of the input fields.
            const auto physics=residual_options(run.reconstruction_variables,run.euler_solver,1.4);
            world.local([&] { query_gpu_memory(gpu.local,run.mpi_gpu_memory_mib,world); });
            run.mpi_rank_weights=PartitionGpu::measure_weights(world,physics,gpu.local.budget);
        }
        for (bool first=true;;first=false) {
            world.local([&] { query_gpu_memory(gpu.local,run.mpi_gpu_memory_mib,world); });
            const auto plan=plan_partition(run,world,&gpu);
            const int result=run_partition(run,reconstruction,world,plan,&gpu.local,first,archive);
            if (result>=0) return result;
        }
    }
#endif
    return run_partition(run,reconstruction,world,plan_partition(run,world),nullptr,true,archive);
}
}

