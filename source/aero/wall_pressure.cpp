#include "partition_operator.h"
#include "../io/settings.h"
#include "../parallel/parallel_for.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_work.h"
#include "../parallel/device_check.h"
#endif
#include <iostream>
#include <sstream>

namespace aero {
namespace {
using Id = MeshIndex;
const Id absent = std::numeric_limits<Id>::max();

// Скоростной напор берём из таблицы сопла или заданного опорного
// состояния. При нескольких входах используем наибольшее значение.
double inlet_dynamic_pressure(const BoundaryPatches& bc,unsigned dim) {
    double q=0.;
    for (const auto& rule:bc.free_rules) {
        const auto& s=rule.reference;
        const unsigned velocities=((1u<<dim)-1u)<<2;
        if ((s.mask & (1u|velocities))==(1u|velocities))
            q=std::max(q,.5*s.primitive[0]*euler_detail::dot(dim,s.primitive+2,s.primitive+2));
    }
    for (const auto& profile:bc.profiles) {
        if (!(profile.columns & ProfileRho) || !(profile.columns & Axial)) continue;
        for (const auto& row:profile.rows) {
            const double radial=(profile.columns & Radial) ? row.radial : 0.;
            q=std::max(q,.5*row.density*(row.axial*row.axial+radial*radial));
        }
    }
    return q>0. ? q : .5; // Нормировка набегающего потока rho_inf=U_inf=1.
}

template<class T> T diagnostic_value(const T* address,bool device) {
#ifdef AERO_WITH_KOKKOS_GPU
    if (device) {
        T result;
        gpu::checked(gpu::copy(&result,address,sizeof(T),gpu::Copy::ToHost),"Pressure guard diagnostic: ");
        return result;
    }
#else
    (void)device;
#endif
    return *address;
}
}

void PartitionOperator::prepare_pressure_guard(const MeshPartition& part,const PeriodicPartition* periodic,
                                              const BoundaryPatches& bc,const RunSettings& run) {
    // Номера слоёв согласуем через владельцев, включая периодические швы:
    // 1 — стенка, 2 — датчик рядом со стенкой, 3 — только сосед датчика.
    std::vector<double> layer(state_nodes(),0.);
    for (Id f=grid_.edges; f<grid_.faces; ++f) {
        const auto& face=surfaces_[f];
        if (solid_wall(bc.by_patch[face.patch].type())) layer[face.begin]=1.;
    }
    if (!world_.sum(static_cast<Id>(std::count(layer.begin(),layer.end(),1.)))) return;
    PartitionTransfers setup(nodes_,1);
    if (world_.size>1) setup.allocate(false);
    for (unsigned level=1;level<=3;++level) {
        if (world_.size>1) setup.transfer(world_,{1,layer.data(),nullptr});
        if (level==3) break;
        // Пишет только владелец. Новый уровень не может продолжить сам себя.
        for (Id n:owned) if (!layer[n]) {
            for (Id k=offsets_[n]; k<offsets_[n+1]; ++k) {
                const auto f=incidence_[k].face();
                if (f>=grid_.edges) continue;
                const auto& edge=surfaces_[f];
                if (layer[edge.begin==n?edge.end:edge.begin]==level) {
                    layer[n]=level+1;
                    break;
                }
            }
        }
    }
    wall_pressure_.reset(new WallPressure);
    auto& w=*wall_pressure_;
    w.reference=inlet_dynamic_pressure(bc,grid_.dim);
    if (run.mach>0.) w.ambient_pressure=1./(run.gamma*run.mach*run.mach);
    if (!world_.rank) std::cout<<"Wall pressure guard: reference dynamic pressure = "<<w.reference<<std::endl;
    std::vector<Id> slot(layer.size(),absent);
    for (Id n=0;n<layer.size();++n) if (layer[n]) {
        slot[n]=w.nodes.size();
        w.nodes.push_back(n);
        w.sites.push_back({0,n,layer[n]<=2,false});
    }
    for (Id row=0;row<owned.size();++row) if (layer[owned[row]]) {
        auto& site=w.sites[slot[owned[row]]];
        site.row=row;
        site.owner=true;
    }
    w.points.resize(w.nodes.size());
    auto mesh=part.mesh.view();
    for (Id i=0;i<mesh.node_count;++i) {
        const Id n=periodic ? periodic->node_dof[i] : i;
        if (n<slot.size() && slot[n]!=absent) w.points[slot[n]]=mesh.coordinate(i);
    }
    w.offsets.push_back(0);
    for (const auto& site:w.sites) {
        if (site.owner) for (Id k=offsets_[site.node];k<offsets_[site.node+1];++k) {
            const auto f=incidence_[k].face();
            if (f>=grid_.edges) continue;
            const auto& e=surfaces_[f];
            const auto j=slot[e.begin==site.node?e.end:e.begin];
            if (j!=absent && w.sites[j].monitor) w.neighbours.push_back(j);
        }
        w.offsets.push_back(w.neighbours.size());
    }
    for (Id n=0;n/32<=(grid_.nodes+31)/32;n+=32)
        w.blocks.push_back(std::lower_bound(w.nodes.begin(),w.nodes.end(),n)-w.nodes.begin());
    if (world_.size>1) {
        w.halo.reset(new PartitionTransfers(world_.size,1));
        for (int p=0;p<world_.size;++p) {
            for (Id n:nodes_.exports[p]) if (layer[n] && layer[n]<=2) w.halo->exports[p].push_back(slot[n]);
            for (Id n:nodes_.imports[p]) if (layer[n] && layer[n]<=2) w.halo->imports[p].push_back(slot[n]);
        }
        w.halo->allocate(run.use_cuda);
    }
    w.history.resize(w.nodes.size()); w.signals.resize(w.nodes.size()); w.caps.resize(w.nodes.size());
    w.active_sites=w.sites.data(); w.active_history=w.history.data();
    w.active_signals=w.signals.data(); w.active_caps=w.caps.data();
    w.active_offsets=w.offsets.data(); w.active_neighbours=w.neighbours.data();
    w.lookup={w.blocks.data(),w.nodes.data(),w.active_caps};
}

void PartitionOperator::update_pressure_guard(double cfl) {
    auto& w=*wall_pressure_;
    const Id count=w.sites.size();
    const bool gpu_active=device()!=nullptr;
    partition_detail::PressureRate rate{grid_,flux_.data(),volumes_,accepted_field_,nullptr,nullptr};
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu_) {
        const auto& g=gpu_->active_grid();
        rate={g.geometry,g.flux,g.volumes,g.state,nullptr,nullptr};
    }
#endif
    const partition_detail::ObservePressure read{rate,w.active_sites,w.active_history,w.active_signals,w.reference,w.first};
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) { device_work::check(device_work::aero_device_observe_pressure(read,count)); return; }
#endif
        parallel_for(count,read);
    });
    if (w.halo) w.halo->transfer(world_,{1,w.active_signals,nullptr},device());
    const partition_detail::UpdatePressureCaps limit{w.active_sites,w.active_offsets,w.active_neighbours,
        w.active_signals,steady_steps_->values,w.active_caps,cfl_grid(),accepted_fraction_,cfl,w.first};
    Id event=absent;
    world_.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (gpu_) device_work::check(device_work::aero_device_pressure_caps(limit,count,&event));
        else
#endif
        {
            host_parallel::ensure_runtime();
            Kokkos::parallel_reduce("Aero: wall pressure limits",host_parallel::Range(0,count),limit,Kokkos::Min<Id>(event));
        }
        if (!event) throw std::domain_error("Nonfinite pressure or pressure residual in local time stepping");
    });
    w.first=false;
    // Только при новом ограничении скачиваем диагностические значения одного узла.
    const Id id=event==absent ? absent : ids_[w.nodes[event-1]];
    const Id first=world_.size==1 ? id : world_.first_id(id);
    if (first==absent) return;
    std::ostringstream message;
    world_.local([&] {
        if (id!=first) return;
        const auto i=event-1;
        const auto h=diagnostic_value(w.active_history+i,gpu_active);
        const auto cap=diagnostic_value(w.active_caps+i,gpu_active);
        const auto dt=accepted_fraction_*diagnostic_value(steady_steps_->values+w.nodes[i],gpu_active);
        const auto& x=w.points[i];
        message<<"Wall pressure guard: node "<<id<<" xyz ("<<x[0]<<", "<<x[1]<<", "<<x[2]<<")"
               <<" p "<<h.previous_pressure<<" -> "<<h.pressure
               <<" dp/dt "<<h.previous_rate<<" -> "<<h.rate<<" accepted dt "<<dt<<" new cap "<<cap;
        if (w.ambient_pressure>0.) message<<" Cp "<<(h.previous_pressure-w.ambient_pressure)/.5
                                         <<" -> "<<(h.pressure-w.ambient_pressure)/.5;
    });
    for (const auto& line:world_.gather_text(message.str())) if (!line.empty()) std::cout<<line<<std::endl;
}
}
