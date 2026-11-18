#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <cstdlib>
#endif
#include "scheduled_output.h"
#include "../eulersolver/euler.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace aero {
namespace {
// Stage in the destination directory so a failed write preserves the previous snapshot.
void publish(const FieldSnapshot& snapshot,const std::string& path,FieldFormat format,bool replace) {
    const auto directory=path.substr(0,path.find_last_of("/\\")+1);
#ifdef _WIN32
    char temp[MAX_PATH];
    if (!GetTempFileNameA(directory.c_str(),"aer",0,temp))
        throw FieldError("Cannot create temporary file for "+path);
    const char* temporary=temp;
#else
    std::string pattern=directory+".aero-fields-XXXXXX";
    const int fd=mkstemp(&pattern[0]);
    if (fd<0) throw FieldError("Cannot create temporary file for "+path);
    const char* temporary=pattern.c_str();
    if (close(fd)!=0) {
        std::remove(temporary);
        throw FieldError("Cannot close temporary file for "+path);
    }
#endif
    try {
        snapshot.write(temporary,format);
#ifdef _WIN32
        if (!MoveFileExA(temporary,path.c_str(),replace ? MOVEFILE_REPLACE_EXISTING : 0))
            throw FieldError("Cannot publish "+path+" (Windows error "+std::to_string(GetLastError())+")");
#else
        if (replace) {
            if (std::rename(temporary,path.c_str())!=0) throw FieldError("Cannot replace "+path);
        } else {
            if (link(temporary,path.c_str())!=0) throw FieldError("Cannot publish "+path);
            std::remove(temporary);
        }
#endif
    } catch (...) { std::remove(temporary); throw; }
}
}

ScheduledOutput::ScheduledOutput(const std::vector<OutputZone>& zones,const std::string& directory,
                                 const Mesh& mesh,double gamma,const std::vector<std::uint64_t>& node_dof)
    :directory_(directory),gamma_(gamma) {
    validate_output_zones(zones);
    if (zones.empty()) return;
    if (directory.empty() || !std::isfinite(gamma) || gamma<=1)
        throw std::invalid_argument("output: invalid directory or gamma");
    metadata_=FieldSnapshot(mesh);
    const auto count=metadata_.counts[0];
    const unsigned d=metadata_.dimension;
    if (count>std::vector<double>().max_size()/(d+2)) throw FieldError("output: too many nodes");
    nodes_=static_cast<std::size_t>(count);
    if (!node_dof.empty()) {
        if (node_dof.size()!=nodes_) throw FieldError("output: wrong node map length");
        const auto maximum=*std::max_element(node_dof.begin(),node_dof.end());
        if (maximum>=count) throw FieldError("output: invalid node map");
        nodes_=static_cast<std::size_t>(maximum+1);
        node_dof_=node_dof;
    }
    const std::vector<std::string> names={"rho","ux","uy","uz","p","E"};
    for (const auto& config:zones) {
        Zone z;
        z.config=config;
        for (const auto& var:config.variables) {
            const auto column=static_cast<unsigned>(std::find(names.begin(),names.end(),var)-names.begin());
            if (column<=3 && column>d) throw FieldError("["+config.name+"] "+var+" exceeds mesh dimension");
            z.columns.push_back(column);
        }
        zones_.push_back(z);
    }
}
double ScheduledOutput::Zone::deadline() const {
    return static_cast<double>(next_tick)*config.every_time;
}
bool ScheduledOutput::Zone::due(std::uint64_t step,double time) const {
    if (step<=last_step) return false;
    if (config.every_steps) return step%config.every_steps==0;
    const double target=deadline();
    if (!std::isfinite(target)) return false;
    // Merge deadlines that differ only by roundoff, e.g. 3 * 0.1 and 0.3.
    const double tolerance=std::min(config.every_time*0.25,4*std::numeric_limits<double>::epsilon()*target);
    if (time-target>tolerance) throw FieldError("["+config.name+"] missed output time; use limit_dt before stepping");
    return target-time<=tolerance;
}
double ScheduledOutput::limit_dt(double time,double candidate) const {
    if (!(time>=0 && std::isfinite(time) && candidate>0 && std::isfinite(candidate)))
        throw FieldError("output: invalid time or proposed dt");
    for (const auto& zone:zones_) if (zone.config.every_time>0) {
        const double remaining=zone.deadline()-time;
        if (!(remaining>0)) throw FieldError("["+zone.config.name+"] pending output at current time");
        candidate=std::min(candidate,remaining);
    }
    return candidate;
}
bool ScheduledOutput::due(std::uint64_t step,double time) const {
    if (!(time>=0 && std::isfinite(time))) throw FieldError("output: invalid accepted time");
    bool any=false;
    for (const auto& zone:zones_) if (zone.due(step,time)) any=true;
    return any;
}
void ScheduledOutput::write(std::uint64_t step,double time,const std::vector<double>& state) {
    if (!due(step,time)) return;
    const unsigned width=metadata_.dimension+2;
    if (state.size()!=nodes_*width) throw FieldError("output: incorrect conservative state size");
    for (auto& zone:zones_) {
        if (!zone.due(step,time)) continue;
        auto snap=metadata_;
        snap.step=step;
        snap.time=time;
        const auto count=static_cast<std::size_t>(snap.counts[0]);
        for (const auto& name:zone.config.variables) {
            snap.fields.emplace_back();
            auto& field=snap.fields.back();
            field.name=name;
            field.location=FieldLocation::Node;
            field.components={"value"};
            field.values.resize(count);
        }
        parallel_for(count,[&](std::size_t i) {
            const auto node=node_dof_.empty() ? i : static_cast<std::size_t>(node_dof_[i]);
            const auto* u=state.data()+node*width;
            EulerState primitive;
            if (euler_state(snap.dimension,u,gamma_,&primitive)!=EulerStatus::Success)
                throw FieldError("output: invalid Euler state at node "+std::to_string(node));
            const double values[]={u[0],primitive.velocity[0],primitive.velocity[1],primitive.velocity[2],
                                   primitive.pressure,u[width-1]};
            for (std::size_t k=0;k<zone.columns.size();++k) snap.fields[k].values[i]=values[zone.columns[k]];
        });
        auto name=output_filename(zone.config,step);
        std::replace(name.begin(),name.end(),'\\','/');
        publish(snap,directory_+"/"+name,zone.config.format,!zone.config.multi);
        zone.last_step=step;
        if (zone.config.every_time>0) ++zone.next_tick;
    }
}
}
