#include "partition_output.h"
#include "mpi_fields.h"
#include "../io/flow_values.h"
#include "../io/sha256.h"
#include <cstring>
#include <set>
#include "startup.h"
#include <fstream>
#include <numeric>

namespace aero {
namespace {
// Каждое ребро учитывается на процессе, которому принадлежит его начало.
FieldNodeSelection select_field_nodes(const MeshPartition& part, const Processes& world) {
    const auto mesh = part.mesh.view();
    MeshIndex local_edges = 0;
    for (MeshIndex edge = 0; edge < mesh.edge_count; ++edge) {
        if (part.node_owners[mesh.edge_begin[edge]] == world.rank) {
            ++local_edges;
        }
    }
    return {part.nodes,
            part.mesh_sha256,
            mesh.dimension,
            {{part.global_counts[0], world.sum(local_edges), part.global_counts[1], part.global_counts[2]}}};
}

// Рядом с локальными полями сохраняем сетку и соответствие локальных узлов исходным.
void write_local_mesh(const MeshPartition& part, const std::string& directory) {
    part.mesh.write(directory + "/grid.mesh");
    std::ofstream out(directory + "/nodes.tsv");
    out << "# local\tglobal\towner\n";
    for (MeshIndex node = 0; node < part.nodes.size(); ++node) {
        out << node << '\t' << part.nodes[node] << '\t' << part.node_owners[node] << '\n';
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Cannot write " + directory + "/nodes.tsv");
    }
}
}

PartitionFileLayout::PartitionFileLayout(const RunSettings& run, const MeshPartition& part,
                                         const PeriodicPartition* seam, const Processes& world)
    : selection(select_field_nodes(part, world)),
      rank_directory("rank-" + std::to_string(world.rank) + "-of-" + std::to_string(world.size)) {
    for (const auto& zone : run.output_zones) {
        if (world.size > 1 && zone.global) {
            global_fields = true;
        } else {
            local_fields = true;
        }
    }
    if (global_fields && part.mesh_sha256.empty()) {
        throw FieldError(
            "Global field output needs the original mesh SHA-256; regenerate the .mdec with meshdec");
    }
    if (global_fields && local_fields) {
        world.local([&] {
            for (const auto& zone : run.output_zones) {
                if (!zone.global) {
                    continue;
                }
                const auto path = output_path_key(output_filename(zone, 1));
                if (path.substr(0, path.find('/')) == rank_directory) {
                    throw FieldError("Global field output overlaps rank directory: " + zone.file);
                }
            }
        });
    }
    // Итоговая контрольная точка всегда использует исходные номера узлов.
    world.local([&] {
        for (MeshIndex node = 0; node < part.nodes.size(); ++node) {
            if (part.node_owners[node] != world.rank) {
                continue;
            }
            const auto slot = seam ? seam->node_dof[node] : node;
            const auto representative = seam ? seam->ids[slot] : part.nodes[node];
            rows.push_back({part.nodes[node], representative, slot});
        }
    });
}

PartitionOutput::PartitionOutput(const RunSettings& run, const MeshPartition& part,
                                 const PartitionFileLayout& layout, const Processes& world,
                                 const std::string& directory, std::vector<MeshIndex>& node_dof,
                                 MeshIndex state_nodes, const BoundaryPatches& patches)
    : schedule(run.output_zones, layout.selection.dimension), world_(world), layout_(layout),
      directory_(directory), background_(run.background), gamma_(run.gamma) {
    const bool single = world.size == 1;
    const auto mesh=part.mesh.view();
    for (const auto& zone:run.output_zones) {
        if (zone.wall_label.empty() || walls_.count(zone.wall_label)) continue;
        const auto& names=part.mesh.data().patch_names;
        const auto last=names.begin()+part.physical_patches;
        const auto found=std::find(names.begin(),last,zone.wall_label);
        if (found==last) throw FieldError("Unknown wall_label: "+zone.wall_label);
        const auto patch=static_cast<MeshIndex>(found-names.begin());
        if (!solid_wall(patches.by_patch[patch].type()))
            throw FieldError("Cp needs a solid wall: "+zone.wall_label);
        if (background_.active()) {
            // В линейной системе давление уже записано как отклонение от фона.
            dynamic_pressure_=.5*background_.rho*euler_detail::dot(mesh.dimension,background_.velocity,background_.velocity);
            if (!(dynamic_pressure_>0.)) throw FieldError("Cp needs a nonzero background speed");
        } else {
            if (!(run.mach>0.)) throw FieldError("Cp needs gas.txt: mach > 0");
            reference_pressure_=1./(gamma_*run.mach*run.mach);
            dynamic_pressure_=.5; // В принятых безразмерных единицах rho_inf = U_inf = 1.
        }
        // Одна вершина может входить в несколько граней выбранной метки.
        std::set<MeshIndex> nodes;
        for (MeshIndex f=0;f<mesh.boundary_face_count;++f) if (mesh.boundary_patch[f]==patch)
            for (auto n:mesh.face_nodes(mesh.boundary_faces[f])) nodes.insert(n);
        auto& wall=walls_[zone.wall_label];
        for (auto n:nodes) wall.push_back({part.nodes[n],node_dof.empty()?n:node_dof[n],mesh.coordinate(n)});
    }
    if (layout.local_fields) {
        const auto node_count = part.mesh.view().node_count;
        if (node_dof.empty() && state_nodes != node_count) {
            // Дополнительные узлы шаблона не входят в записываемую сетку.
            node_dof.resize(node_count);
            std::iota(node_dof.begin(), node_dof.end(), MeshIndex(0));
        }
        const auto rank_path = directory + "/" + layout.rank_directory;
        const auto fields_path = single ? directory : rank_path + "/fields";
        local_ = LocalFieldOutput(fields_path, part.mesh, run.gamma, std::move(node_dof), state_nodes, background_);
        if (!single) {
            prepare_zone_directories(fields_path, run.output_zones, false);
            write_local_mesh(part, rank_path);
        }
    }
    if (single) {
        auto output_run = run;
        output_run.directories.output = directory;
        prepare_output_directories(output_run);
    }
    if (!world.rank && layout.global_fields) {
        prepare_zone_directories(directory, run.output_zones, true);
    }
}

// Для Cp в файл попадают только узлы указанной стенки. На общей границе
// двух разделов один и тот же исходный номер попадает в общий снимок один раз.
void PartitionOutput::write_wall(const OutputZone& zone, std::uint64_t step, double time,
                                 const std::vector<double>& state) {
    struct Row {MeshIndex id; Coordinate xyz; double cp;};
    const bool global=zone.global && world_.size>1;
    const auto dim=layout_.selection.dimension;
    std::vector<Row> rows;
    world_.local([&] {
        for (const auto& n:walls_.at(zone.wall_label)) {
            const double* u=state.data()+n.slot*(dim+2);
            const double p=output_values(dim,u,gamma_,n.slot,background_)[diagnostic::Pressure];
            rows.push_back({n.id,n.point,(p-reference_pressure_)/dynamic_pressure_});
        }
    });
    if (global) {
        Processes::Messages messages;
        world_.local([&] {
            messages.resize(world_.size);
            if (!rows.empty()) messages[0].assign(reinterpret_cast<const char*>(rows.data()),rows.size()*sizeof(Row));
        });
        const auto received=world_.exchange(messages);
        world_.local([&] {
            rows.clear();
            if (world_.rank) return;
            for (const auto& bytes:received) {
                const auto first=rows.size();
                if (bytes.size()%sizeof(Row)) throw FieldError("Invalid surface record");
                rows.resize(first+bytes.size()/sizeof(Row));
                if (!bytes.empty()) std::memcpy(rows.data()+first,bytes.data(),bytes.size());
            }
        });
    }
    world_.local([&] {
        if (global && world_.rank) return;
        std::sort(rows.begin(),rows.end(),[](const Row& a,const Row& b){return a.id<b.id;});
        rows.erase(std::unique(rows.begin(),rows.end(),[](const Row& a,const Row& b){return a.id==b.id;}),rows.end());
        FieldSnapshot snap;
        snap.dimension=dim; snap.counts[0]=rows.size(); snap.step=step; snap.time=time;
        for (const char* label:{"node_id","x","y","z","Cp"}) snap.fields.push_back(output_field(label));
        // Отпечаток относится к выборке точек, его можно сверять между снимками и разбиениями.
        io_detail::Sha256Buffer hash;
        std::ostream key(&hash);
        key<<zone.wall_label<<'\n';
        for (const auto& row:rows) {
            key.write(reinterpret_cast<const char*>(&row.id),sizeof(row.id));
            key.write(reinterpret_cast<const char*>(row.xyz.values),sizeof(row.xyz.values));
            snap.fields[0].values.push_back(static_cast<double>(row.id));
            for (unsigned k=0;k<3;++k) snap.fields[k+1].values.push_back(row.xyz[k]);
            snap.fields[4].values.push_back(row.cp);
        }
        snap.mesh_sha256=hash.hex_digest();
        const auto directory=global || world_.size==1?directory_:directory_+"/"+layout_.rank_directory+"/fields";
        StagedFieldFile file(directory+"/"+output_filename(zone,step));
        snap.write(file.temporary(),zone.format);
        file.publish(!zone.multi);
    });
}

void PartitionOutput::write(std::uint64_t step, double time, const std::vector<double>& state, bool final_field,const diagnostic::Fields& scalars) {
    schedule.write_due(step, time, [&](const OutputZone& zone, const std::vector<unsigned>& columns) {
        if (!zone.wall_label.empty()) { write_wall(zone,step,time,state); return; }
        if (world_.size > 1 && zone.global) {
            write_global_fields(world_, directory_, zone, columns, layout_.selection, layout_.rows, gamma_,
                                step, time, state, background_,scalars);
        } else {
            world_.local([&] { local_.write(zone, columns, step, time, state,scalars); });
        }
    }, final_field);
}
}
