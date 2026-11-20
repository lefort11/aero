#include "partition_output.h"
#include "mpi_fields.h"
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
    // STOP может запросить recovery вне расписания. Сохраняем соответствие строк до передачи периодической нумерации выводу.
    if (!global_fields && !run.control_every && !run.recovery_every_steps && !run.directories.recover) {
        return;
    }
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
                                 MeshIndex state_nodes)
    : schedule(run.output_zones, layout.selection.dimension), world_(world), layout_(layout),
      directory_(directory), background_(run.background), gamma_(run.gamma) {
    const bool single = world.size == 1;
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

void PartitionOutput::write(std::uint64_t step, double time, const std::vector<double>& state, bool final_field) {
    schedule.write_due(step, time, [&](const OutputZone& zone, const std::vector<unsigned>& columns) {
        if (world_.size > 1 && zone.global) {
            write_global_fields(world_, directory_, zone, columns, layout_.selection, layout_.rows, gamma_,
                                step, time, state, background_);
        } else {
            world_.local([&] { local_.write(zone, columns, step, time, state); });
        }
    }, final_field);
}
}
