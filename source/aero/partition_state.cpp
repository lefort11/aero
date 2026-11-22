#include "partition_state.h"
#include "../initial/initial.h"
#include "../io/settings.h"
#include <algorithm>
#include <fstream>

namespace aero {
void PartitionState::prepare(const RunSettings& run, const MeshPartition& part, const PeriodicPartition* seam,
                             const FieldNodeSelection& selection, MeshIndex state_nodes,
                             const Processes& world) {
    const unsigned width = selection.dimension + 2; // Плотность, компоненты импульса и полная энергия.
    if (state_nodes > initial.max_size() / width) {
        throw std::length_error("Distributed state too large");
    }
    const auto value_count = state_nodes * width;
    if (run.directories.recover) {
        initial.resize(value_count);
        current.resize(value_count);
        return;
    }

    std::ifstream input(run.initial_path);
    if (!input) {
        throw std::runtime_error("Cannot open " + run.initial_path);
    }
    const auto* selected_nodes = world.size == 1 ? nullptr : &selection;
    current = read_initial(part.mesh, run.gamma, input, run.directories.input, nullptr, selected_nodes,
                           run.background, &run.input_warnings, run.mach, &initial_checkpoint);
    if (!initial_checkpoint.empty()) {
        // Загрузчик recovery запишет представителей сразу в расчётной нумерации.
        initial.resize(value_count);
        return;
    }
    if (!seam) {
        initial.swap(current); // Обычная сетка уже использует локальный порядок узлов.
    }
    initial.resize(value_count); // Оставляем место и для удалённых узлов шаблона реконструкции.
    if (!seam) {
        return;
    }

    // Из всех периодических копий начальное значение задаёт процесс-владелец представителя.
    for (MeshIndex node = 0; node < part.nodes.size(); ++node) {
        const auto slot = seam->node_dof[node];
        if (part.nodes[node] == seam->ids[slot] && part.node_owners[node] == world.rank) {
            std::copy_n(current.data() + node * width, width, initial.data() + slot * width);
        }
    }
}

void PartitionState::start_from_initial(const MeshPartition& part, const PeriodicPartition* seam) {
    if (seam && initial_checkpoint.empty()) {
        const unsigned width = part.mesh.view().dimension + 2;
        for (MeshIndex node = 0; node < part.nodes.size(); ++node) {
            const auto slot = seam->node_dof[node];
            check_periodic_initial(initial.data() + slot * width, current.data() + node * width, width,
                                   seam->ids[slot], part.nodes[node]);
        }
    }
    current = initial;
}
}
