#ifndef AERO_PARTITION_KERNELS_H
#define AERO_PARTITION_KERNELS_H
#include "../parallel/kernel_math.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"
#include "../mesh/mesh_index.h"

namespace aero {
namespace partition_detail {
using residual_detail::Index;
// Указатели этого представления действительны на устройстве, выбранном данным MPI-процессом.
struct Grid {
    residual_detail::Geometry geometry;
    const MeshIndex *computed, *owned;
    Index computed_count, owned_count;
    const double* volumes;
    double *initial, *state, *original;
    EulerFluxResult* flux;
    Index* bad;
};
// Обновляет один собственный узел прямо в массиве состояния. k — его номер среди узлов процесса.
AERO_HOST_DEVICE inline bool update_node(const residual_detail::Geometry& geometry,
                                         const EulerFluxResult* flux, const double* volumes,
                                         const MeshIndex* owned, double* state, double* original, Index k,
                                         explicit_detail::Stage stage, double dt, bool save_original) {
    const Index node = owned ? owned[k] : k;
    const unsigned width = geometry.components();
    double residual[5];
    // Сначала собираем всю строку выходящих потоков из готового буфера стадии.
    if (!residual_detail::assemble_node_residual(geometry, node, flux, residual))
        return false;

    for (unsigned c = 0; c < width; ++c) {
        // В поле есть и узлы соседей. В компактном снимке позиция k считается только среди собственных узлов.
        double& current = state[node * width + c];
        const Index saved_entry = k * width + c;
        if (save_original)
            original[saved_entry] = current;
        // На стадиях со смешиванием читаем U_n. Для Эйлера буфер снимка может отсутствовать.
        const double step_start = stage.uses_step_start() ? original[saved_entry] : current;
        if (!explicit_detail::advance_component(stage, step_start, current, residual[c], volumes[node], dt,
                                                current))
            return false;
    }
    return true;
}
AERO_HOST_DEVICE inline void transfer_record(MeshIndex i, unsigned width, double* state,
                                             EulerFluxResult* flux, double* record, bool edges, bool pack) {
    double* values = edges ? flux[i].flux : state + i * width;
    for (unsigned v = 0; v < width; ++v) {
        if (pack)
            record[v] = values[v];
        else
            values[v] = record[v];
    }
    if (edges) {
        if (pack)
            record[width] = flux[i].max_wave_speed;
        else
            flux[i].max_wave_speed = record[width];
    }
}
void update(Grid grid, explicit_detail::Stage stage, double dt, bool save_original);
void transfer(Grid grid, const MeshIndex* ids, Index count, double* records, bool edges, bool pack);
}
}
#endif
