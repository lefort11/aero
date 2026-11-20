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
template <unsigned fixed_width = 0>
AERO_HOST_DEVICE inline bool update_node(const residual_detail::Geometry& geometry,
                                         const EulerFluxResult* flux, const double* volumes,
                                         const MeshIndex* owned, double* state, double* original, Index k,
                                         explicit_detail::Stage stage, double dt, bool save_original) {
    const Index node = owned ? owned[k] : k;
    const unsigned width = fixed_width ? fixed_width : geometry.components();
    double residual[5];
    // Сначала собираем всю строку выходящих потоков из готового буфера стадии.
    if (!residual_detail::assemble_node_residual<fixed_width>(geometry, node, flux, residual))
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
    // Без списка owned обрабатываем всё поле. Пока строка в кэше, готовим газ для следующей стадии.
    // В разделённой сетке сначала должен закончиться обмен состояниями с соседями.
    if (!owned && geometry.packed.gas)
        residual_detail::decode_nodal_gas(geometry.dim, state, geometry.options.gamma, node, geometry.packed.gas);
    return true;
}
// Вид передаваемой записи. Обычное поле имеет width последовательных чисел.
 // Для потока к ним добавляется оценка скорости, хранящаяся отдельно в EulerFluxResult.
struct RecordField {
    unsigned width;
    double* values;
    EulerFluxResult* flux;
    AERO_HOST_DEVICE unsigned stride() const { return width + unsigned(flux != nullptr); }
    AERO_HOST_DEVICE void copy_record(MeshIndex i, MeshIndex k, const double* input, double* output) const {
        double* field = flux ? flux[i].flux : values + i * width;
        const auto offset = k * stride();
        for (unsigned v = 0; v < width; ++v) {
            if (output) output[offset + v] = field[v];
            else field[v] = input[offset + v];
        }
        if (flux) {
            if (output) output[offset + width] = flux[i].max_wave_speed;
            else flux[i].max_wave_speed = input[offset + width];
        }
    }
};
void update(Grid grid, explicit_detail::Stage stage, double dt, bool save_original);
void transfer(RecordField field, const MeshIndex* ids, Index count, const double* input, double* output);
}
}
#endif
