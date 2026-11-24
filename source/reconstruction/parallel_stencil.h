#ifndef AERO_PARALLEL_STENCIL_H
#define AERO_PARALLEL_STENCIL_H
#include "reconstruction.h"
#include "../parallel/parallel_for.h"
#include <algorithm>

namespace aero {
namespace detail {
// Добавляет локальные коэффициенты в итоговые массивы параллельной сборки шаблона.
template <class T> void append_values(std::vector<T>& to, const std::vector<T>& from) {
    to.insert(to.end(), from.begin(), from.end());
}
// Переносит локальные смещения в общий массив, пропуская начальный ноль следующего фрагмента.
inline void append_offsets(std::vector<ReconstructionIndex>& to,
                           const std::vector<ReconstructionIndex>& from) {
    // Нулю локального фрагмента соответствует конец последней собранной строки.
    const auto base = to.back();
    for (std::size_t i = 1; i < from.size(); ++i)
        to.push_back(base + from[i]);
}
// Для обеих сторон каждого ребра поля метаданных добавляются в том же порядке, что и веса.
inline void append_edge_rows(ReconstructionStencilData& to, const ReconstructionStencilData& from) {
    append_offsets(to.offsets, from.offsets);
    append_values(to.nodes, from.nodes);
    append_values(to.weights, from.weights);
    append_values(to.row_owner, from.row_owner);
    append_values(to.complete, from.complete);
    append_values(to.side_scheme, from.side_scheme);
    append_values(to.outer_distance, from.outer_distance);
    append_values(to.far_distance, from.far_distance);
    append_offsets(to.slope_offsets, from.slope_offsets);
    append_values(to.slope_nodes, from.slope_nodes);
    append_values(to.slope_weights, from.slope_weights);
    append_values(to.weno_order, from.weno_order);
    append_values(to.weno_epsilon_scale, from.weno_epsilon_scale);
}

// Параллельно считаем непрерывные диапазоны рёбер. Готовые строки CSR объединяем в порядке рёбер.
// После каждого пакета освобождаем его временные шаблоны.
template <class Build> void build_edge_rows(ReconstructionStencilData& data, const Build& build) {
#ifdef _OPENMP
    if (data.edge_count >= 256) {
        // Короткие участки дают много барьеров и тысячи мелких расширений массивов.
        // На большой сетке за одно распределение потокам отдаём до 65536 рёбер.
        const ReconstructionIndex chunk = data.edge_count < 8192 ? 32 : 256, batch = 256 * chunk;
        for (ReconstructionIndex start = 0; start < data.edge_count;) {
            const auto size = std::min(batch, data.edge_count - start);
            std::vector<ReconstructionStencilData> parts(
                static_cast<std::size_t>((size + chunk - 1) / chunk));
            parallel_for(
                parts.size(),
                [&](std::size_t i) {
                    const auto end = start + std::min((i + 1) * chunk, size);
                    for (auto e = start + i * chunk; e < end; ++e)
                        build(e, parts[i]);
                },
                2);
            for (const auto& part : parts)
                append_edge_rows(data, part);
            start += size;
        }
        return;
    }
#endif
    for (ReconstructionIndex e = 0; e < data.edge_count; ++e)
        build(e, data);
}
}
}
#endif
