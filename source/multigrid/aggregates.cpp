#include "aggregates.h"
#include <algorithm>
#include <limits>

namespace aero {
namespace multigrid {
// Объединяет собственные узлы в агрегаты грубого уровня.
Aggregates::Aggregates(const residual_detail::Geometry& mesh, MeshIndex rows, const MeshIndex* owned) {
    const MeshIndex missing = std::numeric_limits<MeshIndex>::max();
    // Обратное соответствие нужно только при подготовке. Для узла соседнего ранга
    // оставляем missing: права изменять его состояние у этого процесса нет.
    std::vector<MeshIndex> available(mesh.nodes, missing);
    for (MeshIndex row = 0; row < rows; ++row) available[owned ? owned[row] : row] = row;
    offsets.push_back(0);
    members.reserve(rows);
    std::vector<std::pair<double,MeshIndex>> neighbours;
    for (MeshIndex row = 0; row < rows; ++row) {
        const MeshIndex node = owned ? owned[row] : row;
        if (available[node] == missing) continue;
        members.push_back(row);
        available[node] = missing;
        neighbours.clear();
        for (MeshIndex entry = mesh.offsets[node]; entry < mesh.offsets[node+1]; ++entry) {
            const MeshIndex face = mesh.incidence[entry].face();
            if (face >= mesh.edges) continue; // У граничной площадки нет второго контрольного объёма.
            const auto& edge = mesh.surfaces[face];
            const MeshIndex other = edge.begin == node ? edge.end : edge.begin;
            if (available[other] != missing) neighbours.emplace_back(-edge.measure, other);
        }
        // На вытянутой сетке у соседей с большей общей площадью сильнее обмен потоком.
        // Одинаковые площади разрешаем по номеру узла для воспроизводимости.
        std::sort(neighbours.begin(), neighbours.end());
        for (const auto& candidate : neighbours) {
            auto& index = available[candidate.second];
            if (index == missing) continue;
            members.push_back(index);
            index = missing;
            if (members.size() - offsets.back() >= (1u << mesh.dim)) break;
        }
        offsets.push_back(members.size());
    }
}
}
}
