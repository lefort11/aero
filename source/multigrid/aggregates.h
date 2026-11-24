#pragma once
#include "../residual/residual_detail.h"
#include <vector>

namespace aero {
namespace multigrid {
// Строки принадлежат текущему процессу. У каждой строки ровно один агрегат.
// Связи с соседними процессами сохраняет умножение исходного якобиана на вектор.
struct Aggregates {
    std::vector<MeshIndex> offsets; // Начало списка строк каждого агрегата и конец последнего списка.
    std::vector<MeshIndex> members; // Номера собственных строк, сгруппированные по агрегатам.

    // Объединяет собственные узлы в агрегаты грубого уровня.
    Aggregates(const residual_detail::Geometry& mesh, MeshIndex rows, const MeshIndex* owned);
    // Возвращает число агрегатов грубого уровня.
    MeshIndex count() const { return offsets.size() - 1; }
};
}
}
