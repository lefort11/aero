#ifndef AERO_MPI_BALANCE_H
#define AERO_MPI_BALANCE_H
#include "../mesh/leaf_balance.h"

namespace aero {
struct RunSettings;
inline MeshIndex partition_bytes(MeshIndex base, MeshIndex count, MeshIndex bytes) {
    if (count && bytes > (mdec::absent - base) / count)
        throw MeshError("MPI partition: estimated mesh size overflow");
    return base + count * bytes;
}
// Бюджет карты, диапазон листьев и результат пробного выделения памяти для него.
struct GpuMemoryFit {
    MeshIndex free_bytes = 0, budget = 0, first = 0, end = 0, required = 0;
    bool failed = false;
};
// Локальное состояние карты участвует в коллективном подборе следующего разбиения.
struct GpuPlacement {
    GpuMemoryFit local;
    std::vector<GpuRankCapacity> cards; // История размещения по картам на процессе 0.
};
// Подбирает непрерывные диапазоны листьев с учётом скоростей процессов и доступной памяти.
LeafLayout plan_partition(const RunSettings& run, const Processes& world, GpuPlacement* gpu = nullptr);
}
#endif
