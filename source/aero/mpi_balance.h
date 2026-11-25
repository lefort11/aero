#ifndef AERO_MPI_BALANCE_H
#define AERO_MPI_BALANCE_H
#include "../mesh/leaf_balance.h"

namespace aero {
struct RunSettings;
// Прибавляет размер массива к оценке памяти с проверкой переполнения.
inline MeshIndex partition_bytes(MeshIndex base, MeshIndex count, MeshIndex bytes) {
    if (count && bytes > (mdec::absent - base) / count)
        throw MeshError("MPI partition: estimated mesh size overflow");
    return base + count * bytes;
}
// Бюджет карты, диапазон листьев и результат пробного выделения памяти для него.
struct GpuMemoryFit {
    // Свободная память, доступный бюджет, границы диапазона и потребность в байтах.
    MeshIndex free_bytes = 0, budget = 0, first = 0, end = 0, required = 0;
    // Точная раскладка массивов раздела, включая выравнивание. Оценка памяти СЛАУ входит только в required.
    MeshIndex array_count = 0, layout_bytes = 0;
    // Вклад реконструкции, кэша, неявного шага и грубой коррекции в бюджет.
    MeshIndex reconstruction_bytes = 0, cache_bytes = 0, implicit_bytes = 0, coarse_bytes = 0;
    // План размещения не укладывается в ограничения.
    bool failed = false;
    // Бюджет последней однопроцессной раскладки с известным графом матрицы.
    MeshIndex compact_budget = 0;
};
// Локальное состояние карты участвует в коллективном подборе следующего разбиения.
struct GpuPlacement {
    // Оценка памяти для текущего процесса.
    GpuMemoryFit local;
    std::vector<GpuRankCapacity> cards; // История размещения по картам на процессе 0.
};
// Подбирает непрерывные диапазоны листьев с учётом скоростей процессов и доступной памяти.
LeafLayout plan_partition(const RunSettings& run, const Processes& world, GpuPlacement* gpu = nullptr);
}
#endif
