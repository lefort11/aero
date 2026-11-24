#ifndef AERO_LEAF_BALANCE_H
#define AERO_LEAF_BALANCE_H
#include "partition.h"
#include <functional>

namespace aero {
// Оценки объёма работы и байтов памяти для префикса каталога листьев.
struct LeafCost {
    // Оценки вычислительной работы и объёма памяти листа.
    MeshIndex work, bytes;
};
// Процессы одной машины делят единый бюджет памяти для сетки.
struct CpuNodeShare {
    // MPI-процессы одного вычислительного узла.
    std::vector<int> ranks;
    // Бюджет памяти узла для размещения сетки.
    long double mesh_bytes;
};
// Производительность и допустимые диапазоны листьев для одного ранга GPU.
struct GpuRankCapacity {
    long double work_capacity; // Единицы работы по архиву, с уточнением по фактическому размещению на карте.
    // Для этих диапазонов целых листьев на данном процессе действительно не удалось выделить память устройства.
    std::vector<std::pair<MeshIndex, MeshIndex>> rejected;
};
// Накопленная стоимость диапазона [0, end). Достаточно прочитать каталог в точках проверяемых разрезов.
using LeafPrefix = std::function<LeafCost(MeshIndex)>;
// Распределяет листья сетки между процессами с учётом работы и памяти.
LeafLayout balance_leaves(MeshIndex leaves, const LeafPrefix& prefix, const std::vector<CpuNodeShare>& nodes,
                          const std::vector<long double>& speeds);
// Распределяет листья с учётом производительности и вместимости GPU.
LeafLayout balance_gpu_leaves(MeshIndex leaves, const LeafPrefix& prefix,
                              const std::vector<CpuNodeShare>& nodes, const std::vector<long double>& speeds,
                              const std::vector<GpuRankCapacity>& cards);
}
#endif
