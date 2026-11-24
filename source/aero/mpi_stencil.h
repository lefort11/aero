#ifndef AERO_MPI_STENCIL_H
#define AERO_MPI_STENCIL_H
#include "mpi_periodic.h"
#include <set>

namespace aero {
// Связь локальной поверхности с двумя расчётными узлами и исходным ребром до периодической склейки.
struct StencilEdge {
    MeshIndex surface, begin, end;
    std::pair<MeshIndex, MeshIndex> original;
};
// Запросы рёбер шаблона и узлов, где нужен след первого порядка.
struct StencilRequests {
    std::vector<StencilEdge> edges;
    // На этих исходных узлах нужен узловой след вместо высокопорядкового.
    std::set<MeshIndex> inlet_nodes;
};
// Локальные строки реконструкции и их привязка к распределённому полю.
struct PartitionStencil {
    // CSR-таблица весов: смещения строк, номера узлов и коэффициенты.
    struct Rows {
        std::vector<ReconstructionIndex> offsets, nodes;
        std::vector<double> weights;
    };
    // Линейная строка на сторону EBR или четыре строки наклонов для WENO.
    Rows linear, slopes;
    Mp5Settings mp5{}; // Выбор и коэффициент ограничителя для локальных рёбер.
    // Номер локального конца ребра, относительно которого считается приращение.
    std::vector<ReconstructionIndex> row_owner;
    std::vector<double> epsilon_scale;
    // Порядок, доступный каждой стороне с учётом полноты геометрического шаблона.
    std::vector<unsigned char> weno_order;
    std::vector<unsigned char> nodal_traces;
    // Дополнительные состояния, которые надо получать от владельца для удалённых узлов шаблона.
    std::vector<std::pair<MeshIndex, int>> extra_nodes;
    MeshIndex incomplete = 0;
    // Возвращает невладеющее представление локальных строк распределённого шаблона.
    ReconstructionStencilView view(MeshIndex state_nodes) const;
};
// Обе строки шаблона готовит процесс, который затем вычислит поток этого ребра.
PartitionStencil prepare_partition_stencil(const MeshPartition&, const PeriodicPartition*,
                                           const std::vector<PeriodicPair>&, PartitionInput*,
                                           const StencilRequests&, MeshIndex surfaces,
                                           const ReconstructionOptions&, const Processes&);
}
#endif
