#ifndef AERO_PARTITION_OPERATOR_H
#define AERO_PARTITION_OPERATOR_H
#include "partition_transfers.h"
#include "mpi_stencil.h"
#include "solver_timings.h"
#include "../residual/residual_detail.h"
#include "../explicit/explicit_stage.h"

namespace aero {
// Пространственный оператор одной подобласти. Сетка и её владельцы живут дольше оператора.
class PartitionOperator {
    using Id = MeshIndex;

public:
    // Строит локальные поверхности, определяет владельцев потоков и готовит геометрию оператора.
    PartitionOperator(const MeshPartition& part, CvView cv, const PeriodicPartition* periodic,
                      const BoundaryPatches& patches, ResidualOptions options, const ResidualFrame& frame,
                      const Processes& world, StencilRequests* requests, bool cuda);
    // Число узлов состояния с учётом удалённых узлов шаблонов реконструкции.
    Id state_nodes() const;
    // Возвращает объём работы и обменов процесса для отчёта о разбиении.
    PartitionWork work(Id leaves) const;
    // Строит шаблоны и добавляет в обмен удалённые узлы, нужные для реконструкции.
    Id prepare_stencil(const MeshPartition& part, const PeriodicPartition* seam,
                       const std::vector<PeriodicPair>& pairs, PartitionInput* input,
                       const StencilRequests& requests, const ReconstructionOptions& options);
    // Согласует с соседями порядок узлов и рёбер и выделяет буферы для повторных обменов.
    void connect(bool cuda);
#ifdef AERO_WITH_CUDA
    // Размещает данные сетки и индексы обмена в общей области памяти GPU.
    void layout_gpu(bool compact_buffer);
    // Оценивает объём по той же раскладке и резервирует память карты.
    void reserve_gpu(bool compact_buffer, GpuMemoryFit& fit);
    // Передаёт начальное и текущее состояния на GPU перед первым шагом.
    void upload_state(const std::vector<double>& state, const std::vector<double>& initial);
#endif
    // Собирает значения только собственных узлов в компактный буфер.
    void copy_owned(const std::vector<double>& state, std::vector<double>& compact) const;
    // Обновляет копию текущих полей на CPU для записи результатов.
    void download(std::vector<double>& state) const;
    // Обменивает значения узлов после стадии на активном вычислительном устройстве.
    void halo(std::vector<double>& u, double* seconds);
    // Заполняет узлы соседей в памяти CPU при подготовке и восстановлении расчёта.
    void halo_host(std::vector<double>& u);
    // Вычисляет потоки на принадлежащих процессу поверхностях и получает остальные от соседей.
    void assemble_fluxes(const std::vector<double>& u, const std::vector<double>& initial,
                         SolverTimings& timing);
    // Находит допустимый шаг по собственным узлам и берёт минимум по всем процессам.
    double cfl_step(double cfl);
    // Обновляет собственные узлы на одной стадии интегратора.
    void update_owned(explicit_detail::Stage stage, std::vector<double>& start, double dt,
                      std::vector<double>& state, bool save_original);

    std::vector<MeshIndex> owned; // Локальные номера узлов, которые обновляет этот процесс.
private:
    // Добавляет рёбра с собственным узлом и запоминает, кому нужно передать поток.
    void add_edge_surfaces(MeshView mesh, CvView cv, const PeriodicPartition* periodic,
                           const ResidualFrame& frame, StencilRequests* requests);
    // Добавляет физические границы, за которые отвечает эта подобласть.
    void add_boundary_surfaces(CvView cv, const PeriodicPartition* periodic, Id physical_patches,
                               const BoundaryPatches& patches, const ResidualFrame& frame,
                               StencilRequests* requests);
    // Возвращает активное устройство или nullptr для обмена в памяти CPU.
    PartitionGpu* device() const;
#ifdef AERO_WITH_CUDA
    std::unique_ptr<PartitionGpu> gpu_; // Раскладка, рабочая память и ядра текущей карты.
#endif
    PartitionTransfers nodes_; // Значения узлов, общие для соседних подобластей.
    PartitionTransfers edges_; // Результаты расчёта потоков на рёбрах между подобластями.
    unsigned width_;           // Число консервативных переменных в одном узле.
    const double* volumes_;    // Объёмы контрольных ячеек в расчётной нумерации.
    const std::vector<Id>&
        ids_; // Исходный номер каждого локального узла или его периодического представителя.
    const std::vector<int>& owners_;                 // MPI-процесс, отвечающий за каждый узел.
    const Processes& world_;                         // Все процессы этого расчёта.
    void prepare_nodal_gas(); // Резервирует одну запись на собственный или удалённый узел.
    template <unsigned order> void prepare_weno_tiles(std::vector<residual_detail::WenoEdgeTile<order>>& tiles); // Готовит общую для CPU и GPU геометрию полных двумерных шаблонов WENO3 и WENO5.
    std::vector<residual_detail::WenoEdgeTile<3>> weno3_tiles_; // По 32 шаблона третьего порядка.
    std::vector<residual_detail::WenoEdgeTile<5>> weno5_tiles_; // Дальние наклоны нужны пятому порядку.
    std::vector<MeshIndex> other_faces_; // Оставшиеся поверхности этого процесса и физические границы.
    std::vector<residual_detail::FaceNormal> normals_; // Направление потока на каждом ребре, на CPU.
    std::vector<EulerState> nodal_gas_; // Величины текущей стадии для рёбер на CPU.
    bool gas_current_ = false; // В однопроцессном расчёте стадия обновляет сразу и поле, и этот массив.
    PartitionStencil stencil_;                       // Строки реконструкции и дополнительные удалённые узлы.
    std::vector<residual_detail::Surface> surfaces_; // Сначала рёбра, затем части физических границ.
    std::vector<EulerFluxResult> flux_;              // Потоки для каждой поверхности при расчёте на CPU.
    std::vector<Id> compute_; // Индексы поверхностей, на которых процесс сам вычисляет поток.
    std::vector<residual_detail::Index> offsets_;          // Начало списка поверхностей каждого узла.
    std::vector<residual_detail::OrientedFace> incidence_; // Связи узлов с прилегающими поверхностями.
    residual_detail::Geometry grid_{}; // Представление этих массивов для вычислительных ядер.
};
}
#endif
