#pragma once
#include "transport.h"
#include "settings.h"
#include "../aero/mpi_periodic.h"
#include "../residual/residual.h"
#include <functional>

namespace aero {
namespace viscosity {
// Владеет геометрией и рабочей памятью CPU. Нумерацию неизвестных и обмен значениями
// задаёт общий оператор солвера. Геометрические коэффициенты строятся только при старте.
class Aes {
public:
    using Locate = std::function<MeshIndex(MeshIndex,int)>;
    void prepare(const MeshPartition&, const PeriodicPartition*, const ResidualFrame&, const Processes&,
                 const ViscositySettings&, double gamma, double mach, MeshIndex rows, const Locate&);
    void allocate_host(MeshIndex state_nodes);
    // После явной стадии на одном процессе узловая вязкость уже готова.
    // В остальных случаях пересчитываем её по текущему полю, включая узлы соседей.
    void evaluate(const double* state, const EulerState* cached = nullptr, bool gas_is_current = false);
    const Operator& view() const { return op_; }
    std::vector<unsigned char> wall; // Единица у узла неподвижной теплоизолированной стенки.

    // Система размещения GPU вызывает эту функцию при подсчёте памяти и при загрузке.
    template<class Storage> Operator upload(Storage& storage, MeshIndex state_nodes) const {
        Operator g = op_;
        g.node_count = state_nodes;
        g.cells = storage.upload(cells_);
        g.stencil.pairs = storage.upload(metrics_);
        g.offsets = storage.upload(offsets_);
        g.incidence = storage.upload(incidence_);
        g.gas = storage.template allocate<Gas>(state_nodes);
        g.material = storage.template allocate<Material>(g.cell_count);
        g.rows = storage.template allocate<Row>(g.row_count);
        return g;
    }
private:
    Operator op_{};
    std::vector<Cell> cells_;
    std::vector<PairMetric> metrics_; // Различающиеся геометрические коэффициенты.
    std::vector<MeshIndex> offsets_, incidence_; // Строка узла содержит номера элементов и местные номера вершин.
    std::vector<Gas> gas_;
    std::vector<Material> material_;
    std::vector<Row> rows_;
};
}
}
