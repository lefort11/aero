#pragma once
#include "transport.h"
#include "settings.h"
#include "../aero/mpi_periodic.h"
#include "../residual/residual.h"
#include <functional>
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/storage.h"
#endif

namespace aero {
namespace viscosity {
// Владеет геометрией и рабочей памятью CPU. Нумерацию неизвестных и обмен значениями
// задаёт общий оператор солвера. Геометрические коэффициенты строятся только при старте.
class Aes {
public:
    using Locate = std::function<MeshIndex(MeshIndex,int)>;
    void prepare(const MeshPartition&, const PeriodicPartition*, const ResidualFrame&, const Processes&,
                 const ViscositySettings&, double gamma, double mach, MeshIndex rows, const Locate&);
#ifdef AERO_HAS_TURBULENCE
    turbulence::Storage turbulence_field;
    template<class Grid> void prepare_sa_faces(const Grid& grid,bool host) {
        turbulence_field.prepare_faces(grid,op_.sa,host);
    }
    void prepare_turbulence(const turbulence::Settings& settings, bool host) {
        turbulence_field.prepare_diffusion(op_,op_.sa,host);
        turbulence_field.prepare(op_.sa,settings,1./op_.law.inverse_reynolds,op_.law.prandtl,host);
    }
#endif
    void allocate_host(MeshIndex state_nodes);
    // После явной стадии на одном процессе узловая вязкость уже готова.
    // В остальных случаях пересчитываем её по текущему полю, включая узлы соседей.
    void evaluate(const double* state, const EulerState* cached = nullptr, bool gas_is_current = false,
                  Evaluation phase = Evaluation::Full);
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
#ifdef AERO_HAS_TURBULENCE
        if (g.sa.active) turbulence_field.upload(g.sa,storage);
#endif
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
