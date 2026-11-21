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
    void configure_les(const turbulence::Settings& settings, double prandtl, bool gpu=false, bool explicit_whole_mesh=false) {
        // На GPU многократные обращения к одним и тем же вязкостям соседей дороги.
        // На CPU исходный обход элемента использует аппаратный кэш и не пишет промежуточный массив.
        cache_material_=gpu && explicit_whole_mesh && settings.enabled;
        material_from_sa_=settings.uses_sa();
        gpu_filter_=gpu;
        op_.les=settings.les;
        op_.c_des=settings.ddes ? settings.c_des : 0.;
        op_.fixed_ddes_length=settings.ddes && !settings.les.directional();
        op_.les_pr_ratio=prandtl/settings.prandtl;
    }
    turbulence::Storage turbulence_field;
    template<class Grid> void prepare_sa_faces(const Grid& grid,bool host) {
        turbulence_field.prepare_faces(grid,op_.sa,host);
    }
    void prepare_turbulence(const turbulence::Settings& settings, bool host, bool implicit=false,
                            const std::vector<MeshIndex>* steady_nodes=nullptr) {
        // Внутри шага BDF газ при уточнении nu_tilde фиксирован. Для явного
        // расчёта промежуточная память под производные скоростей не нужна.
        // На GPU с прямоугольными элементами коэффициенты многократно попадают
        // в кэш карты: повторная сборка быстрее записи дополнительного массива.
        store_iteration_gradients_=implicit && (host || box_nodes_.empty());
        if (implicit && host && !op_.element_order) {
            gradient_cache_.resize(incidence_.size());
            op_.gradient_cache=gradient_cache_.data();
        }
        if (op_.fixed_ddes_length) {
            // Без твёрдой стенки d_DDES=C_DES*Δ. Если на всей подобласти длина
            // уже известна из сетки или расстояния до стены, оценка градиента не нужна.
            op_.skip_ddes_filter=true;
            for (MeshIndex n=0;n<op_.row_count;++n) {
                const double inverse_wall=turbulence_field.distance[n];
                if (inverse_wall>0. && ddes_nodes_[n].grid_length<1./inverse_wall) {
                    op_.skip_ddes_filter=false;
                    break;
                }
            }
        }
        turbulence_field.prepare_diffusion(op_,op_.sa,host,implicit,steady_nodes);
        turbulence_field.prepare(op_.sa,settings,1./op_.law.inverse_reynolds,op_.law.prandtl,host,implicit);
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
        if (g.computes_les()) {
            g.box_nodes=storage.upload(box_nodes_);
            g.hex_filter={storage.upload(hex_nodes_),storage.upload(hex_coefficients_),op_.hex_filter.lanes};
            g.filters=storage.upload(filters_);
            g.filter_indices=storage.upload(filter_indices_);
            g.les_volumes=storage.upload(les_volumes_);
            g.samples=storage.template allocate<turbulence::LesSample>(g.cell_count);
            if (g.c_des>0.) {
                g.sa.ddes=storage.upload(ddes_nodes_);
                g.cell_gradient_norm=storage.template allocate<double>(g.cell_count);
            } else {
                g.eddy_nu=storage.template allocate<double>(state_nodes);
            }
        }
        if (g.vertices_per_cell) {
            // Фильтр и вязкий поток обходят исходные вершины в одном порядке.
            // Уже загруженную таблицу номеров гексаэдра используем для обоих обходов.
            g.material_vertices=hex_nodes_.empty() ? storage.upload(material_vertices_) : g.hex_filter.nodes;
            g.material_nodes=storage.template allocate<Operator::NodalMaterial>(state_nodes);
        }
        if (g.sa.active) turbulence_field.upload(g.sa,storage);
#endif
        if (g.element_order || store_iteration_gradients_)
            g.gradient_cache=storage.template allocate<Gradient>(g.element_order ? g.vertex_count : incidence_.size());
        return g;
    }
private:
    Operator op_{};
    bool store_iteration_gradients_=false; // Для BDF с переносом SA можно повторно использовать неизменное поле газа.
    std::vector<Gradient> gradient_cache_;
#ifdef AERO_HAS_TURBULENCE
    void prepare_hex_filter();
    void prepare_material_vertices();
    unsigned host_vertices_=0; // Общее число вершин в элементе, ноль для смешанной сетки.
    bool cache_material_=false, material_from_sa_=false;
    std::vector<std::uint32_t> material_vertices_;
    std::vector<std::uint32_t> hex_nodes_;
    std::vector<double> hex_coefficients_;
    bool gpu_filter_=false;
    std::vector<std::uint32_t> box_nodes_; // Для каждой геометрической вершины — номер неизвестной.
    std::vector<turbulence::FilterCell> filters_; // Общая геометрия элементов одинаковой формы.
    std::vector<MeshIndex> filter_indices_;
    std::vector<double> les_volumes_; // Суммарный вес элементов вокруг каждого узла.
    std::vector<turbulence::LesSample> samples_;
    std::vector<double> eddy_nu_, cell_gradient_norm_;
    std::vector<turbulence::DdesPoint> ddes_nodes_;
#endif
    std::vector<Cell> cells_;
    std::vector<PairMetric> metrics_; // Различающиеся геометрические коэффициенты.
    std::vector<MeshIndex> offsets_, incidence_; // Строка узла содержит номера элементов и местные номера вершин.
    std::vector<Gas> gas_;
    std::vector<Material> material_;
    std::vector<Row> rows_;
};
}
}
