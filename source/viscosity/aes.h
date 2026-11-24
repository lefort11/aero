#pragma once
#include "transport.h"
#include "settings.h"
#include "../aero/mpi_periodic.h"
#include "../residual/residual.h"
#include "../parallel/upload_compact.h"
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
    // Находит локальный индекс по номеру и идентификатору владельца.
    using Locate = std::function<MeshIndex(MeshIndex,int)>;
    // Строит геометрию и рабочие связи AES для выбранной подобласти и базиса импульса.
    void prepare(const MeshPartition&, const PeriodicPartition*, const ResidualFrame&, const Processes&,
                 const ViscositySettings&, double gamma, double mach, MeshIndex rows, const Locate&);
#ifdef AERO_HAS_TURBULENCE
    // Подключает настройки фильтра и замыкания LES к вязкому оператору.
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
    // Владеет рабочими полями переноса SA и геометрией диффузии.
    turbulence::Storage turbulence_field;
    // Готовит граничные и внутренние площадки конвективного переноса SA.
    template<class Grid> void prepare_sa_faces(const Grid& grid,bool host) {
        turbulence_field.prepare_faces(grid,op_.sa,host);
    }
    // Подготавливает геометрию и рабочие поля активной турбулентной модели.
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
    // Для резервного размещения сохраняем геометрию на хосте до проверки свободной памяти.
    void prepare_device_geometry() { keep_affine_input_=true; }
    // Переключает вычисление на компактное представление аффинных гексаэдров.
    bool use_affine_geometry();
    // Исходная нумерация вершин повторяется на многих элементах. Только список из
    // тетраэдров общий для такой нумерации, координаты каждой ячейки индивидуальны.
    bool use_coordinate_geometry(bool implicit=false);
    // Проверяет использование координатной геометрии в неявном вязком операторе.
    bool uses_implicit_coordinates() const { return implicit_coordinates_; }
    // Проверяет выбор компактного представления геометрии AES.
    bool uses_compact_geometry() const { return compact_geometry_; }
    // Невязка и якобиан одного процесса по очереди занимают один буфер градиентов.
    bool has_gradient_tiles() const { return !gradient_tiles_.empty(); }
    // Завершает подготовку и размещение геометрии вязкого оператора на устройстве.
    void finish_device_geometry();
    // Освобождает исходные CPU-данные после подготовки геометрии устройства.
    void release_device_geometry_input() {
        std::vector<AffineInput>().swap(affine_input_);
        if (!coordinate_geometry_) {
            std::vector<Coordinate>().swap(tetra_position_);
            std::vector<VolumeCellNodes>().swap(tetra_cells_);
        }
    }
    // Выделяет рабочие массивы вязкого оператора на CPU.
    void allocate_host(MeshIndex state_nodes);
    // После явной стадии на одном процессе узловая вязкость уже готова.
    // В остальных случаях пересчитываем её по текущему полю, включая узлы соседей.
    void evaluate(const double* state, const EulerState* cached = nullptr, bool gas_is_current = false,
                  Evaluation phase = Evaluation::Full);
    // Возвращает невладеющее представление геометрии, материалов и рабочих полей AES.
    const Operator& view() const { return op_; }
    std::vector<unsigned char> wall; // Единица у узла неподвижной теплоизолированной стенки.

    // Система размещения GPU вызывает эту функцию при подсчёте памяти и при загрузке.
    template<class Storage> Operator upload(Storage& storage, MeshIndex state_nodes,
                                            const FlowScratch* scratch=nullptr) const {
        Operator g = op_;
        g.node_count = state_nodes;
        g.cells = upload_records<Cell32>(storage,cells_.size(),cells_.data());
        // При подсчёте памяти известен размер таблицы. Заполняем её на CPU
        // только после выбора размещения, если она действительно нужна карте.
        g.affine=compact_geometry_ && !coordinate_geometry_ ?
            storage.template allocate<AffineHex>(g.cell_count,affine_.data()) : nullptr;
        if (coordinate_geometry_) {
            g.element_order=true; // Градиенты упорядочены по вершинам каждого элемента.
            // При полной идентичности номеров точек в каждом элементе используем
            // связность AES. Если найдена хотя бы одна периодическая копия,
            // сохраняем исходные номера, по которым действительно заданы координаты.
            const VolumeCellTable vertices=shared_coordinate_nodes_ ?
                VolumeCellTable{nullptr,g.cells,storage.upload(tetra_pattern_ids_)} :
                VolumeCellTable{storage.upload(tetra_cells_),nullptr,nullptr};
            g.tetra={storage.upload(tetra_position_),vertices,
                     storage.upload(tetra_patterns_),storage.upload(tetra_means_),nullptr,coordinate_all_hex_,0};
#ifdef AERO_HAS_TURBULENCE
            // Освободилось восемь координат на форму фильтра. Только за счёт
            // этих же байтов запоминаем обратные объёмы, не увеличивая раскладку GPU.
            if (!implicit_coordinates_)
                g.tetra.cached_cells=std::min(coordinate_filters_.size(),tetra_cells_.size());
            g.tetra.volume_factors=storage.template allocate<double>(24*g.tetra.cached_cells);
#endif
        }
        g.stencil.pairs=compact_geometry_ ? nullptr : storage.upload(metrics_);
        g.stencil.pair_heat=storage.upload(pair_heat_);
        g.offsets = upload_indices(storage,offsets_.size(),offsets_.data());
        g.incidence = upload_indices(storage,incidence_.size(),incidence_.data());
        g.gas = scratch ? scratch->gas : storage.template allocate<Gas>(state_nodes);
        g.material = storage.template allocate<Material>(g.cell_count);
        g.rows = scratch ? scratch->rows : storage.template allocate<Row>(g.row_count);
#ifdef AERO_HAS_TURBULENCE
        if (g.computes_les()) {
            g.box_nodes=storage.upload(box_nodes_);
            // Координатный фильтр берёт узлы из основной связности. Плитку номеров
            // оставляем, только если её использует упакованный фильтр или явный перенос SA.
            const bool need_filter_nodes=!compact_geometry_ || g.vertices_per_cell;
            g.hex_filter={need_filter_nodes ? storage.upload(hex_nodes_) : nullptr,compact_geometry_ ? nullptr : storage.upload(hex_coefficients_),
                          op_.hex_filter.lanes};
            // Если градиенты уже упакованы, загружаем только геометрию, нужную
            // при уточнении диаметра. На смешанной сетке используется полная запись.
            g.filter_extents=coordinate_filters_.empty() ? storage.upload(filter_extents_) : nullptr;
            g.coordinate_filters=storage.upload(coordinate_filters_);
            g.mean_areas={storage.upload(mean_area_forms_),storage.upload(mean_area_terms_),storage.upload(mean_area_ids_)};
            g.filters=filter_extents_.empty() && coordinate_filters_.empty() ? storage.upload(filters_) : nullptr;
            g.filter_indices=storage.upload(filter_indices_);
            g.les_volumes=storage.upload(les_volumes_);
            // Фильтр пишет временные значения по элементам и суммирует их в узлы.
            // После этого та же память нужна коэффициентам потока. У пробного газа
            // Ньютон сохраняет собранные строки отдельно; после отката коэффициенты
            // вычисляются из восстановленного состояния перед использованием.
            const bool share_filter=implicit_coordinates_ && g.c_des>0.;
            static_assert(sizeof(Material)>=sizeof(turbulence::LesSample)+sizeof(double),
                          "AES material workspace is too small for DDES samples");
            g.samples=share_filter ? reinterpret_cast<turbulence::LesSample*>(g.material) :
                      storage.template allocate<turbulence::LesSample>(g.cell_count);
            if (g.c_des>0.) {
                g.sa.ddes=storage.upload(ddes_nodes_);
                // При предварительном подсчёте памяти адреса пусты. Смещение от
                // нулевого указателя вычислять нельзя даже для оценки размещения.
                g.cell_gradient_norm=share_filter ? (g.samples ? reinterpret_cast<double*>(g.samples+g.cell_count) : nullptr) :
                                     storage.template allocate<double>(g.cell_count);
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
        if (g.sa.active) turbulence_field.upload(g.sa,storage,compact_geometry_);
#endif
        if (!gradient_tiles_.empty()) {
            // Одна рабочая область для всех групп. Первый диапазон имеет максимальную длину.
            g.gradient_slices=upload_records<GradientSlice32>(storage,gradient_slices_.size(),gradient_slices_.data());
            g.host_gradient_tiles=gradient_tiles_.data();
            g.gradient_tile_count=gradient_tiles_.size();
            g.gradient_cache=storage.template allocate<Gradient>(8*gradient_tiles_.front().cells);
        } else if (g.element_order || store_iteration_gradients_)
            g.gradient_cache=storage.template allocate<Gradient>(g.element_order ? g.vertex_count : incidence_.size());
        return g;
    }
private:
    // Подготавливает геометрические коэффициенты на целой сетке без фильтра LES.
    void prepare_local_cells(const MeshPartition&, const PeriodicPartition*, const ResidualFrame&, const Locate&);
    // Превращает локальные номера пар из каждого блока в номера общей таблицы.
    void merge_cell_metrics(std::vector<std::vector<PairMetric>>&);
    // Собирает список вязких вкладов каждого узла в исходном порядке элементов.
    void index_local_vertices();
    // Исходные координаты и средние градиенты аффинного элемента.
    struct AffineInput {
        Coordinate position[8]; // Исходные координаты до подготовки фильтра DDES.
        MeanGradient mean;      // Интегралы базисных функций штатной схемы AES.
        std::uint64_t adjacency; // Восемь битовых масок рёбер локальных вершин.
    };
    // Сохраняет исходные данные для выбора аффинного представления.
    bool keep_affine_input_=false;
    // Исходная геометрия кандидатов на аффинную упаковку.
    std::vector<AffineInput> affine_input_;
    // Метрики аффинных гексаэдров.
    std::vector<AffineHex> affine_;
    bool compact_geometry_=false; // Выбирает раскладку до выделения таблицы на CPU.
    // Коэффициенты тепла нужны газу и SA при обоих способах хранения геометрии.
    void prepare_compact_geometry();
    // Используется координатная геометрия и выполнено ли условие однородной гексаэдральной сетки.
    bool coordinate_geometry_=false, coordinate_all_hex_=false;
    bool implicit_coordinates_=false; // При BDF рабочее место приоритетно отдаём якобиану и предобуславливателю.
    // Координатные ядра используют общую таблицу вершин.
    bool shared_coordinate_nodes_=false;
    // Не предполагаем определённого типа ячеек, порядка узлов или отсутствия
    // периодичности. На каждой ячейке сравниваем столько номеров, сколько у неё вершин.
    void share_coordinate_nodes(); // Проверяем реальные номера всех вершин до загрузки на GPU.
    // Номера схем разбиения элементов на тетраэдры.
    std::vector<std::uint32_t> tetra_pattern_ids_;
    // Узловые координаты для тетраэдрального разбиения.
    std::vector<Coordinate> tetra_position_;
    // Номера вершин ячеек для координатной геометрии.
    std::vector<VolumeCellNodes> tetra_cells_;
    // Повторно используемые схемы тетраэдрального разбиения.
    std::vector<VolumePattern> tetra_patterns_;
    // Средние градиенты элементов координатной схемы.
    std::vector<MeanGradient> tetra_means_;
    std::vector<double> pair_heat_; // Исходные коэффициенты для скалярной диффузии SA и производной теплового потока.
    // Плитки для промежуточного вычисления градиентов.
    std::vector<GradientTile> gradient_tiles_;
    // Участки плиток для накопления узловых вкладов.
    std::vector<GradientSlice> gradient_slices_;
    // Градиенты газа нужны, пока суммируются вклады в узлы. Обрабатываем одну группу
    // элементов и затем отдаём ту же рабочую память следующей группе. При склейке MPI
    // несколько локальных вершин могут обозначать один узел, поэтому используем готовую incidence_.
    void prepare_gradient_tiles(MeshIndex capacity=65536);
    // Сохраняет исходную аффинную геометрию для последующего фильтра LES/DDES.
    void remember_affine_input(MeshIndex cell,const Element&,const MeanGradient&);
    // Представление геометрии и полей, передаваемое вычислительным ядрам.
    Operator op_{};
    bool store_iteration_gradients_=false; // Для BDF с переносом SA можно повторно использовать неизменное поле газа.
    // Сохранённые элементные градиенты.
    std::vector<Gradient> gradient_cache_;
#ifdef AERO_HAS_TURBULENCE
    // Готовит геометрию гексаэдрального фильтра LES/DDES.
    void prepare_hex_filter();
    // На произвольной сетке ориентацию каждого тетраэдра определяем по фактическим
    // координатам. Только после этого можно разделить одну формулу между ячейками.
    bool prepare_mean_areas(); // Выбираем общие целочисленные формы вместо таблицы градиентов.
    // Формулы средних площадей по типам ячеек.
    std::vector<MeanAreaForm> mean_area_forms_;
    // Члены хранимых формул средних площадей.
    std::vector<MeanAreaTerm> mean_area_terms_;
    // Индексы формул для отдельных элементов.
    std::vector<std::uint32_t> mean_area_ids_;
    // Формирует связи элементов с узлами для осреднения материальных коэффициентов.
    void prepare_material_vertices();
    unsigned host_vertices_=0; // Общее число вершин в элементе, ноль для смешанной сетки.
    // Управляют кэшем материала и использованием вязкости SA.
    bool cache_material_=false, material_from_sa_=false;
    // Номера вершин при кэшировании узловых коэффициентов.
    std::vector<std::uint32_t> material_vertices_;
    // Упакованная связность гексаэдров для фильтра.
    std::vector<std::uint32_t> hex_nodes_;
    // Метрики фильтра гексаэдров.
    std::vector<double> hex_coefficients_;
    // Признак вычисления фильтра на GPU.
    bool gpu_filter_=false;
    std::vector<std::uint32_t> box_nodes_; // Для каждой геометрической вершины — номер неизвестной.
    std::vector<turbulence::FilterCell> filters_; // Общая геометрия элементов одинаковой формы.
    std::vector<turbulence::FilterExtent> filter_extents_; // Запасные данные для упакованного фильтра GPU.
    std::vector<turbulence::FilterLengths> coordinate_filters_; // Размеры и пары узлов для фильтра по координатам AES.
    // Соответствие элементов записям фильтра.
    std::vector<MeshIndex> filter_indices_;
    std::vector<double> les_volumes_; // Суммарный вес элементов вокруг каждого узла.
    // Элементные данные модели LES.
    std::vector<turbulence::LesSample> samples_;
    // Узловая вихревая вязкость и элементные нормы градиентов.
    std::vector<double> eddy_nu_, cell_gradient_norm_;
    // Узловые параметры масштаба DDES.
    std::vector<turbulence::DdesPoint> ddes_nodes_;
#endif
    // Метрики и связность ячеек на CPU.
    std::vector<Cell> cells_;
    std::vector<PairMetric> metrics_; // Различающиеся геометрические коэффициенты.
    std::vector<MeshIndex> offsets_, incidence_; // Строка узла содержит номера элементов и местные номера вершин.
    // Узловые примитивные величины для вязкого оператора.
    std::vector<Gas> gas_;
    // Средние транспортные коэффициенты ячеек.
    std::vector<Material> material_;
    // Узловые строки вязкой невязки.
    std::vector<Row> rows_;
};
}
}
