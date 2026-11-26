#include "matrix_execution.h"
#include "spatial_jacobian.h"
#include "block_jacobi.h"
#include "csgs_sweep.h"
#include "csgs_order.h"
#include "spatial_jacobian_detail.h"
#include "../parallel/parallel_for.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#endif
#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/jacobian_pattern.h"
#include "../viscosity/row_execution.h"
#endif
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace aero {
namespace implicit {
namespace {
// Проверка размера нужна как при оценке памяти, так и при фактической загрузке.
bool indices_fit_32(const std::vector<MeshIndex>& source) {
    return source.empty() || *std::max_element(source.begin(),source.end()) <= std::numeric_limits<std::uint32_t>::max();
}
// Проверяет, помещаются ли смещения упакованных строк CSGS в 32 бита.
bool csgs_rows_fit_32(MeshIndex count, unsigned lanes) {
    return count <= std::numeric_limits<std::uint32_t>::max()-lanes;
}
#ifdef AERO_WITH_KOKKOS_GPU
// Выбор формата определяется содержимым графа, в том числе для периодических
// связей и смешанных элементов. Широкий номер никогда не обрезается.
IndexList upload_indices(gpu::DeviceArrays& arrays, const std::vector<MeshIndex>& source) {
    if (source.empty()) return {};
    if (!indices_fit_32(source))
        return arrays.upload(source);
    std::vector<std::uint32_t> packed(source.size());
    parallel_for(source.size(),[&](MeshIndex i) { packed[i] = static_cast<std::uint32_t>(source[i]); });
    return IndexList(arrays.upload(packed));
}

// Для раскраски на CPU восстанавливаем обычные номера. Это требуется, когда
// матрица была подготовлена до первого запроса на CSGS и исходный граф уже освобождён.
void read_indices(const IndexList& list, std::vector<MeshIndex>& result) {
    if (result.empty()) return;
    if (list.wide) {
        gpu::checked(gpu::copy(result.data(),list.wide,result.size()*sizeof(MeshIndex),gpu::Copy::ToHost));
        return;
    }
    std::vector<std::uint32_t> packed(result.size());
    gpu::checked(gpu::copy(packed.data(),list.narrow,packed.size()*sizeof(packed[0]),gpu::Copy::ToHost));
    parallel_for(result.size(),[&](MeshIndex i) { result[i] = packed[i]; });
}
#endif

// Привязка вхождения ребра в строку к столбцу соседнего узла вычисляется
// один раз. При обновлении коэффициентов поиск по номерам узлов уже не нужен.
struct BlockPattern {
    // Границы блочных строк, номера столбцов и соответствия соседей.
    std::vector<MeshIndex> rows, columns, neighbour;
    // Строит блочный граф матрицы по собственным узлам, площадкам и обменным связям.
    void build(const residual_detail::Geometry& g, MeshIndex count, const Partition& part) {
        const auto* owned = part.host_owned;
        rows.reserve(count + 1);
        neighbour.resize(g.offsets[g.nodes]);
        std::vector<char> seen; // Переиспользуем отметки кратных связей для следующего узла.
        for (MeshIndex row = 0; row < count; ++row) {
            const MeshIndex i = owned ? owned[row] : row;
            rows.push_back(columns.size());
            columns.push_back(i);
            const auto start = columns.size();
            for (MeshIndex p = g.offsets[i]; p < g.offsets[i+1]; ++p) {
                const auto f = g.incidence[p].face();
                if (f >= g.edges) continue;
                const auto& edge = g.surfaces[f];
                neighbour[p] = edge.begin == i ? edge.end : edge.begin;
                if (neighbour[p] != i) columns.push_back(neighbour[p]);
            }
#ifdef AERO_HAS_VISCOSITY
            if (part.host_aes.dim && !part.compact_jacobian)
                viscosity::append_columns(part.host_aes,i,columns);
#endif
            auto begin = columns.begin() + static_cast<std::ptrdiff_t>(start);
            std::sort(begin, columns.end());
            columns.erase(std::unique(begin, columns.end()), columns.end());
            // Для обычного ребра можно сразу присвоить внедиагональный блок.
            // На периодическом шве встречаются кратные рёбра: со второго вхождения складываем.
            seen.assign(columns.size() - start, 0);
            for (MeshIndex p = g.offsets[i]; p < g.offsets[i+1]; ++p) {
                if (g.incidence[p].face() >= g.edges) continue;
                if (neighbour[p] == i) {
                    neighbour[p] = 0; // Петля в одной степени свободы не даёт вклада.
                } else {
                    const auto slot = std::lower_bound(columns.begin() + static_cast<std::ptrdiff_t>(start),
                                                       columns.end(), neighbour[p]);
                    const auto index = static_cast<MeshIndex>(slot - columns.begin());
                    neighbour[p] = 2*index + static_cast<MeshIndex>(seen[index-start]);
                    seen[index-start] = 1;
                }
            }
        }
        rows.push_back(columns.size());
    }
};

// При совместной раскладке матрицы и CSGS после диагоналей идут блоки соседей.
// Запоминаем адрес каждого блока в порядке CSR, по которому работает сборка.
void coefficient_offsets(const BlockPattern& graph, unsigned width, bool mixed,
                         const CsgsLayout* sweep, std::vector<MeshIndex>& first) {
    if (!mixed && !sweep) return;
    const auto rows = graph.rows.size()-1;
    const unsigned stride = width * width;
    first.resize(graph.columns.size());
    parallel_for(rows,[&](MeshIndex row) {
        const auto start=graph.rows[row];
        first[start] = row*stride;
        if (!sweep)
            for (auto p=start+1; p<graph.rows[row+1]; ++p)
                first[p] = (p-row-1)*stride;
    });
    if (!sweep) return;
    const CsgsPacking packing{nullptr,nullptr,sweep->source.size(),width,sweep->lanes,{},nullptr};
    const MeshIndex base = mixed ? 0 : rows*stride;
    parallel_for(sweep->source.size(),[&](MeshIndex pos) {
        const auto slot=sweep->source[pos];
        if (slot!=static_cast<MeshIndex>(-1)) first[slot]=base+packing.first_coefficient(pos);
    });
}

// Обходы по цветам на GPU и по независимым участкам CPU используют те же строки.
std::unique_ptr<CsgsOrder> prepare_sweep(MeshIndex nodes,const BlockPattern& graph,bool gpu,CsgsLayout& layout) {
    host_parallel::ensure_runtime();
    std::unique_ptr<CsgsOrder> order(new CsgsOrder(nodes,graph.rows,graph.columns,gpu,
        static_cast<MeshIndex>(host_parallel::Execution().concurrency())));
    layout.build(*order,graph.rows,graph.columns,gpu);
    return order;
}
// Формирует диагностическое сообщение об ошибке сборки якобиана.
std::domain_error assembly_error(MeshIndex bad, const residual_detail::Geometry& g) {
    if (bad < g.edges) return std::domain_error("Spatial Jacobian: invalid edge " + std::to_string(bad));
    if (bad < g.faces) return std::domain_error("Spatial Jacobian: invalid boundary piece " + std::to_string(bad-g.edges));
    return std::domain_error("Spatial Jacobian: invalid time coefficient or nonfinite entry at node " +
                             std::to_string(bad-g.faces));
}
// Обновляет временную диагональ или собирает полные строки. Вид операции фиксирован
// при компиляции. Оба пути одинаково переводят отказ строки в номер узла, сохраняя
// параллельный сбор первого сбоя на GPU и исключения отдельных строк на CPU.
template<bool time_only>
void assemble_checked_rows(int device, const SpatialBlocks& blocks, const double* volumes, double dt) {
    if (!blocks.row_count) return;
    MeshIndex failed;
#ifdef AERO_WITH_KOKKOS_GPU
    if (device >= 0) {
        device_work::check(time_only
            ? device_work::aero_device_time_diagonal(device,blocks,volumes,dt,&failed)
            : device_work::aero_device_spatial_rows(device,blocks,volumes,dt,&failed));
    } else
#else
    (void)device;
#endif
    {
        host_parallel::ensure_runtime();
        const host_parallel::Execution exec;
        if (time_only) matrix_rows::time_diagonal(exec,blocks,volumes,dt,&failed);
        else matrix_rows::assemble(exec,blocks,volumes,dt,&failed);
    }
    if (failed < blocks.grid.nodes) throw assembly_error(blocks.grid.faces+failed,blocks.grid);
}

}

// Граф матрицы, раскраска и оценка размещения коэффициентов на GPU.
struct GpuMatrixPlan {
    BlockPattern pattern;              // Строки, столбцы и адреса внутренних рёбер.
    std::unique_ptr<CsgsOrder> order;  // Содержит точную раскраску графа этой сетки.
    CsgsLayout sweep;                  // Раскладка с выравниванием групп по 32 строки, если оно выгодно.
    std::vector<MeshIndex> first;     // Адреса коэффициентов матрицы в выбранной раскладке.
    MeshIndex bytes=0;                // Остаётся доступным также после передачи таблиц в решатель.
};

// Подготавливает структуру матрицы и раскраску перед выделением памяти GPU.
std::shared_ptr<GpuMatrixPlan> gpu_matrix_plan(const residual_detail::Geometry& g,MeshIndex rows,
                                               const Partition& part) {
    auto plan=std::make_shared<GpuMatrixPlan>();
    plan->pattern.build(g,rows,part);
    auto nodes=g.nodes;
#ifdef AERO_HAS_VISCOSITY
    if (part.host_aes.dim) nodes=part.host_aes.node_count;
#endif
    if (part.shared_csgs) plan->order=prepare_sweep(nodes,plan->pattern,true,plan->sweep);
    const auto width=g.components();
    coefficient_offsets(plan->pattern,width,mixed_linear,part.shared_csgs ? &plan->sweep : nullptr,plan->first);
    // Учитываем именно заполненные таблицы. Периодические рёбра могут делить один
    // блок; в CSGS размер массива включает пустые места для выравнивания цвета.
    const auto add=[&](MeshIndex count,MeshIndex item_bytes) {
        if (count && item_bytes>(std::numeric_limits<MeshIndex>::max()-plan->bytes)/count)
            throw std::length_error("GPU matrix memory size overflow");
        plan->bytes+=count*item_bytes;
    };
    // Подключает массив индексов в подходящем формате хранения матрицы.
    const auto indices=[&](const std::vector<MeshIndex>& data) {
        add(data.size(),indices_fit_32(data) ? sizeof(std::uint32_t) : sizeof(MeshIndex));
    };
    const auto& graph=plan->pattern;
    add(graph.rows.size(),sizeof(MeshIndex));
    indices(graph.columns);
    indices(graph.neighbour);
    indices(plan->first);
    add(rows,width*width*sizeof(double));
    const auto neighbour_count=part.shared_csgs ? plan->sweep.source.size() : graph.columns.size()-rows;
    add(neighbour_count,width*width*sizeof(LinearCoefficient));
    if (part.shared_csgs) {
        indices(plan->order->sequence);
        indices(plan->sweep.neighbours);
        add(rows,csgs_rows_fit_32(neighbour_count,plan->sweep.lanes) ? sizeof(CsgsRow32) : sizeof(CsgsRow));
        add(std::max<MeshIndex>(1,rows*width),sizeof(double));
    }
    return plan;
}

// Оценивает память GPU для запланированной структуры матрицы.
MeshIndex gpu_matrix_bytes(const GpuMatrixPlan& plan) { return plan.bytes; }

// Владельцы геометрии, блоков и буферов пространственной матрицы.
struct SpatialJacobian::Storage {
    std::shared_ptr<const SpatialResidual::Impl> geometry_owner; // Владелец адресов в blocks.grid.
    SpatialBlocks blocks{};    // Указатели на калькуляционные массивы выбранного исполнителя.
    // Устройство, на котором размещены блоки матрицы.
    int device = -1;
    bool ready = false;        // Некорректная площадка запрещает пользоваться частично обновлённой матрицей.
    Partition partition;      // Обмен данными и глобальные нормы данной подобласти.
    std::unique_ptr<CsgsOrder> order;  // Разбиение по цветам или потокам, только для CSGS.
    std::unique_ptr<CsgsOrder> reserved_order; // Готовая раскраска, ещё не загруженная при первой поправке.
    // Порядок строк для цветных проходов.
    CsgsLayout layout;
    // Представление упакованных коэффициентов.
    CsgsPacking packing{};
    std::vector<double> sweep_values; // Коэффициенты предобусловливателя для CPU.
    CsgsSweep sweep{};                // Постоянные адреса для проходов.
    std::vector<double> forward_host; // Промежуточный результат прямого прохода.
    // Промежуточный вектор прямого прохода.
    double* forward = nullptr;
#ifdef AERO_HAS_VISCOSITY
    // Вязкая часть линеаризованного оператора.
    viscosity::Jacobian viscous{};
    // Постоянная структура вязких связей матрицы.
    viscosity::JacobianPattern viscous_pattern;
    // Рабочие градиенты вязкого якобиана.
    std::vector<viscosity::Gradient> gradients;
#endif
    BlockPattern pattern;     // Номера строк, соседей и блоков на CPU.
    std::vector<double> diagonal; // По одному числу на уравнение и узел при сохранении матрицы.
    std::vector<double> entries, faces; // Коэффициенты всей матрицы и временные производные площадок на CPU.
    bool mixed = false;                 // Формат хранения матрицы поправки BiCGSTAB.
    std::vector<LinearCoefficient> neighbour_entries, sweep_neighbours; // Соседние блоки CSR и проходов CPU.
    std::vector<MeshIndex> first;       // Адрес в массиве диагонали или в массиве соседей.
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays; // Та же раскладка в памяти карты.
#endif
};

// Подготавливает структуру пространственной матрицы для заданной геометрии.
SpatialJacobian::SpatialJacobian(const SpatialResidual& residual, bool for_linear_solver) : storage_(new Storage) {
    auto& s = *storage_;
    s.geometry_owner = residual.p_;
    residual_detail::Geometry host;
    s.device = residual.jacobian_geometry(host, s.blocks.grid);
    allocate_blocks(host,for_linear_solver);
}
// Подготавливает структуру пространственной матрицы для заданной геометрии.
SpatialJacobian::SpatialJacobian(const residual_detail::Geometry& host, const residual_detail::Geometry& active,
                                     int device, const Partition& partition, bool for_linear_solver) : storage_(new Storage) {
    storage_->partition = partition;
    storage_->device = device;
    storage_->blocks.grid = active;
    allocate_blocks(host,for_linear_solver);
}
// Выделяет коэффициенты матрицы согласно подготовленной раскладке.
void SpatialJacobian::allocate_blocks(const residual_detail::Geometry& host, bool for_linear_solver) {
    auto& s = *storage_;
    const auto& partition = s.partition;
    auto& b = s.blocks;
    b.row_count = partition.extend_vector ? partition.rows : host.nodes;
    b.face_count = partition.share_blocks ? partition.faces : host.faces;
    b.computed = partition.computed;
    // При проверке бюджета уже построили эти таблицы на CPU. Перенос
    // владения в решатель исключает вторую раскраску многомиллионной сетки.
    const auto& reserved=partition.gpu_matrix_plan;
    const bool reuse=s.device>=0 && for_linear_solver && reserved && !reserved->pattern.rows.empty();
    if (reuse) {
        s.pattern=std::move(reserved->pattern);
        s.first=std::move(reserved->first);
        s.reserved_order=std::move(reserved->order);
        s.layout=std::move(reserved->sweep);
    } else {
        s.pattern.build(host,b.row_count,partition);
    }
    const auto stride = s.blocks.stride();
    const auto limit = s.entries.max_size() / stride;
    if (host.faces > limit || host.edges > limit - host.faces || s.pattern.columns.size() > limit)
        throw std::length_error("Spatial Jacobian: matrix too large");
    const auto entries = s.pattern.columns.size() * stride;
    s.mixed = mixed_linear && for_linear_solver;
    const auto doubles = s.mixed ? b.row_count * stride : entries;
    const auto singles = entries - doubles;
    // В начале каждой строки CSR ровно одна диагональ. При раздельном
    // хранении вычитаем число предшествующих диагоналей из адреса соседа.
    // Для совместной раскладки GPU+CSGS адреса позже задаст порядок проходов.
    if (!reuse && s.mixed && !(s.device >= 0 && s.partition.shared_csgs))
        coefficient_offsets(s.pattern,host.components(),s.mixed,nullptr,s.first);
    const auto faces = (host.faces + host.edges) * stride;
#ifdef AERO_HAS_VISCOSITY
    b.viscous = partition.host_aes.dim != 0;
    if (b.viscous) {
        b.direct_faces=s.device>=0 && (partition.active_aes.affine || partition.active_aes.tetra.cells) && partition.compact_jacobian &&
                       !partition.share_blocks && !partition.extend_vector && host.dim==3;
        if (!b.direct_faces)
            s.viscous_pattern.build(partition.host_aes,s.pattern.rows,s.pattern.columns,partition.compact_jacobian);
        s.viscous.aes = partition.active_aes;
        s.viscous.value_count = doubles;
        s.viscous.single_count = singles;
    }
#endif
#ifdef AERO_WITH_KOKKOS_GPU
    if (s.device >= 0) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        s.arrays.device = s.device;
        b.rows = s.arrays.allocate(s.pattern.rows.size(), s.pattern.rows.data());
        b.columns = upload_indices(s.arrays,s.pattern.columns);
        b.neighbour = upload_indices(s.arrays,s.pattern.neighbour);
        if (!s.partition.shared_csgs) {
            b.values = s.arrays.allocate<double>(doubles);
            if (s.mixed) {
                b.layout = {upload_indices(s.arrays,s.first),1,s.arrays.allocate<LinearCoefficient>(singles)};
                std::vector<MeshIndex>().swap(s.first);
            }
        }
        // Вязкий Якобиан расходует градиенты до вычисления производных потока
        // на рёбрах. Обе стадии используют одну выровненную область на GPU.
        std::size_t scratch_bytes=b.direct_faces ? 0 : faces*sizeof(double);
#ifdef AERO_HAS_VISCOSITY
        if (b.viscous) {
            const auto& pattern = s.viscous_pattern;
            if (b.direct_faces) {
                s.viscous.row_offsets=b.rows;
                s.viscous.column_nodes=b.columns;
            } else {
                s.viscous.offsets=s.arrays.upload(pattern.offsets);
                s.viscous.elements=s.arrays.upload(pattern.elements);
            }
            // На целой сетке можно собрать столбцы той же группой элементов,
            // что и невязку. Градиенты потребляются до перехода к следующей группе.
            if (b.direct_faces)
                s.viscous.gradients = s.viscous.aes.gradient_cache;
            const auto vertices = s.viscous.gradients ? 0 : s.viscous.aes.vertex_count;
            if (vertices > std::numeric_limits<std::size_t>::max() / sizeof(viscosity::Gradient))
                throw std::length_error("Spatial Jacobian: gradient array too large");
            scratch_bytes = std::max(scratch_bytes, static_cast<std::size_t>(vertices) * sizeof(viscosity::Gradient));
            s.viscous.values = b.values;
            s.viscous.layout = b.layout;
            s.viscous_pattern = {};
        }
#endif
        auto* scratch = s.arrays.allocate<unsigned char>(scratch_bytes);
        b.face_values = reinterpret_cast<double*>(scratch);
#ifdef AERO_HAS_VISCOSITY
        if (b.viscous && !s.viscous.gradients)
            s.viscous.gradients = reinterpret_cast<viscosity::Gradient*>(scratch);
#endif
        // Для раскраски до первой сборки нужны только строки и столбцы.
        // Начальную норму steady солвер вычисляет ещё до запроса самого Якобиана.
        if (s.partition.shared_csgs) std::vector<MeshIndex>().swap(s.pattern.neighbour);
        else s.pattern = BlockPattern{};
        return;
    }
#endif
    s.entries.resize(doubles);
    if (s.mixed) {
        s.neighbour_entries.resize(singles);
        b.layout = {s.first.data(),1,s.neighbour_entries.data()};
    }
    s.faces.resize(faces);
    b.rows = s.pattern.rows.data();
    b.columns = s.pattern.columns.data();
    b.neighbour = s.pattern.neighbour.data();
    b.values = s.entries.data();
    b.face_values = s.faces.data();
#ifdef AERO_HAS_VISCOSITY
    if (b.viscous) {
        s.viscous.offsets = s.viscous_pattern.offsets.data();
        s.viscous.elements = s.viscous_pattern.elements.data();
        s.gradients.resize(s.viscous.aes.vertex_count);
        s.viscous.gradients = s.gradients.data();
        s.viscous.values = b.values;
        s.viscous.layout = b.layout;
    }
#endif
}
// Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
SpatialJacobian::~SpatialJacobian() = default;

// Поля лежат на исполнителе невязки. Начальное поле фиксировано при
// дифференцировании и может отсутствовать, если его не требует граница.
void SpatialJacobian::assemble(const double* u, const double* initial) {
    assemble_matrix(u, initial, nullptr, 0.);
}
// Возвращает число узлов в текущем представлении.
MeshIndex SpatialJacobian::node_count() const { return storage_->blocks.row_count; }
// Возвращает обмены и редукции распределённой матрицы.
const Partition& SpatialJacobian::partition() const { return storage_->partition; }
// Возвращает пространственную размерность обрабатываемой сущности.
unsigned SpatialJacobian::dimension() const { return storage_->blocks.grid.dim; }
// Возвращает номер устройства; отрицательное значение соответствует CPU.
int SpatialJacobian::device() const { return storage_->device; }

// Подготавливает порядок обхода и упакованные блоки предобусловливателя CSGS.
void SpatialJacobian::prepare_csgs() {
    auto& s = *storage_;
    auto& b = s.blocks;
    // Пересчитывает зависимые потоки или коэффициенты после изменения поля.
    const auto refresh = [&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (s.device>=0) {
            if (s.partition.shared_csgs) return; // Сборка уже пишет в коэффициенты обоих проходов.
            device_work::check(device_work::aero_device_pack_csgs(s.device,s.packing,b.values));
            return;
        }
#endif
        host_parallel::ensure_runtime();
        matrix_rows::pack(host_parallel::Execution(),s.packing,b.values);
    };
    if (s.order) { refresh(); return; }
    const BlockPattern* graph = &s.pattern;
#ifdef AERO_WITH_KOKKOS_GPU
    // Раскраска выполняется один раз на CPU. Забираем только индексы CSR,
    // поэтому расчёт с BJ не хранит ни раскраску, ни вторую копию графа.
    BlockPattern downloaded;
    if (s.device >= 0 && s.pattern.rows.empty()) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        downloaded.rows.resize(b.row_count + 1);
        gpu::checked(gpu::copy(downloaded.rows.data(), b.rows, downloaded.rows.size() * sizeof(MeshIndex),
                               gpu::Copy::ToHost));
        downloaded.columns.resize(downloaded.rows.back());
        read_indices(b.columns,downloaded.columns);
        graph = &downloaded;
    }
#endif
    MeshIndex state_nodes = b.grid.nodes;
#ifdef AERO_HAS_VISCOSITY
    if (b.viscous) state_nodes = s.viscous.aes.node_count;
#endif
    auto order=std::move(s.reserved_order);
    if (!order) order=prepare_sweep(state_nodes,*graph,s.device>=0,s.layout);
    auto& view = s.sweep;
    auto& plan=s.packing;
    plan.input=b.layout;
    view.external=std::any_of(s.layout.neighbours.begin(),s.layout.neighbours.end(),
                              [](MeshIndex n){return (n&1)!=0;});
    plan.count=s.layout.source.size();
    plan.width=b.grid.components();
    plan.lanes=view.lanes=s.layout.lanes;
    if (plan.count>std::numeric_limits<MeshIndex>::max()/sizeof(double)/plan.width/plan.width)
        throw std::length_error("CSGS coefficient array too large");
    const auto coefficients=plan.count*plan.width*plan.width;
    const MeshIndex size=std::max<MeshIndex>(1,b.row_count*plan.width);
#ifdef AERO_WITH_KOKKOS_GPU
    if (s.device >= 0) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        view.sequence=upload_indices(s.arrays,order->sequence);
        // Начало даже пустой строки может указывать чуть дальше последнего
        // коэффициента (номер полосы внутри пакета). Это тоже должно поместиться.
        if (csgs_rows_fit_32(plan.count,plan.lanes)) {
            std::vector<CsgsRow32> short_rows(s.layout.rows.size());
            parallel_for(short_rows.size(),[&](MeshIndex i) {
                for (unsigned pass=0; pass<2; ++pass) {
                    short_rows[i].first[pass]=static_cast<std::uint32_t>(s.layout.rows[i].first[pass]);
                    short_rows[i].count[pass]=static_cast<std::uint32_t>(s.layout.rows[i].count[pass]);
                }
            });
            view.short_rows=s.arrays.upload(short_rows);
        } else {
            view.rows=s.arrays.upload(s.layout.rows);
        }
        view.neighbours=upload_indices(s.arrays,s.layout.neighbours);
        if (s.partition.shared_csgs) {
            // Диагонали идут подряд в начале массива для LU. За ними лежат
            // внедиагональные блоки в том же порядке, который читают ядра CSGS.
            const auto diagonal = b.row_count * b.stride();
            if (diagonal > std::numeric_limits<std::size_t>::max()/sizeof(double)-coefficients)
                throw std::length_error("CSGS matrix too large");
            const auto double_count = s.mixed ? diagonal : diagonal + coefficients;
            b.values = s.arrays.allocate<double>(double_count);
            if (s.mixed) plan.neighbour_values = s.arrays.allocate<LinearCoefficient>(coefficients);
            else plan.values = b.values ? b.values + diagonal : nullptr;
            if (s.first.empty()) coefficient_offsets(*graph,plan.width,s.mixed,&s.layout,s.first);
            b.layout = {upload_indices(s.arrays,s.first), plan.lanes, plan.neighbour_values};
            std::vector<MeshIndex>().swap(s.first);
#ifdef AERO_HAS_VISCOSITY
            // Обнуление перед вязкой сборкой должно охватить также промежутки
            // выравнивания строк CSGS. Иначе в конце последнего пакета останутся старые значения.
            s.viscous.value_count = double_count;
            s.viscous.single_count = s.mixed ? coefficients : 0;
            s.viscous.layout = b.layout;
            s.viscous.values = b.values;
#endif
        } else {
            plan.source=s.arrays.upload(s.layout.source);
            if (s.mixed) plan.neighbour_values = s.arrays.allocate<LinearCoefficient>(coefficients);
            else plan.values=s.arrays.allocate<double>(coefficients);
        }
        s.forward=s.arrays.allocate<double>(size);
        s.layout=CsgsLayout{};
    } else
#endif
    {
        s.forward_host.resize(size);
        s.forward=s.forward_host.data();
        view.sequence=order->sequence.data();
        view.rows=s.layout.rows.data();
        view.neighbours=s.layout.neighbours.data();
        plan.source=s.layout.source.data();
        if (s.mixed) {
            s.sweep_neighbours.resize(coefficients);
            plan.neighbour_values=s.sweep_neighbours.data();
        } else {
            s.sweep_values.resize(coefficients);
            plan.values=s.sweep_values.data();
        }
    }
    view.blocks=plan.values;
    view.neighbour_values=plan.neighbour_values;
    refresh();
    // Разделение графа уже записано в строках проходов. Для запусков
    // сохраняем только границы групп и, на CPU, порядок посещения.
    std::vector<MeshIndex>().swap(order->position);
    std::vector<MeshIndex>().swap(order->row_of_node);
    std::vector<MeshIndex>().swap(order->domain);
    if (s.device>=0) std::vector<MeshIndex>().swap(order->sequence);
    s.order = std::move(order);
}

// Выполняет прямой и обратный проходы CSGS с необходимыми обменами.
void SpatialJacobian::apply_csgs(const BlockJacobi& diagonal, const double* rhs, double* output,double* image) const {
    const auto& s = *storage_;
    auto sweep = s.sweep;
    sweep.diagonal = diagonal.factors();
    sweep.rhs = rhs;
    sweep.middle_rhs = output; // В прямом проходе выход ещё свободен для b−L*y.
    const auto& offsets = s.order->offsets;
    const MeshIndex groups = offsets.size() - 1;
    for (unsigned pass = 0; pass < 2; ++pass) {
        sweep.backward = pass == 1;
        sweep.output = pass ? output : s.forward;
        // Все ранги, включая пустые, выполняют один обмен между проходами.
        // При работе без MPI исходная нумерация совпадает с номерами строк.
        if (pass)
            sweep.forward_with_halo =
                s.partition.extend_vector ? s.partition.extend_vector(s.forward) : s.forward;
        s.partition.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
            if (s.device >= 0) {
                for (MeshIndex j = 0; j < groups; ++j) {
                    const auto color = pass ? groups - 1 - j : j;
                    device_work::check(device_work::aero_device_csgs_color(
                        s.device, sweep, offsets[color], offsets[color + 1] - offsets[color]));
                }
                return;
            }
#endif
            parallel_for(
                groups,
                [&](MeshIndex group) {
                    const auto begin = offsets[group], end = offsets[group + 1];
                    for (auto k = begin; k < end; ++k)
                        sweep(pass ? end - 1 - (k - begin) : k);
                },
                2);
        });
    }
    // В MPI соседние компоненты z требуют полноценного обмена. На цельной
    // сетке обе подстановки уже дали достаточно данных для умножения.
    if (!image) return;
    if (s.partition.extend_vector || s.partition.host_owned) { apply(output,image); return; }
    s.partition.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
        if (s.device>=0) {
            device_work::check(device_work::aero_device_csgs_image(s.device,sweep,s.blocks.row_count,image));
            return;
        }
#endif
        host_parallel::ensure_runtime();
        matrix_rows::image(host_parallel::Execution(),sweep,s.blocks.row_count,image);
    });
}

// Сохраняет пространственную диагональ для обновления временного вклада.
void SpatialJacobian::reserve_diagonal() {
    auto& s = *storage_;
    auto& b = s.blocks;
    if (b.spatial_diagonal || !b.row_count) return;
    const auto count = b.row_count * b.grid.components();
#ifdef AERO_WITH_KOKKOS_GPU
    if (s.device >= 0) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        b.spatial_diagonal = s.arrays.allocate<double>(count);
        return;
    }
#endif
    s.diagonal.resize(count);
    b.spatial_diagonal = s.diagonal.data();
}
// Обновляет временной вклад диагонали при повторном использовании пространственной матрицы.
void SpatialJacobian::update_time_diagonal(const double* volumes, double dt) {
    auto& s = *storage_;
    auto& b = s.blocks;
    if (!b.row_count) return;
    // При смене режима перестраивается только временная часть диагонали и её разложение.
    b.local_steps = SteadyNodeSteps::active(s.partition.steady_steps);
    if (!b.spatial_diagonal || !volumes) throw std::logic_error("BDF: missing cached diagonal or volumes");
    assemble_checked_rows<true>(s.device, b, volumes, dt);
}

// Подготавливает разложение диагональных блоков текущей матрицы.
void SpatialJacobian::prepare_jacobi(BlockJacobi& jacobi) const {
    const auto& b = storage_->blocks;
    jacobi.prepare(b.values, b.layout.first ? nullptr : b.rows);
}

// Собирает пространственные производные и временной вклад матрицы шага.
void SpatialJacobian::assemble_matrix(const double* u, const double* initial, const double* volumes, double dt,
                                     bool simplified, bool roe, const PhaseTiming& timing) {
#ifndef AERO_HAS_VISCOSITY
    (void)timing;
#endif
    auto& s = *storage_;
    s.ready = false;
    auto& b = s.blocks;
    b.local_steps = SteadyNodeSteps::active(s.partition.steady_steps);
    b.simplified = simplified;
    b.roe = roe;
    // Только владелец площадки дифференцирует поток. До обмена все ранги
    // согласуют ошибки, в том числе ранги без единственной собственной строки.
    s.partition.local([&] {
        if ((b.face_count || b.row_count) && !u) throw std::invalid_argument("Spatial Jacobian: missing state");
#ifdef AERO_WITH_KOKKOS_GPU
        // Раскраска и выделение коэффициентов относятся к первой поправке Ньютона.
        // Это сохраняет границы замера шага, по которым auto LTS регулирует рост dt.
        if (s.device >= 0 && s.partition.shared_csgs && !s.order) {
            prepare_csgs();
            s.pattern = BlockPattern{};
        }
#endif
#ifdef AERO_HAS_VISCOSITY
        if (b.viscous && b.row_count) timing.measure(PhaseTiming::ViscousMatrix,[&] {
            // Заполняем вязкую часть первой. Строки конвекции затем добавят свои блоки,
            // закрепят импульс на стенке и сохранят полную пространственную диагональ.
            const auto& v = s.viscous;
#ifdef AERO_WITH_KOKKOS_GPU
            if (s.device >= 0) {
                device_work::check(device_work::aero_device_viscous_matrix(s.device,v,u));
            } else
#endif
            {
                host_parallel::ensure_runtime();
                const host_parallel::Execution exec;
                viscosity::assemble_jacobian(v,u,viscosity::execution::MatrixAssembly<host_parallel::Execution>{exec});
            }
        });
#endif
        if (b.direct_faces) {
            b.assembly_state=u;
            b.assembly_initial=initial;
            return;
        }
        // Теперь все вязкие производные записаны в CSR. Буфер градиентов свободен
        // до следующей сборки. Запуски Kokkos идут последовательно в одном потоке GPU.
#ifdef AERO_WITH_KOKKOS_GPU
        if (s.device >= 0 && b.face_count) {
            MeshIndex bad;
            device_work::check(device_work::aero_device_spatial_faces(s.device, b, u, initial, &bad));
            if (bad < b.grid.faces) throw assembly_error(bad, b.grid);
            return;
        }
#endif
        if (b.face_count) {
            host_parallel::ensure_runtime();
            MeshIndex bad;
            matrix_rows::differentiate(host_parallel::Execution(),b,u,initial,&bad);
            if (bad < b.grid.faces) throw assembly_error(bad,b.grid);
        }
    });
    if (s.partition.share_blocks) s.partition.share_blocks(b.face_values);
    s.partition.local([&] {
        assemble_checked_rows<false>(s.device, b, volumes, dt);
    });
    s.ready = true;
}
// Умножает последнюю успешно собранную матрицу на вектор. Массивы
// размером nodes*(dimension+2) должны занимать разные области памяти.
void SpatialJacobian::apply(const double* x, double* y) const {
    const auto& s = *storage_;
    if (!s.ready) throw std::logic_error("Spatial Jacobian: assemble before applying");
    s.partition.local([&] {
        if (s.blocks.row_count && (!x || !y || x == y))
            throw std::invalid_argument("Spatial Jacobian: separate input and output required");
    });
    // Расширенный вход содержит значения соседних узлов, выход — только свои строки.
    const double* extended = s.partition.extend_vector ? s.partition.extend_vector(x) : x;
    s.partition.local([&] {
        if (!s.blocks.row_count) return;
#ifdef AERO_WITH_KOKKOS_GPU
        if (s.device >= 0) {
            device_work::check(device_work::aero_device_spatial_apply(s.device, s.blocks, extended, y));
            return;
        }
#endif
        host_parallel::ensure_runtime();
        matrix_rows::multiply(host_parallel::Execution(),s.blocks,extended,y);
    });
}
}
}
