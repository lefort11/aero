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
#endif
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace aero {
namespace implicit {
namespace {
// Привязка вхождения ребра в строку к столбцу соседнего узла вычисляется
// один раз. При обновлении коэффициентов поиск по номерам узлов уже не нужен.
struct BlockPattern {
    std::vector<MeshIndex> rows, columns, neighbour;
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
            if (part.host_aes.dim) viscosity::append_columns(part.host_aes,i,columns);
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
std::domain_error assembly_error(MeshIndex bad, const residual_detail::Geometry& g) {
    if (bad < g.edges) return std::domain_error("Spatial Jacobian: invalid edge " + std::to_string(bad));
    if (bad < g.faces) return std::domain_error("Spatial Jacobian: invalid boundary piece " + std::to_string(bad-g.edges));
    return std::domain_error("Spatial Jacobian: invalid time coefficient or nonfinite entry at node " +
                             std::to_string(bad-g.faces));
}
}

struct SpatialJacobian::Storage {
    std::shared_ptr<const SpatialResidual::Impl> geometry_owner; // Владелец адресов в blocks.grid.
    SpatialBlocks blocks{};    // Указатели на калькуляционные массивы выбранного исполнителя.
    int device = -1;
    bool ready = false;        // Некорректная площадка запрещает пользоваться частично обновлённой матрицей.
    Partition partition;      // Обмен данными и глобальные нормы данной подобласти.
    std::unique_ptr<CsgsOrder> order;  // Разбиение по цветам или потокам, только для CSGS.
    CsgsSweep sweep{};                // Постоянные адреса для проходов.
    std::vector<double> forward_host; // Промежуточный результат прямого прохода.
    double* forward = nullptr;
#ifdef AERO_HAS_VISCOSITY
    viscosity::Jacobian viscous{};
    viscosity::JacobianPattern viscous_pattern;
    std::vector<viscosity::Gradient> gradients;
#endif
    BlockPattern pattern;     // Номера строк, соседей и блоков на CPU.
    std::vector<double> diagonal; // По одному числу на уравнение и узел при сохранении матрицы.
    std::vector<double> entries, faces; // Коэффициенты всей матрицы и временные производные площадок на CPU.
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays; // Та же раскладка в памяти карты.
#endif
};

SpatialJacobian::SpatialJacobian(const SpatialResidual& residual) : storage_(new Storage) {
    auto& s = *storage_;
    s.geometry_owner = residual.p_;
    residual_detail::Geometry host;
    s.device = residual.jacobian_geometry(host, s.blocks.grid);
    allocate_blocks(host);
}
SpatialJacobian::SpatialJacobian(const residual_detail::Geometry& host, const residual_detail::Geometry& active,
                                     int device, const Partition& partition) : storage_(new Storage) {
    storage_->partition = partition;
    storage_->device = device;
    storage_->blocks.grid = active;
    allocate_blocks(host);
}
void SpatialJacobian::allocate_blocks(const residual_detail::Geometry& host) {
    auto& s = *storage_;
    const auto& partition = s.partition;
    auto& b = s.blocks;
    b.row_count = partition.extend_vector ? partition.rows : host.nodes;
    b.face_count = partition.share_blocks ? partition.faces : host.faces;
    b.computed = partition.computed;
    s.pattern.build(host, b.row_count, partition);
    const auto stride = s.blocks.stride();
    const auto limit = s.entries.max_size() / stride;
    if (host.faces > limit || host.edges > limit - host.faces || s.pattern.columns.size() > limit)
        throw std::length_error("Spatial Jacobian: matrix too large");
    const auto entries = s.pattern.columns.size() * stride;
    const auto faces = (host.faces + host.edges) * stride;
#ifdef AERO_HAS_VISCOSITY
    b.viscous = partition.host_aes.dim != 0;
    if (b.viscous) {
        s.viscous_pattern.build(partition.host_aes,s.pattern.rows,s.pattern.columns);
        s.viscous.aes = partition.active_aes;
        s.viscous.value_count = entries;
    }
#endif
#ifdef AERO_WITH_KOKKOS_GPU
    if (s.device >= 0) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        s.arrays.device = s.device;
        b.rows = s.arrays.allocate(s.pattern.rows.size(), s.pattern.rows.data());
        b.columns = s.arrays.allocate(s.pattern.columns.size(), s.pattern.columns.data());
        b.neighbour = s.arrays.allocate(s.pattern.neighbour.size(), s.pattern.neighbour.data());
        b.values = s.arrays.allocate<double>(entries);
        b.face_values = s.arrays.allocate<double>(faces);
#ifdef AERO_HAS_VISCOSITY
        if (b.viscous) {
            const auto& pattern = s.viscous_pattern;
            s.viscous.offsets = s.arrays.upload(pattern.offsets);
            s.viscous.elements = s.arrays.upload(pattern.elements);
            s.viscous.gradients = s.arrays.allocate<viscosity::Gradient>(s.viscous.aes.vertex_count);
            s.viscous.values = b.values;
            s.viscous_pattern = {};
        }
#endif
        s.pattern = BlockPattern{}; // На GPU портрет матрицы хранится только на карте.
        return;
    }
#endif
    s.entries.resize(entries);
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
    }
#endif
}
SpatialJacobian::~SpatialJacobian() = default;

void SpatialJacobian::assemble(const double* u, const double* initial) {
    assemble_matrix(u, initial, nullptr, 0.);
}
MeshIndex SpatialJacobian::node_count() const { return storage_->blocks.row_count; }
const Partition& SpatialJacobian::partition() const { return storage_->partition; }
unsigned SpatialJacobian::dimension() const { return storage_->blocks.grid.dim; }
int SpatialJacobian::device() const { return storage_->device; }

void SpatialJacobian::prepare_csgs() {
    auto& s = *storage_;
    if (s.order) return;
    const auto& b = s.blocks;
    const BlockPattern* graph = &s.pattern;
#ifdef AERO_WITH_KOKKOS_GPU
    // Раскраска выполняется один раз на CPU. Забираем только индексы CSR,
    // поэтому расчёт с BJ не хранит ни раскраску, ни вторую копию графа.
    BlockPattern downloaded;
    if (s.device >= 0) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        downloaded.rows.resize(b.row_count + 1);
        gpu::checked(gpu::copy(downloaded.rows.data(), b.rows, downloaded.rows.size() * sizeof(MeshIndex),
                               gpu::Copy::ToHost));
        downloaded.columns.resize(downloaded.rows.back());
        if (!downloaded.columns.empty())
            gpu::checked(gpu::copy(downloaded.columns.data(), b.columns,
                                   downloaded.columns.size() * sizeof(MeshIndex), gpu::Copy::ToHost));
        graph = &downloaded;
    }
#endif
    host_parallel::ensure_runtime();
    MeshIndex state_nodes = b.grid.nodes;
#ifdef AERO_HAS_VISCOSITY
    if (b.viscous) state_nodes = s.viscous.aes.node_count;
#endif
    std::unique_ptr<CsgsOrder> order(
        new CsgsOrder(state_nodes, graph->rows, graph->columns, s.device >= 0,
                      static_cast<MeshIndex>(host_parallel::Execution().concurrency())));
    auto& view = s.sweep;
    view.rows = b.rows;
    view.columns = b.columns;
    view.blocks = b.values;
    const MeshIndex size = std::max<MeshIndex>(1, b.row_count * b.grid.components());
#ifdef AERO_WITH_KOKKOS_GPU
    if (s.device >= 0) {
        gpu::DeviceScope scope(s.device);
        gpu::checked(scope.status);
        view.sequence = s.arrays.allocate(order->sequence.size(), order->sequence.data());
        view.position = s.arrays.allocate(order->position.size(), order->position.data());
        view.row_of_node = s.arrays.allocate(order->row_of_node.size(), order->row_of_node.data());
        s.forward = s.arrays.allocate<double>(size);
        // После загрузки на карте остаётся вся индексация, CPU нужны только границы цветов.
        std::vector<MeshIndex>().swap(order->sequence);
        std::vector<MeshIndex>().swap(order->position);
        std::vector<MeshIndex>().swap(order->row_of_node);
    } else
#endif
    {
        s.forward_host.resize(size);
        s.forward = s.forward_host.data();
        view.sequence = order->sequence.data();
        view.position = order->position.data();
        view.row_of_node = order->row_of_node.data();
        view.domain = order->domain.data();
    }
    s.order = std::move(order);
}

void SpatialJacobian::apply_csgs(const BlockJacobi& diagonal, const double* rhs, double* output) const {
    const auto& s = *storage_;
    auto sweep = s.sweep;
    sweep.diagonal = diagonal.factors();
    sweep.rhs = rhs;
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
}

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
void SpatialJacobian::update_time_diagonal(const double* volumes, double dt) {
    auto& s = *storage_;
    auto& b = s.blocks;
    if (!b.row_count) return;
    // При смене режима перестраивается только временная часть диагонали и её разложение.
    b.local_steps = SteadyNodeSteps::active(s.partition.steady_steps);
    if (!b.spatial_diagonal || !volumes) throw std::logic_error("BDF: missing cached diagonal or volumes");
#ifdef AERO_WITH_KOKKOS_GPU
    if (s.device >= 0) {
        MeshIndex bad;
        device_work::check(device_work::aero_device_time_diagonal(s.device, b, volumes, dt, &bad));
        if (bad < b.grid.nodes) throw assembly_error(b.grid.faces + bad, b.grid);
        return;
    }
#endif
    parallel_for(b.row_count, [&](MeshIndex row) {
        if (!b.update_time_diagonal(row, volumes, dt)) throw assembly_error(b.grid.faces + b.node(row), b.grid);
    });
}

void SpatialJacobian::prepare_jacobi(BlockJacobi& jacobi) const {
    jacobi.prepare(storage_->blocks.values, storage_->blocks.rows);
}

void SpatialJacobian::assemble_matrix(const double* u, const double* initial, const double* volumes, double dt, bool simplified, bool roe) {
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
        if (s.device >= 0 && b.face_count) {
            MeshIndex bad;
            device_work::check(device_work::aero_device_spatial_faces(s.device, b, u, initial, &bad));
            if (bad < b.grid.faces) throw assembly_error(bad, b.grid);
            return;
        }
#endif
        parallel_for(b.face_count, [&](MeshIndex k) {
            const auto f = b.face(k);
            if (!b.differentiate(f, u, initial)) throw assembly_error(f, b.grid);
        });
    });
    if (s.partition.share_blocks) s.partition.share_blocks(b.face_values);
    s.partition.local([&] {
#ifdef AERO_HAS_VISCOSITY
        if (b.viscous && b.row_count) {
            // Заполняем вязкую часть первой. Строки конвекции затем добавят свои блоки,
            // закрепят импульс на стенке и сохранят полную пространственную диагональ.
            const auto& v = s.viscous;
#ifdef AERO_WITH_KOKKOS_GPU
            if (s.device >= 0) {
                device_work::check(device_work::aero_device_viscous_matrix(s.device,v,u));
            } else
#endif
            {
                parallel_for(b.row_count,[&](MeshIndex row) {
                    std::fill(v.values+b.rows[row]*b.stride(),v.values+b.rows[row+1]*b.stride(),0.);
                });
                parallel_for(v.aes.node_count,[&](MeshIndex n) { v.aes.prepare_gas(n,u,nullptr); });
                parallel_for(v.aes.cell_count,[&](MeshIndex e) { v.prepare_cell(e); });
                parallel_for(v.aes.node_count,[&](MeshIndex n) {
                    for (unsigned c = 0; c < b.grid.components(); ++c) v.assemble_column(n,c);
                });
            }
        }
#endif
#ifdef AERO_WITH_KOKKOS_GPU
        if (s.device >= 0 && b.row_count) {
            MeshIndex bad;
            device_work::check(device_work::aero_device_spatial_rows(s.device, b, volumes, dt, &bad));
            if (bad < b.grid.nodes) throw assembly_error(b.grid.faces + bad, b.grid);
            return;
        }
#endif
        parallel_for(b.row_count, [&](MeshIndex row) {
            if (!b.assemble_row(row, volumes, dt)) throw assembly_error(b.grid.faces + b.node(row), b.grid);
        });
    });
    s.ready = true;
}
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
        parallel_for(s.blocks.row_count, [&](MeshIndex row) { s.blocks.multiply_row(row, extended, y); });
    });
}
}
}
