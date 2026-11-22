#include "aes.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <set>

namespace aero {
namespace viscosity {
namespace {
Element describe(const MeshPartition& part, const PeriodicPartition* seam, const ResidualFrame& frame,
                 MeshIndex local) {
    Element e{};
    const auto mesh = part.mesh.view();
    const auto vertices = mesh.cell_nodes(local);
    e.id = part.cells.empty() ? local : part.cells[local];
    if (vertices.size > 8) throw MeshError("AES: unsupported cell at " + std::to_string(e.id));
    e.count = static_cast<unsigned>(vertices.size);
    const auto origin = mesh.coordinate(vertices[0]);
    for (unsigned k = 0; k < e.count; ++k) {
        const auto node = vertices[k];
        const auto slot = seam ? seam->node_dof[node] : node;
        e.nodes[k] = seam ? seam->ids[slot] : part.nodes[node];
        e.owners[k] = seam ? seam->owners[slot] : part.node_owners[node];
        const Coordinate dx = mesh.coordinate(node) - origin;
        for (unsigned m = 0; m < mesh.dimension; ++m) {
            const auto a = frame.axis[m];
            e.position[k][m] = a.x*dx[0] + a.y*dx[1] + a.z*dx[2];
        }
    }
    for (auto edge : mesh.cell_edges(local)) {
        const auto a = std::find(vertices.begin(), vertices.end(), mesh.edge_begin[edge]) - vertices.begin();
        const auto b = std::find(vertices.begin(), vertices.end(), mesh.edge_end[edge]) - vertices.begin();
        e.neighbours[a] |= 1u << b;
        e.neighbours[b] |= 1u << a;
    }
    return e;
}

Cell prepare_cell(GradientBuilder& gradients, const Element& e, unsigned dim, double gamma, double prandtl,
                  std::vector<PairMetric>& metrics, MeanGradient* mean = nullptr) {
    Cell cell{};
    cell.count = e.count;
    cell.inverse_count = 1. / e.count;
    gradients.prepare(e, dim, gamma, prandtl, cell, metrics, mean);
    return cell;
}

}

// Обрабатываем независимые диапазоны элементов. Ограничение в 64 блока
// сохраняет кэш одинаковых форм внутри каждого диапазона и ограничивает память.
void Aes::prepare_local_cells(const MeshPartition& part, const PeriodicPartition* seam,
                              const ResidualFrame& frame, const Locate& locate) {
    const auto mesh = part.mesh.view();
    // Все узлы принадлежат этому процессу. Подготавливаем отображение один раз,
    // в том числе для периодической сетки с несколькими копиями одного узла.
    std::vector<MeshIndex> slots(mesh.node_count);
    for (MeshIndex n = 0; n < mesh.node_count; ++n) {
        const MeshIndex global = seam ? seam->ids[seam->node_dof[n]] : part.nodes[n];
        slots[n] = locate(global,0);
    }
    const MeshIndex blocks = std::min<MeshIndex>(64, (mesh.cell_count + 511) / 512);
    std::vector<std::vector<PairMetric>> pairs(blocks);
    cells_.resize(mesh.cell_count);
    parallel_for(blocks, [&](MeshIndex b) {
        // Каждый блок имеет собственный кэш повторяющихся форм. Поэтому сборка
        // интегралов по элементам не блокирует остальные потоки.
        GradientBuilder cache;
        for (MeshIndex c = mesh.cell_count*b/blocks; c < mesh.cell_count*(b+1)/blocks; ++c) {
            const Element e = describe(part,seam,frame,c);
            auto& cell = cells_[c];
            cell = prepare_cell(cache,e,mesh.dimension,op_.law.gamma,op_.law.prandtl,pairs[b]);
            const auto nodes = mesh.cell_nodes(c);
            for (unsigned i = 0; i < cell.count; ++i) cell.node[i] = slots[nodes[i]];
        }
    }, 2);
    merge_cell_metrics(pairs);
}

void Aes::merge_cell_metrics(std::vector<std::vector<PairMetric>>& pairs) {
    const MeshIndex blocks = pairs.size();
    // Сумма длин предыдущих блоков даёт адрес первой пары данного блока.
    std::vector<MeshIndex> base(blocks + 1);
    for (MeshIndex b = 0; b < blocks; ++b) base[b+1] = base[b] + pairs[b].size();
    metrics_.resize(base.back());
    parallel_for(blocks, [&](MeshIndex b) {
        std::copy(pairs[b].begin(), pairs[b].end(), metrics_.begin() + base[b]);
        for (MeshIndex c = cells_.size()*b/blocks; c < cells_.size()*(b+1)/blocks; ++c)
            cells_[c].first_pair += base[b];
    }, 2);
}

void Aes::index_local_vertices() {
    // Формируем строки в исходном порядке ячеек: порядок сложения вязкой невязки сохраняется.
    offsets_.assign(op_.row_count+1,0);
    for (auto& cell : cells_) {
        cell.first_vertex = op_.vertex_count;
        op_.vertex_count += cell.count;
        op_.neighbour_count += cell.count*(cell.count-1);
        for (unsigned i = 0; i < cell.count; ++i) ++offsets_[cell.node[i]+1];
    }
    std::partial_sum(offsets_.begin(),offsets_.end(),offsets_.begin());
    incidence_.resize(offsets_.back());
    auto next = offsets_;
    for (MeshIndex c = 0; c < cells_.size(); ++c) {
        const auto& cell = cells_[c];
        for (unsigned i = 0; i < cell.count; ++i) incidence_[next[cell.node[i]]++] = 8*c+i;
    }
}

void Aes::prepare(const MeshPartition& part, const PeriodicPartition* seam, const ResidualFrame& frame,
                  const Processes& world, const ViscositySettings& settings, double gamma, double mach,
                  MeshIndex rows, const Locate& locate) {
    const auto mesh = part.mesh.view();
    op_.dim = mesh.dimension;
    op_.row_count = rows;
    op_.law = {gamma,gamma*mach*mach,1./settings.reynolds,settings.prandtl,
               settings.law == "sutherland" ? settings.sutherland_s/settings.t_amb : 0.};
    std::vector<std::vector<MeshIndex>> by_node;
    bool parallel_cells = world.size == 1 && mesh.cell_count >= 8192;
#ifdef AERO_HAS_TURBULENCE
    parallel_cells = parallel_cells && !op_.computes_les();
#endif
    if (!parallel_cells) world.local([&] { by_node.resize(rows); });
    GradientBuilder gradients;
#ifdef AERO_HAS_TURBULENCE
    // Индекс первой геометрической пары AES уже служит точным ключом формы:
    // порядок вершин, связность и относительные координаты при совпадении одинаковы.
    std::map<MeshIndex,MeshIndex> filter_by_shape;
#endif
    const auto add = [&](const Element& e) {
        MeanGradient* requested=nullptr;
#ifdef AERO_HAS_TURBULENCE
        MeanGradient mean{};
        if (op_.computes_les()) requested=&mean;
#endif
        Cell cell = prepare_cell(gradients,e,mesh.dimension,gamma,op_.law.prandtl,metrics_,requested);
#ifdef AERO_HAS_TURBULENCE
        if (requested) {
            auto shape=filter_by_shape.emplace(cell.first_pair,filters_.size());
            filter_indices_.push_back(shape.first->second);
            if (shape.second) {
                turbulence::FilterCell filter{};
                filter.volume=mean.volume;
                filter.vol_length=mesh.dimension==3 ? std::cbrt(mean.volume) :
                                  mesh.dimension==2 ? std::sqrt(mean.volume) : mean.volume;
                filter.min_edge=std::numeric_limits<double>::max();
                for (unsigned i=0;i<e.count;++i) {
                    filter.position[i]=e.position[i];
                    filter.gradient[i]=mean.basis[i];
                    for (unsigned j=0;j<i;++j) if (e.edge(i,j)) {
                        const double length=coordinate_length(e.position[i]-e.position[j]);
                        filter.min_edge=std::min(filter.min_edge,length);
                        filter.max_edge=std::max(filter.max_edge,length);
                    }
                }
                for (unsigned j=0;j<mesh.dimension;++j) {
                    double sum=0.;
                    for (unsigned i=0;i<e.count;++i) sum+=std::abs(mean.basis[i][j]);
                    filter.gradient_sum=std::max(filter.gradient_sum,sum);
                }
                filter.prepare_geometry(e.count,op_.les.scale==turbulence::Scale::OmegaTilde ||
                                                op_.les.scale==turbulence::Scale::Sla);
                if (filter.rectangular_box) {
                    // Для формулы среднего градиента важны и координаты, и
                    // связность: три ребра каждого угла идут вдоль трёх осей.
                    for (unsigned k=0;k<8;++k) for (unsigned axis=0;axis<3;++axis) {
                        if (!e.edge(filter.corner_node[k],filter.corner_node[k^(1u<<axis)]) ||
                            !std::isfinite(filter.face_gradient[axis]))
                            filter.rectangular_box=false;
                    }
                }
                filters_.push_back(filter);
            }
        }
#endif
        cell.first_vertex = op_.vertex_count;
        op_.vertex_count += cell.count;
        for (unsigned i = 0; i < e.count; ++i) {
            cell.node[i] = locate(e.nodes[i],e.owners[i]);
            if (e.owners[i] == world.rank) {
                by_node.at(cell.node[i]).push_back(8*cells_.size()+i);
                op_.neighbour_count += e.count-1;
            }
        }
        cells_.push_back(cell);
    };

    if (parallel_cells) {
        prepare_local_cells(part,seam,frame,locate);
        index_local_vertices();
    } else if (world.size == 1) {
        for (MeshIndex i = 0; i < mesh.cell_count; ++i) add(describe(part,seam,frame,i));
    } else {
        Processes::Messages send;
        world.local([&] {
            send.resize(world.size);
            for (MeshIndex i = 0; i < mesh.cell_count; ++i) {
                // Единственная исходная вершина назначает отправителя. Это исключает повторное
                // включение приграничного элемента после обмена периодическими представителями.
                if (part.node_owners[mesh.cell_nodes(i)[0]] != world.rank) continue;
                const auto e = describe(part,seam,frame,i);
                const std::set<int> destinations(e.owners,e.owners+e.count);
                for (int p : destinations) send[p].append(reinterpret_cast<const char*>(&e),sizeof(e));
            }
        });
        const auto received = world.exchange(send);
        world.local([&] {
            // Глобальный порядок элементов даёт одинаковый порядок сложения в строке на одном и нескольких процессах.
            std::map<MeshIndex,Element> sorted;
            for (const auto& message : received) {
                if (message.size()%sizeof(Element)) throw MeshError("AES: incomplete MPI element");
                for (size_t k = 0; k < message.size(); k += sizeof(Element)) {
                    Element e;
                    std::memcpy(&e,message.data()+k,sizeof(e));
                    if (!sorted.emplace(e.id,e).second) throw MeshError("AES: duplicate MPI element");
                }
            }
            for (const auto& item : sorted) add(item.second);
        });
    }
    world.local([&] {
        if (!parallel_cells) offsets_.push_back(0);
        for (const auto& row : by_node) {
            incidence_.insert(incidence_.end(),row.begin(),row.end());
            offsets_.push_back(incidence_.size());
#ifdef AERO_HAS_TURBULENCE
            if (op_.computes_les()) {
                double sum=0., squared_length=0.;
                for (MeshIndex entry:row) {
                    const auto e=entry/8;
                    const auto& filter=filters_[filter_indices_[e]];
                    const double weight=filter.volume*cells_[e].inverse_count;
                    sum+=weight;
                    if (op_.fixed_ddes_length)
                        squared_length+=weight*filter.length_squared(op_.les.scale,cells_[e].count,Coordinate{});
                }
                les_volumes_.push_back(sum);
                if (op_.c_des>0.) {
                    const double length=op_.fixed_ddes_length && sum>0. ? op_.c_des*std::sqrt(squared_length/sum) : 0.;
                    ddes_nodes_.push_back({length,0.});
                }
            }
#endif
        }
#ifdef AERO_HAS_TURBULENCE
        // Если вся локальная сетка состоит из прямоугольных гексаэдров,
        // выписываем восемь номеров в одинаковом геометрическом порядке.
        // На GPU одновременные обращения соседних потоков лежат подряд.
        if (op_.computes_les() && op_.dim==3 && !cells_.empty() &&
            std::all_of(filters_.begin(),filters_.end(),[](const turbulence::FilterCell& f){return f.rectangular_box;})) {
            bool narrow=true;
            for (const auto& cell:cells_)
                for (auto n:cell.node) if (n>UINT32_MAX) narrow=false;
            if (narrow) {
                op_.box_stride=gpu_filter_ ? cells_.size() : 1;
                op_.box_pitch=gpu_filter_ ? 1 : 8;
                box_nodes_.resize(cells_.size()*8);
                for (size_t e=0;e<cells_.size();++e) for (unsigned k=0;k<8;++k) {
                    const auto i=filters_[filter_indices_[e]].corner_node[k];
                    box_nodes_[op_.box_pitch*e+op_.box_stride*k]=static_cast<std::uint32_t>(cells_[e].node[i]);
                }
            }
        }
#endif
#ifdef AERO_HAS_TURBULENCE
        prepare_hex_filter();
        prepare_material_vertices();
#endif
        // Если геометрия повторяется, одна малая таблица коэффициентов хорошо
        // помещается в кэш. На деформированных гексаэдрах у каждой формы своя
        // таблица. Читаем её один раз группой соседних исполнителей элемента.
        // Буфер из восьми производных на элемент ограничен объёмом таблицы пар.
        const size_t pairs_per_buffer=(8*sizeof(Gradient)+sizeof(PairMetric)-1)/sizeof(PairMetric);
        op_.element_order = op_.dim == 3 && !cells_.empty() &&
            metrics_.size()/cells_.size() >= pairs_per_buffer &&
            std::all_of(cells_.begin(),cells_.end(),[](const Cell& cell) { return cell.count==8; });
        op_.cell_count = cells_.size();
        op_.cells = cells_.data();
        op_.stencil.pairs = metrics_.data();
        op_.offsets = offsets_.data();
        op_.incidence = incidence_.data();
        wall.assign(rows,0);
    });
}
#ifdef AERO_HAS_TURBULENCE
void Aes::prepare_material_vertices() {
    if (cells_.empty() || (!material_from_sa_ && !op_.computes_les())) return;
    // На CPU выбираем границы цикла один раз для всей однородной сетки.
    // Номера вершин остаются в Cell, дополнительных массивов этот путь не требует.
    if (!gpu_filter_) {
        const auto count=cells_.front().count;
        if (std::all_of(cells_.begin(),cells_.end(),[&](const Cell& cell){return cell.count==count;}))
            host_vertices_=count;
        return;
    }
    if (!cache_material_) return;
    const auto count=cells_.front().count;
    // Тот же способ усреднения работает на треугольниках и призмах. На
    // смешанной сетке количество вершин по-прежнему читается из каждого элемента.
    if (count != 3 && count != 4 && count != 6 && count != 8) return;
    for (const auto& cell:cells_) {
        if (cell.count!=count) return;
        for (unsigned i=0; i<count; ++i) if (cell.node[i]>UINT32_MAX) return;
    }
    const unsigned lanes = Operator::material_lanes;
    op_.vertices_per_cell=count;
    if (!hex_nodes_.empty()) return; // Исходный порядок вершин уже записан фильтром в такой же раскладке.
    const auto blocks=(cells_.size()-1)/lanes+1;
    if (blocks>material_vertices_.max_size()/(lanes*count)) {
        op_.vertices_per_cell=0;
        return;
    }
    material_vertices_.resize(blocks*lanes*count);
    parallel_for(cells_.size(),[&](MeshIndex e) {
        auto* first=material_vertices_.data()+(e/lanes)*count*lanes+e%lanes;
        for (unsigned i=0; i<count; ++i) first[i*lanes]=static_cast<std::uint32_t>(cells_[e].node[i]);
    });
}
void Aes::prepare_hex_filter() {
    using Table=turbulence::HexFilterData;
    // Для повторяющихся форм таблица исходной геометрии уже помещается в кэш.
    // Дополнительную раскладку строим только для неоднородной сетки гексаэдров.
    const auto count=cells_.size();
    if (!op_.computes_les() || op_.dim!=3 || !count || !box_nodes_.empty() ||
        filters_.size()<count/8 || filters_.empty()) return;
    for (const auto& cell:cells_) {
        if (cell.count!=8) return;
        for (auto node:cell.node) if (node>UINT32_MAX) return;
    }
    const unsigned lanes=gpu_filter_ ? 32u : 1u;
    const auto blocks=(count-1)/lanes+1;
    if (blocks>hex_coefficients_.max_size()/(lanes*Table::Coefficients)) return;
    const auto padded=blocks*lanes;
    hex_coefficients_.resize(padded*Table::Coefficients);
    hex_nodes_.resize(padded*8);
    op_.hex_filter={hex_nodes_.data(),hex_coefficients_.data(),lanes};
    parallel_for(count,[&](MeshIndex e) {
        const auto& geometry=filters_[filter_indices_[e]];
        auto* col=hex_coefficients_.data()+op_.hex_filter.first(e,Table::Coefficients);
        auto* nodes=hex_nodes_.data()+op_.hex_filter.first(e,8);
        col[Table::Volume*lanes]=geometry.volume;
        col[Table::VolumeLength*lanes]=geometry.vol_length;
        col[Table::GradientSum*lanes]=geometry.gradient_sum;
        col[Table::OtherDiameter*lanes]=geometry.bounded_diameter ? geometry.remaining_length2 : -1.;
        for (unsigned k=0; k<8; ++k) {
            nodes[k*lanes]=static_cast<std::uint32_t>(cells_[e].node[k]);
            if (k) for (unsigned j=0; j<3; ++j)
                col[(Table::Basis+3*(k-1)+j)*lanes]=geometry.gradient[k][j];
        }
        if (geometry.bounded_diameter) for (unsigned k=0; k<4; ++k) {
            const auto diagonal=geometry.long_diagonal(k);
            for (unsigned j=0; j<3; ++j) col[(Table::Diagonals+3*k+j)*lanes]=diagonal[j];
        }
    });
}
#endif

void Aes::allocate_host(MeshIndex nodes) {
    if (op_.element_order) {
        gradient_cache_.resize(op_.vertex_count);
        op_.gradient_cache=gradient_cache_.data();
    }
#ifdef AERO_HAS_TURBULENCE
    if (op_.computes_les()) {
        samples_.resize(cells_.size());
        if (op_.c_des>0.) {
            cell_gradient_norm_.resize(cells_.size());
            op_.sa.ddes=ddes_nodes_.data();
            op_.cell_gradient_norm=cell_gradient_norm_.data();
        } else {
            eddy_nu_.resize(nodes);
            op_.eddy_nu=eddy_nu_.data();
        }
        op_.no_slip=wall.data();
        op_.filter_indices=filter_indices_.data(); op_.les_volumes=les_volumes_.data();
        op_.box_nodes=box_nodes_.empty() ? nullptr : box_nodes_.data();
        op_.filters=filters_.data(); op_.samples=samples_.data();
    }
#endif
    gas_.resize(nodes);
    material_.resize(cells_.size());
    rows_.resize(wall.size());
    op_.node_count = nodes;
    op_.gas = gas_.data();
    op_.material = material_.data();
    op_.rows = rows_.data();
}
#ifdef AERO_HAS_TURBULENCE
// Тип модели и размерность одинаковы для всех элементов. Передаём их
// компилятору до обхода, чтобы он развернул вычисление производных скорости.
template<bool ddes,unsigned dimension>
void host_subgrid(const Operator& op) {
    if (dimension==3 && op.hex_filter.nodes)
        parallel_for(op.cell_count,[&](MeshIndex e) { op.sample_hex<ddes>(e); });
    else if (dimension==3 && op.box_nodes)
        parallel_for(op.cell_count,[&](MeshIndex e) { op.sample_les<ddes,3,true>(e); });
    else
        parallel_for(op.cell_count,[&](MeshIndex e) { op.sample_les<ddes,dimension>(e); });
    if (!ddes) {
        parallel_for(op.row_count,[&](MeshIndex n) { op.gather_filter<false>(n); });
    }
}
// У массовых типов элементов число компонент и вершин известно до входа
// в параллельный цикл. Общая формула усреднения остаётся в Operator.
template<unsigned dimension,unsigned vertices>
void average_host_material(const Operator& op) {
    if (op.sa.active)
        parallel_for(op.cell_count,[&](MeshIndex e){op.average_material<dimension,vertices,Operator::Mixing::Sa>(e);});
    else
        parallel_for(op.cell_count,[&](MeshIndex e){op.average_material<dimension,vertices,Operator::Mixing::Sigma>(e);});
}
#endif
void Aes::evaluate(const double* state, const EulerState* cached, bool gas_is_current, Evaluation phase) {
    const EvaluationPlan work(phase);
    if (!gas_is_current && op_.needs_gas(phase))
        parallel_for(op_.node_count,[&](MeshIndex node) { op_.prepare_gas(node,state,cached); });
#ifdef AERO_HAS_TURBULENCE
    if (work.filter && !op_.skip_ddes_filter) {
        if (op_.c_des==0.) host_subgrid<false,3>(op_);
        else if (op_.dim==3) host_subgrid<true,3>(op_);
        else host_subgrid<true,0>(op_);
    }
    if (work.sa_material)
        parallel_for(op_.node_count,[&](MeshIndex n) { op_.prepare_sa_material(n); });
#endif
    if (work.cell_material) {
#ifdef AERO_HAS_TURBULENCE
        if (host_vertices_==8 && op_.dim==3) average_host_material<3,8>(op_);
        else if (host_vertices_==4 && op_.dim==2) average_host_material<2,4>(op_);
        else
#endif
            parallel_for(op_.cell_count,[&](MeshIndex e){op_.average_material(e);});
    }
    if (!work.rows) return;
    using Mode=Operator::Gradients;
    if (op_.element_order) {
        if (!work.reuse_gradients)
            parallel_for(op_.vertex_count,[&](MeshIndex k) { op_.prepare_hex_gradient(k); });
        parallel_for(op_.row_count,[&](MeshIndex n) { op_.assemble_row<3,Mode::Element>(n); });
    } else if (!op_.gradient_cache)
        parallel_for(op_.row_count,[&](MeshIndex node) { op_.assemble(node); });
    else if (work.reuse_gradients)
        parallel_for(op_.row_count,[&](MeshIndex node) { op_.assemble<Mode::Reuse>(node); });
    else
        parallel_for(op_.row_count,[&](MeshIndex node) { op_.assemble<Mode::Save>(node); });
}
}
}
