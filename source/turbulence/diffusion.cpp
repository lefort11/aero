#include "storage.h"
#include "diffusion_geometry.h"
#include "../parallel/small_weight_row.h"
#include "../viscosity/transport.h"
#include <algorithm>
#include <map>
#include <tuple>

namespace aero {
namespace turbulence {
// Объединяет совпадающие геометрические строки диффузии для сокращения памяти.
bool Storage::share_diffusion_rows() {
    // Лексикографически упорядочивает строки диффузионных коэффициентов.
    struct Compare {
        bool operator()(const std::vector<DiffusionTerm> &a, const std::vector<DiffusionTerm> &b) const {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                                [](const DiffusionTerm &x, const DiffusionTerm &y) {
                                                    return std::tie(x.element, x.neighbour, x.geometry) <
                                                           std::tie(y.element, y.neighbour, y.geometry);
                                                });
        }
    };
    std::map<std::vector<DiffusionTerm>, RowRange, Compare> dictionary;
    std::vector<RowRange> rows;
    std::vector<DiffusionTerm> shared, key;
    std::vector<MeshIndex> anchors;
    const auto limit = diffusion_terms.size() / 2;
    for (MeshIndex i = 0; i < diffusion_rows.size(); ++i) {
        const auto row = diffusion_rows[i];
        const auto anchor = row.first < row.end ? diffusion_terms[row.first].element : 0;
        key.clear();
        for (auto p = row.first; p < row.end; ++p) {
            const auto &term = diffusion_terms[p];
            // Индексы беззнаковые: обратное сложение восстанавливает и номера меньше базового.
            key.push_back({term.element - anchor, term.neighbour - i, term.geometry});
        }
        auto it = dictionary.lower_bound(key);
        if (it == dictionary.end() || Compare{}(key, it->first)) {
            if (shared.size() + key.size() > limit)
                return false;
            const RowRange span{shared.size(), shared.size() + key.size()};
            shared.insert(shared.end(), key.begin(), key.end());
            if (dictionary.size() >= 4096)
                dictionary.clear();
            it = dictionary.emplace(key, span).first;
        }
        rows.push_back(it->second);
        anchors.push_back(anchor);
    }
    diffusion_rows.swap(rows);
    diffusion_terms.swap(shared);
    diffusion_anchors.swap(anchors);
    return true;
}

// Упаковывает компактные строки с тремя соседями в специализированный формат.
bool Storage::pack_three_neighbours(size_t node_count) {
    if (node_count>UINT32_MAX || element_count>UINT32_MAX || diffusion_terms.empty()
        || diffusion_terms.size()%3) return false;
    for (const auto& row:diffusion_rows)
        if (row.first%3 || row.end%3) return false;
    for (size_t p=0;p<diffusion_terms.size();p+=3)
        if (diffusion_terms[p].element!=diffusion_terms[p+1].element ||
            diffusion_terms[p].element!=diffusion_terms[p+2].element) return false;
    triples.reserve(diffusion_terms.size()/3);
    for (size_t p=0;p<diffusion_terms.size();p+=3) {
        DiffusionTriple group{};
        group.element=static_cast<std::uint32_t>(diffusion_terms[p].element);
        for (unsigned k=0;k<3;++k) {
            const auto& term=diffusion_terms[p+k];
            group.neighbour[k]=static_cast<std::uint32_t>(term.neighbour);
            group.geometry[k]=term.geometry;
            group.diagonal+=std::abs(term.geometry);
        }
        triples.push_back(group);
    }
    for (auto& row:diffusion_rows) { row.first/=3; row.end/=3; }
    std::vector<DiffusionTerm>().swap(diffusion_terms);
    return true;
}

// Упаковывает плотные диффузионные строки для регулярного доступа.
bool Storage::pack_dense_rows(MeshIndex possible) {
    if (!possible || diffusion_terms.size()<=possible-possible/3 ||
        distance.size()>UINT32_MAX || element_count>UINT32_MAX) return false;
    // Сначала считаем число элементов в каждой строке. Периодические копии
    // могут добавить восьмого соседа, тогда сохраняем общий список пар.
    std::vector<RowRange> rows(diffusion_rows.size());
    std::vector<unsigned char> too_wide(rows.size());
    parallel_for(rows.size(),[&](size_t i) {
        const auto row=diffusion_rows[i];
        size_t elements=0, neighbours=0;
        for (auto p=row.first;p<row.end;++p) {
            if (p==row.first || diffusion_terms[p].element!=diffusion_terms[p-1].element) {
                ++elements;
                neighbours=0;
            }
            if (++neighbours>DiffusionCell::neighbours) too_wide[i]=1;
        }
        rows[i].end=elements;
    });
    if (std::find(too_wide.begin(),too_wide.end(),1)!=too_wide.end()) return false;
    size_t count=0;
    for (auto& row:rows) {
        row.first=count;
        count+=row.end;
        row.end=count;
    }
    if (count>diffusion_terms.size()*sizeof(DiffusionTerm)/sizeof(DiffusionCell)) return false;
    cell_rows.resize(count);
    parallel_for(rows.size(),[&](size_t i) {
        const auto row=diffusion_rows[i];
        auto p=row.first;
        for (auto k=rows[i].first;k<rows[i].end;++k) {
            const auto element=diffusion_terms[p].element;
            auto& block=cell_rows[k];
            block.element=static_cast<std::uint32_t>(element);
            unsigned j=0;
            do {
                const auto& term=diffusion_terms[p++];
                block.node[j]=static_cast<std::uint32_t>(term.neighbour);
                block.geometry[j++]=term.geometry;
                block.diagonal+=std::abs(term.geometry);
            } while (p<row.end && diffusion_terms[p].element==element);
            // Нулевой вес и собственный узел дополняют неполную строку.
            for (;j<DiffusionCell::neighbours;++j) block.node[j]=static_cast<std::uint32_t>(i);
        }
    });
    diffusion_rows.swap(rows);
    std::vector<DiffusionTerm>().swap(diffusion_terms);
    return true;
}

// Матрица скалярной диффузии симметрична. На косой сетке почти все
// пары ненулевые; сохраняем верхний треугольник и суммы модулей по вершинам.
bool Storage::prepare_element_pairs(const viscosity::Operator& aes,double scale) {
    if (!aes.neighbour_count || diffusion_terms.size()<=aes.neighbour_count-aes.neighbour_count/3)
        return false;
    for (MeshIndex e=0;e<aes.cell_count;++e) {
        const auto& c=aes.cells[e];
        for (unsigned i=0;i<c.count;++i) for (unsigned j=0;j<i;++j)
            if (c.node[i]==c.node[j]) return false;
    }
    std::map<MeshIndex,MeshIndex> seen;
    shape_index.reserve(aes.cell_count);
    for (MeshIndex e=0;e<aes.cell_count;++e) {
        const auto& c=aes.cells[e];
        const auto entry=seen.emplace(c.first_pair,pair_weights.size());
        shape_index.push_back(entry.first->second);
        if (!entry.second) continue;
        double diagonal[8]{};
        for (unsigned i=0;i<c.count;++i) for (unsigned j=i+1;j<c.count;++j) {
            const double g=diffusion_weight(aes.stencil,c,i,j,scale);
            pair_weights.push_back(g);
            diagonal[i]+=std::abs(g); diagonal[j]+=std::abs(g);
        }
        pair_weights.insert(pair_weights.end(),diagonal,diagonal+c.count);
    }
    // Номер 8*e+i уже записан в узловой таблице вхождений AES.
    // Та же адресация избавляет сборку от чтения геометрии соседнего элемента.
    vertex_contributions.resize(8*aes.cell_count);
    std::vector<DiffusionTerm>().swap(diffusion_terms);
    std::vector<RowRange>().swap(diffusion_rows);
    return true;
}

// В CPU-строках и при выборе сильных связей порядок суммирования одинаков:
// сначала элементы контрольного объёма, затем вершины каждого элемента.
template<class Visitor>
static void visit_diffusion_row(const viscosity::Operator& aes, MeshIndex node, double scale, Visitor visit) {
    for (auto pos=aes.offsets[node]; pos<aes.offsets[node+1]; ++pos) {
        const auto entry=aes.incidence[pos], e=entry/8;
        const auto i=static_cast<unsigned>(entry%8);
        const auto& cell=aes.cells[e];
        for (unsigned j=0; j<cell.count; ++j) {
            const double weight=diffusion_weight(aes.stencil,cell,i,j,scale);
            if (weight!=0.) visit(e,cell.node[j],weight);
        }
    }
}

// Сосед входит в линию только при взаимном выборе. Так у вершины будет не
// более двух связей, а на границе MPI линия закончится на собственном узле.
void Storage::prepare_lines(const viscosity::Operator& aes,double scale,const std::vector<MeshIndex>& owned) {
    const auto missing=static_cast<MeshIndex>(-1);
    // Обратная топологическая ссылка с исходным адресом в таблице.
    struct Link { MeshIndex node; double weight; };
    const Link none{missing,0.};
    using Pair=std::array<Link,2>;
    std::vector<Pair> preferred(distance.size(),Pair{{none,none}});
    // Каждый исполнитель выбирает сильные связи только для своих строк.
    parallel_for(owned.size(), [&](size_t row_index) {
        const auto i = owned[row_index];
        if (distance[i]<0.) return;
        // На структурированном участке обычно до 26 соседей. Накопление по
        // элементам и порядок выбора при равных весах остаются прежними.
        thread_local SmallWeightRow<MeshIndex,32> geometry;
        geometry.clear();
        visit_diffusion_row(aes,i,scale,[&](MeshIndex,MeshIndex neighbour,double weight) {
            geometry[neighbour]+=weight;
        });
        Link strongest[3]={none,none,none};
        for (const auto& edge:geometry) {
            Link next{edge.first,edge.second};
            for (auto& current:strongest) if (next.weight>current.weight) std::swap(next,current);
        }
        // При сравнимой связи в третьем направлении точечные итерации остаются дешевле.
        if (strongest[0].weight<=4.*strongest[2].weight) return;
        preferred[i][0]=strongest[0];
        if (strongest[1].weight>.125*strongest[0].weight) preferred[i][1]=strongest[1];
    });
    std::vector<Pair> links(distance.size(),Pair{{none,none}});
    parallel_for(owned.size(), [&](size_t row_index) {
        const auto i = owned[row_index];
        for (unsigned k=0;k<2;++k) {
            const auto edge=preferred[i][k];
            if (edge.node!=missing && (preferred[edge.node][0].node==i || preferred[edge.node][1].node==i))
                links[i][k]=edge;
        }
    });
    // Сначала посещаем свободные концы. Оставшиеся замкнутые линии (периодичность)
    // начинаем с первого ещё не посещённого узла. Между отрезками связь остаётся в невязке.
    std::vector<bool> seen(distance.size());
    bool coupled=false;
    for (unsigned pass=0;pass<2;++pass) for (auto first:owned) {
        if (seen[first] || (!pass && links[first][0].node!=missing && links[first][1].node!=missing)) continue;
        MeshIndex i=first, previous=missing, start=line_vertices.size();
        while (i!=missing) {
            seen[i]=true;
            double left=0.;
            Link next=none;
            for (const auto& edge:links[i]) {
                if (edge.node==missing) continue;
                if (edge.node==previous) left=edge.weight;
                else if (!seen[edge.node]) next=edge;
            }
            const bool cut=line_vertices.size()+1-start==SaLines::length || next.node==missing;
            line_vertices.push_back({i,left,cut ? 0. : next.weight});
            if (cut) {
                coupled=coupled || line_vertices.size()-start>1;
                line_segments.push_back({start,line_vertices.size()});
                start=line_vertices.size();
            }
            previous=cut ? missing : i;
            i=next.node;
        }
    }
    if (!coupled) { line_segments.clear(); line_vertices.clear(); }
}
// Строит геометрические строки диффузии рабочей вязкости SA.
void Storage::prepare_diffusion(const viscosity::Operator& aes,View& view,bool host,bool implicit,
                                const std::vector<MeshIndex>* steady_nodes) {
    element_count = aes.cell_count;
    const double scale = diffusion_scale(aes.law.gamma,aes.law.prandtl);
    if (steady_nodes) {
        prepare_lines(aes,scale,*steady_nodes);
        if (host) view.lines={line_vertices.data(),line_segments.data(),line_segments.size()};
    }
    // На GPU каждый узел обходит готовую геометрию AES. При steady на карту
    // потребуются выбранные выше линии, временную таблицу всех пар не строим.
    if (!host) return;
    diffusion_rows.resize(aes.row_count);
    // Одинаковый обход сначала определяет длину строки, затем заполняет её диапазон.
    // Порядок слагаемых внутри узла совпадает с порядком элементов в таблице AES.
    const auto build_row = [&](MeshIndex node, DiffusionTerm* output) {
        size_t count = 0;
        visit_diffusion_row(aes,node,scale,[&](MeshIndex e,MeshIndex neighbour,double weight) {
            if (output) output[count]={e,neighbour,weight};
            ++count;
        });
        return count;
    };
    parallel_for(aes.row_count, [&](MeshIndex node) {
        diffusion_rows[node].end = build_row(node, nullptr);
    });
    size_t total = 0;
    for (auto& row : diffusion_rows) {
        row.first = total;
        total += row.end;
        row.end = total;
    }
    diffusion_terms.resize(total);
    parallel_for(aes.row_count, [&](MeshIndex node) {
        const auto row = diffusion_rows[node];
        if (row.first != row.end) build_row(node, diffusion_terms.data()+row.first);
    });
    // В BDF при каждой скалярной итерации повторно считаем среднюю вязкость.
    // На CPU тот же обход элемента вычисляет парные вклады в обе вершины.
    // У явной схемы такого повторного обхода нет, ей выгоднее собирать строку в узле.
    bool dense = implicit && prepare_element_pairs(aes,scale);
    if (!dense) dense = pack_dense_rows(aes.neighbour_count);
    if (!dense) {
        share_diffusion_rows();
        pack_three_neighbours(distance.size());
    }
    if (distance.size()<=UINT32_MAX && element_count<=UINT32_MAX && !diffusion_terms.empty()) {
        local_terms.resize(diffusion_terms.size());
        parallel_for(local_terms.size(), [&](size_t i) {
            const auto& term=diffusion_terms[i];
            local_terms[i]={static_cast<std::uint32_t>(term.element),
                                   static_cast<std::uint32_t>(term.neighbour),term.geometry};
        });
        std::vector<DiffusionTerm>().swap(diffusion_terms);
    }
    view.diffusion.short_blocks = aes.cell_count && std::all_of(aes.cells,aes.cells+aes.cell_count,
                                        [](const viscosity::Cell& c){ return c.count<=4; });
    element_mean.resize(element_count);
    molecular_mean.resize(element_count);
    view.diffusion = {
        diffusion_rows.data(), diffusion_anchors.empty() ? nullptr : diffusion_anchors.data(),
        diffusion_terms.data(), local_terms.empty() ? nullptr : local_terms.data(),
        pair_weights.data(), shape_index.data(), vertex_contributions.empty() ? nullptr : vertex_contributions.data(),
        cell_rows.empty() ? nullptr : cell_rows.data(), triples.empty() ? nullptr : triples.data(),
        molecular_mean.data(), element_mean.data(), view.diffusion.short_blocks};
}
}
}
