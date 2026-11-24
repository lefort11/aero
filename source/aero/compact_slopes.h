#pragma once
#include "../parallel/parallel_for.h"
#include "../reconstruction/reconstruction.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace aero {
// Во время подготовки собираем двухузловые наклоны и редкие исходные строки.
// Геометрические веса и порядок обращения к полю совпадают с исходным CSR.
class CompactSlopes {
    // Тип номеров узлов и смещений шаблона.
    using Index = ReconstructionIndex;
    // Число строк и ненулевых весов компактного шаблона.
    struct Counts { Index rows, terms; };
    // Пары узлов для двухточечных разностей.
    std::vector<TwoNodeSlope> pairs_;
    // Границы строк CSR и номера узлов внутри строк.
    std::vector<Index> offsets_, nodes_;
    // Коэффициенты при узлах соответствующих строк.
    std::vector<double> weights_;
    std::vector<double> edge_epsilon_; // Одна длина для обеих сторон ребра, если значения точно совпадают.
    // Битовая маска сторон с линейной реконструкцией.
    std::vector<unsigned> linear_mask_;
    // Владельцы всех строк совпадают с концами рёбер, поэтому отдельная таблица не нужна.
    bool owners_match_ = false;
    // Признак общего хранения совпадающих строк.
    bool shared_ = false;
    // Выбран компактный набор строк на ребро.
    bool short_rows_ = false;
    // Центральная разность берётся по концам ребра.
    bool central_ = false;
    // Количество хранимых строк шаблона одного ребра.
    unsigned rows_per_edge_ = 8;

    // Одна и та же геометрическая производная может понадобиться с двух сторон.
    // Сравниваем номера узлов и сами double вплоть до последнего бита. Если
    // номера периодических копий различаются, оставляем исходные отдельные записи.
    static bool same_row(const ReconstructionStencilView& s, Index a, Index b) {
        const auto first=s.slope_offsets[a], other=s.slope_offsets[b];
        const auto size=s.slope_offsets[a+1]-first, other_size=s.slope_offsets[b+1]-other;
        if (!size || !other_size) return true;
        return size==other_size &&
               std::memcmp(s.slope_nodes+first,s.slope_nodes+other,size*sizeof(Index))==0 &&
               std::memcmp(s.slope_weights+first,s.slope_weights+other,size*sizeof(double))==0;
    }
    // Пустая сторона не участвует в реконструкции. Берём запись с противоположной
    // стороны, если она есть. Порядок каждой стороны по-прежнему проверяется при чтении.
    static Index common_row(const ReconstructionStencilView& s, Index a, Index b) {
        return s.slope_offsets[a]==s.slope_offsets[a+1] ? b : a;
    }
    // Проверяет возможность общей нумерации до подсчёта памяти, включая редкие CSR-строки.
    static bool can_share_edge_rows(const ReconstructionStencilView& s, Index blocks) {
        std::vector<unsigned char> compatible(blocks,1);
        parallel_for(blocks,[&](Index b) {
            for (Index e=s.edge_count*b/blocks;e<s.edge_count*(b+1)/blocks;++e) {
                const auto r=8*e;
                if (!same_row(s,r,r+4) || !same_row(s,r+1,r+6) || !same_row(s,r+2,r+5)) {
                    compatible[b]=0;
                    return;
                }
            }
        },2);
        return std::find(compatible.begin(),compatible.end(),0)==compatible.end();
    }
    // Привязывает запись компактной таблицы к исходной строке, не меняя порядок её слагаемых.
    Index source_row(const ReconstructionStencilView& s, Index slot) const {
        const auto edge=slot/rows_per_edge_, slot_in_edge=slot%rows_per_edge_, r=8*edge;
        if (!shared_) {
            const auto side_rows=(short_rows_ ? 2u : 4u)-unsigned(central_);
            return r+4*(slot_in_edge/side_rows)+slot_in_edge%side_rows+unsigned(central_);
        }
        const auto k=slot_in_edge+unsigned(central_);
        if (k==0) return common_row(s,r,r+4);
        if (k==1) return common_row(s,r+1,r+6);
        if (k==2) return common_row(s,r+2,r+5);
        return r+(k==3 ? 3 : 7);
    }
public:
    // Проверяет точное представление центральной разности, включая исходный порядок опор.
    static bool central_row(StencilRow row,const residual_detail::Surface& edge) {
        return row.count==2 && row.node(0)==edge.begin && row.node(1)==edge.end && row.weight(1)==1.;
    }
    // При MPI и периодичности нумерация сторон может отличаться от нумерации
    // физических концов. Убираем копию номеров только после сравнения всех рёбер.
    void inspect_edges(const ReconstructionStencilView& s, residual_detail::SurfaceTable edges) {
        if (!s.edge_count || !s.row_owner) return;
        const auto groups = std::min<Index>(128,s.edge_count);
        std::vector<unsigned char> owners(groups,1), same_epsilon(groups,1), central(groups,1), linear(groups,1);
        parallel_for(groups,[&](Index group) {
            for (Index e=s.edge_count*group/groups;e<s.edge_count*(group+1)/groups;++e) {
                if (edges[e].begin!=s.row_owner[2*e] || edges[e].end!=s.row_owner[2*e+1])
                    owners[group]=0;
                if (s.weno_order && s.weno_epsilon_scale[2*e]!=s.weno_epsilon_scale[2*e+1])
                    same_epsilon[group]=0;
                for (unsigned side=0;side<2;++side) {
                    const auto row=2*e+side;
                    if (s.weno_order && s.weno_order[row] && !central_row(s.slope_row(4*row),edges[e])) central[group]=0;
                    const auto fallback=s.linear_row(row);
                    if (fallback.count && !central_row(fallback,edges[e])) linear[group]=0;
                }
            }
        },2);
        owners_match_=std::find(owners.begin(),owners.end(),0)==owners.end();
        central_=s.weno_order && owners_match_ && std::find(central.begin(),central.end(),0)==central.end();
        if (owners_match_ && s.offsets && s.offsets[2*s.edge_count] &&
            std::find(linear.begin(),linear.end(),0)==linear.end()) {
            linear_mask_.resize((2*s.edge_count+31)/32);
            // Каждый поток пишет целое слово: соседние стороны не соревнуются за один бит.
            parallel_for(linear_mask_.size(),[&](Index word) {
                unsigned mask=0;
                for (unsigned bit=0;bit<32 && 32*word+bit<2*s.edge_count;++bit) {
                    const auto row=32*word+bit;
                    if (s.offsets[row]!=s.offsets[row+1]) mask|=1u<<bit;
                }
                linear_mask_[word]=mask;
            });
        }
        if (s.weno_order && std::find(same_epsilon.begin(),same_epsilon.end(),0)==same_epsilon.end()) {
            edge_epsilon_.resize(s.edge_count);
            parallel_for(s.edge_count,[&](Index e) {edge_epsilon_[e]=s.weno_epsilon_scale[2*e];});
        }
    }
    // Подключает геометрию рёбер вместо отдельных массивов владельцев и длин.
    void use_edge_data(ReconstructionStencilView& s, residual_detail::SurfaceTable device_edges) const {
        if (owners_match_) {
            s.row_owner=nullptr;
            s.owner_edges=device_edges;
        }
        if (!edge_epsilon_.empty()) {
            s.epsilon_per_edge=true;
            s.weno_epsilon_scale=edge_epsilon_.data();
        }
        if (!linear_mask_.empty()) {
            s.linear_edge_mask=linear_mask_.data();
            s.offsets=nullptr; s.nodes=nullptr; s.weights=nullptr;
        }
    }
    // Упаковывает подходящие строки WENO в двухузловой формат и проверяет совместимость владельцев.
    void prepare(const ReconstructionStencilView& s,bool share=true) {
        if (!s.weno_order || !s.edge_count || s.node_count > TwoNodeSlope::marker) return;
        short_rows_=std::find(s.weno_order,s.weno_order+2*s.edge_count,5)==s.weno_order+2*s.edge_count;
        const auto blocks = std::min<Index>(128,s.edge_count);
        shared_=share && can_share_edge_rows(s,blocks);
        rows_per_edge_=shared_ ? (short_rows_ ? 3u : 5u)-unsigned(central_)
                              : (short_rows_ ? 4u : 8u)-2*unsigned(central_);
        const Index rows = rows_per_edge_*s.edge_count;
        std::vector<Counts> starts(blocks+1, Counts{});
        // Каждый поток сначала измеряет свой диапазон. Префиксная сумма даёт
        // непересекающиеся места для последующего заполнения исключений.
        parallel_for(blocks,[&](Index b) {
            auto& total = starts[b+1];
            for (Index r=rows*b/blocks; r<rows*(b+1)/blocks; ++r) {
                const auto source=source_row(s,r);
                const auto n = s.slope_offsets[source+1]-s.slope_offsets[source];
                if (n && n!=2) { ++total.rows; total.terms+=n; }
            }
        },2);
        for (Index b=0;b<blocks;++b) {
            starts[b+1].rows+=starts[b].rows;
            starts[b+1].terms+=starts[b].terms;
        }
        const auto extra=starts.back();
        const Index original_terms=s.slope_offsets[8*s.edge_count];
        // Обычный CSR выгоднее, если длинные строки преобладают. Сравниваем
        // размеры до выделения памяти, включая смещения таблицы исключений.
        const auto term_bytes=sizeof(Index)+sizeof(double);
        if (extra.rows>TwoNodeSlope::marker ||
            rows*sizeof(TwoNodeSlope)+(extra.rows+1)*sizeof(Index)+extra.terms*term_bytes >=
            (8*s.edge_count+1)*sizeof(Index)+original_terms*term_bytes) return;
        pairs_.resize(rows);
        offsets_.resize(extra.rows+1);
        nodes_.resize(extra.terms);
        weights_.resize(extra.terms);
        parallel_for(blocks,[&](Index b) {
            auto cursor=starts[b];
            for (Index r=rows*b/blocks; r<rows*(b+1)/blocks; ++r) {
                const auto source=source_row(s,r);
                const auto first=s.slope_offsets[source], n=s.slope_offsets[source+1]-first;
                if (n==2) {
                    pairs_[r]={static_cast<unsigned>(s.slope_nodes[first]),
                               static_cast<unsigned>(s.slope_nodes[first+1]),s.slope_weights[first+1]};
                } else if (!n) {
                    pairs_[r]={TwoNodeSlope::marker,0,0.};
                } else {
                    pairs_[r]={TwoNodeSlope::marker,static_cast<unsigned>(cursor.rows+1),0.};
                    offsets_[cursor.rows++]=cursor.terms;
                    std::copy_n(s.slope_nodes+first,n,nodes_.data()+cursor.terms);
                    std::copy_n(s.slope_weights+first,n,weights_.data()+cursor.terms);
                    cursor.terms+=n;
                }
            }
        },2);
        offsets_.back()=extra.terms;
    }
    // Упаковка меняет адреса только в копии описания, предназначенной для GPU.
    Index bind(ReconstructionStencilView& s) const {
        if (pairs_.empty()) return 0;
        s.pairs=pairs_.data();
        s.shared_edge_slopes=shared_;
        s.short_slope_rows=short_rows_;
        s.central_from_edges=central_;
        s.slope_offsets=offsets_.data();
        s.slope_nodes=nodes_.data();
        s.slope_weights=weights_.data();
        return offsets_.size()-1;
    }
};
}
