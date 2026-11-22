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
    using Index = ReconstructionIndex;
    struct Counts { Index rows, terms; };
    std::vector<TwoNodeSlope> pairs_;
    std::vector<Index> offsets_, nodes_;
    std::vector<double> weights_;
    std::vector<double> edge_epsilon_; // Одна длина для обеих сторон ребра, если значения точно совпадают.
    bool owners_match_ = false;
    bool shared_ = false;

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
    TwoNodeSlope common_row(Index a, Index b) const {
        const auto& left=pairs_[a];
        return left.first==TwoNodeSlope::marker && !left.second ? pairs_[b] : left;
    }
    void share_edge_rows(const ReconstructionStencilView& s, Index blocks) {
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
        if (std::find(compatible.begin(),compatible.end(),0)!=compatible.end()) return;
        std::vector<TwoNodeSlope> merged(5*s.edge_count);
        parallel_for(s.edge_count,[&](Index e) {
            const auto r=8*e;
            auto* dst=merged.data()+5*e;
            dst[0]=common_row(r,r+4);
            dst[1]=common_row(r+1,r+6);
            dst[2]=common_row(r+2,r+5);
            dst[3]=pairs_[r+3];
            dst[4]=pairs_[r+7];
        });
        pairs_.swap(merged);
        shared_=true;
    }
public:
    // При MPI и периодичности нумерация сторон может отличаться от нумерации
    // физических концов. Убираем копию номеров только после сравнения всех рёбер.
    void inspect_edges(const ReconstructionStencilView& s, const residual_detail::Surface* edges) {
        if (!s.edge_count || !s.row_owner) return;
        const auto groups = std::min<Index>(128,s.edge_count);
        std::vector<unsigned char> owners(groups,1), same_epsilon(groups,1);
        parallel_for(groups,[&](Index group) {
            for (Index e=s.edge_count*group/groups;e<s.edge_count*(group+1)/groups;++e) {
                if (edges[e].begin!=s.row_owner[2*e] || edges[e].end!=s.row_owner[2*e+1])
                    owners[group]=0;
                if (s.weno_order && s.weno_epsilon_scale[2*e]!=s.weno_epsilon_scale[2*e+1])
                    same_epsilon[group]=0;
            }
        },2);
        owners_match_=std::find(owners.begin(),owners.end(),0)==owners.end();
        if (s.weno_order && std::find(same_epsilon.begin(),same_epsilon.end(),0)==same_epsilon.end()) {
            edge_epsilon_.resize(s.edge_count);
            parallel_for(s.edge_count,[&](Index e) {edge_epsilon_[e]=s.weno_epsilon_scale[2*e];});
        }
    }
    void use_edge_data(ReconstructionStencilView& s, const residual_detail::Surface* device_edges) const {
        if (owners_match_) {
            s.row_owner=nullptr;
            s.owner_edges=device_edges;
        }
        if (!edge_epsilon_.empty()) {
            s.epsilon_per_edge=true;
            s.weno_epsilon_scale=edge_epsilon_.data();
        }
    }
    void prepare(const ReconstructionStencilView& s,bool share=false) {
        if (!s.weno_order || !s.edge_count || s.node_count > TwoNodeSlope::marker) return;
        const Index rows = 8*s.edge_count;
        const auto blocks = std::min<Index>(128,rows);
        std::vector<Counts> starts(blocks+1, Counts{});
        // Каждый поток сначала измеряет свой диапазон. Префиксная сумма даёт
        // непересекающиеся места для последующего заполнения исключений.
        parallel_for(blocks,[&](Index b) {
            auto& total = starts[b+1];
            for (Index r=rows*b/blocks; r<rows*(b+1)/blocks; ++r) {
                const auto n = s.slope_offsets[r+1]-s.slope_offsets[r];
                if (n && n!=2) { ++total.rows; total.terms+=n; }
            }
        },2);
        for (Index b=0;b<blocks;++b) {
            starts[b+1].rows+=starts[b].rows;
            starts[b+1].terms+=starts[b].terms;
        }
        const auto extra=starts.back();
        const Index original_terms=s.slope_offsets[rows];
        // Обычный CSR выгоднее, если длинные строки преобладают. Сравниваем
        // размеры до выделения памяти, включая смещения таблицы исключений.
        const auto term_bytes=sizeof(Index)+sizeof(double);
        if (extra.rows>TwoNodeSlope::marker ||
            rows*sizeof(TwoNodeSlope)+(extra.rows+1)*sizeof(Index)+extra.terms*term_bytes >=
            (rows+1)*sizeof(Index)+original_terms*term_bytes) return;
        pairs_.resize(rows);
        offsets_.resize(extra.rows+1);
        nodes_.resize(extra.terms);
        weights_.resize(extra.terms);
        parallel_for(blocks,[&](Index b) {
            auto cursor=starts[b];
            for (Index r=rows*b/blocks; r<rows*(b+1)/blocks; ++r) {
                const auto first=s.slope_offsets[r], n=s.slope_offsets[r+1]-first;
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
        if (share) share_edge_rows(s,blocks);
    }
    // Упаковка меняет адреса только в копии описания, предназначенной для GPU.
    Index bind(ReconstructionStencilView& s) const {
        if (pairs_.empty()) return 0;
        s.pairs=pairs_.data();
        s.shared_edge_slopes=shared_;
        s.slope_offsets=offsets_.data();
        s.slope_nodes=nodes_.data();
        s.slope_weights=weights_.data();
        return offsets_.size()-1;
    }
};
}
