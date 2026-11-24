#pragma once
#include "reconstruction.h"
#include "../parallel/parallel_for.h"
#include <limits>
#include <vector>
#include <unordered_map>

namespace aero {
namespace detail {
// Копируем одну таблицу разреженных строк. Число номеров и весов задаётся
// последним смещением. При расчёте размера памяти allocate только считает байты.
template<class Storage>
void upload_stencil_rows(Storage& storage, ReconstructionIndex rows, const ReconstructionIndex*& offsets,
                         const ReconstructionIndex*& nodes, const double*& weights, const unsigned*& indices32,
                         ReconstructionIndex node_count) {
    const auto entries = offsets ? offsets[rows] : 0;
    indices32=nullptr;
    // Выбор зависит только от размера сетки. Веса остаются double, порядок строк
    // и слагаемых не меняется; большие индексы автоматически сохраняют 64 бита.
    if (entries && entries<=std::numeric_limits<unsigned>::max() &&
        node_count<=std::numeric_limits<unsigned>::max()) {
        std::vector<unsigned> compact;
        if (rows>=compact.max_size() || entries>compact.max_size()-rows-1)
            throw std::length_error("Compact reconstruction table too large");
        compact.resize(static_cast<std::size_t>(rows+1+entries));
        parallel_for(rows+1,[&](ReconstructionIndex i) { compact[i]=static_cast<unsigned>(offsets[i]); });
        parallel_for(entries,[&](ReconstructionIndex i) { compact[rows+1+i]=static_cast<unsigned>(nodes[i]); });
        indices32=storage.allocate(compact.size(),compact.data());
        offsets=nullptr; nodes=nullptr;
        weights=storage.allocate(entries,weights);
        return;
    }
    // Если все строки пусты, вместо массива одинаковых нулей достаточно nullptr.
    offsets = storage.allocate(entries ? rows + 1 : 0,offsets);
    nodes = storage.allocate(entries,nodes);
    weights = storage.allocate(entries,weights);
}
}

// Пакуем соседние рёбра по 32, сохраняя порядок опор каждой исходной строки.
// Словарь хранит исходные double. При превышении бюджета остаётся обычный CSR.
template<class Storage>
bool upload_ebr_tiles(Storage& storage, ReconstructionStencilView& view) {
    using Index=ReconstructionIndex;
    if (!view.offsets || view.weno_order || !view.edge_count ||
        view.node_count>std::numeric_limits<unsigned>::max()) return false;
    const Index edges=view.edge_count,tiles=(edges+31)/32,entries=view.offsets[2*edges];
    if (!entries || entries>std::numeric_limits<unsigned>::max()) return false;
    const Index budget=(2*edges+1)*sizeof(unsigned)+entries*(sizeof(unsigned)+sizeof(double));
    std::vector<unsigned> header(3*tiles+2*edges);
    std::vector<EbrTerm> terms;
    std::vector<double> palette;
    for (Index tile=0;tile<tiles;++tile) {
        if (terms.size()>std::numeric_limits<unsigned>::max() || palette.size()>std::numeric_limits<unsigned>::max()) return false;
        const auto begin=terms.size(),base=palette.size();
        header[3*tile]=static_cast<unsigned>(begin);
        header[3*tile+1]=static_cast<unsigned>(base);
        std::unordered_map<double,unsigned short> dictionary;
        dictionary.emplace(0.,0);
        palette.push_back(0.);
        const auto encode=[&](double value) {
            const auto entry=dictionary.emplace(value,static_cast<unsigned short>(dictionary.size()));
            if (entry.second) palette.push_back(value);
            return entry.first->second;
        };
        for (unsigned lane=0;lane<32 && 32*tile+lane<edges;++lane) {
            const Index row=2*(32*tile+lane),mid=view.offsets[row+1],end=view.offsets[row+2];
            Index l=view.offsets[row],r=mid;
            unsigned n=0, masks[2]={};
            while (l<mid || r<end) {
                if (n==32) return false;
                const Index sentinel=~Index(0);
                const Index left=l<mid ? view.nodes[l] : sentinel,right=r<end ? view.nodes[r] : sentinel;
                const Index node=std::min(left,right);
                EbrTerm term={static_cast<unsigned>(node),0,0};
                if (left==node) { term.left=encode(view.weights[l++]); masks[0]|=1u<<n; }
                if (right==node) { term.right=encode(view.weights[r++]); masks[1]|=1u<<n; }
                if (begin+32*(n+1)>terms.size()) terms.resize(begin+32*(n+1));
                terms[begin+32*n+lane]=term;
                ++n;
            }
            header[3*tiles+row]=masks[0];
            header[3*tiles+row+1]=masks[1];
        }
        header[3*tile+2]=static_cast<unsigned>((terms.size()-begin)/32);
        if (header.size()*sizeof(unsigned)+terms.size()*sizeof(EbrTerm)+palette.size()*sizeof(double)>budget) return false;
    }
    view.linear_indices32=storage.allocate(header.size(),header.data());
    view.nodes=reinterpret_cast<const Index*>(storage.allocate(terms.size(),terms.data()));
    view.weights=storage.allocate(palette.size(),palette.data());
    view.offsets=nullptr;
    return true;
}

// Принимает копию описания на CPU и заменяет адреса адресами GPU. Владелец
// памяти определяет способ выделения: отдельные массивы или одна область подобласти.
template<class Storage>
ReconstructionStencilView upload_reconstruction_stencil(Storage& storage, ReconstructionStencilView view,
                                                         ReconstructionIndex exception_rows = 0, bool pack_ebr=false) {
    const auto rows = 2 * view.edge_count;
    if (!pack_ebr || !upload_ebr_tiles(storage,view))
        detail::upload_stencil_rows(storage,rows,view.offsets,view.nodes,view.weights,view.linear_indices32,view.node_count);
    view.row_owner = storage.allocate(view.row_owner ? rows : 0,view.row_owner);
    view.linear_edge_mask=storage.allocate(view.linear_edge_mask ? (rows+31)/32 : 0,view.linear_edge_mask);

    // У WENO и MP на каждую сторону отведено четыре строки наклонов.
    // Линейная схема использует только первую таблицу и обходится без второй.
    const auto nonlinear_rows = view.weno_order ? rows : 0;
    if (!nonlinear_rows) view.slope_offsets = nullptr;
    const auto rows_to_copy = view.pairs ? exception_rows : 4 * nonlinear_rows;
    view.slope_rows=rows_to_copy;
    view.pairs = storage.allocate(view.pair_count(),view.pairs);
    detail::upload_stencil_rows(storage,rows_to_copy,view.slope_offsets,view.slope_nodes,view.slope_weights,
                               view.slope_indices32,view.node_count);
    view.weno_order = storage.allocate(nonlinear_rows,view.weno_order);
    view.weno_epsilon_scale = storage.allocate(view.epsilon_per_edge ? nonlinear_rows/2 : nonlinear_rows,
                                               view.weno_epsilon_scale);
    return view;
}
}
