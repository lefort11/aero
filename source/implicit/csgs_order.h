#ifndef AERO_IMPLICIT_CSGS_ORDER_H
#define AERO_IMPLICIT_CSGS_ORDER_H
#include "csgs_sweep.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace aero {
namespace implicit {
// Порядок зависит только от связности. Строим его один раз для всех шагов Ньютона.
// На GPU offsets ограничивают цвета, на CPU — независимые подобласти потоков.
struct CsgsOrder {
    std::vector<MeshIndex> offsets;     // Начало каждой группы и конец последней.
    std::vector<MeshIndex> sequence;    // Номер компактной строки в порядке прямого прохода.
    std::vector<MeshIndex> row_of_node; // Собственная строка узла. Для чужих узлов равна числу строк.
    std::vector<MeshIndex> position;    // Место строки в прямом проходе.
    std::vector<MeshIndex> domain;      // Номер участка CPU. На GPU весь ранг образует один участок.
    // Строит порядок строк для последовательных или раскрашенных проходов CSGS.
    CsgsOrder(MeshIndex nodes, const std::vector<MeshIndex>& rows, const std::vector<MeshIndex>& columns,
              bool gpu, MeshIndex workers) {
        const MeshIndex count = rows.size() - 1;
        row_of_node.assign(nodes, count);
        for (MeshIndex i = 0; i < count; ++i)
            row_of_node[columns[rows[i]]] = i;
        position.resize(count);
        if (gpu)
            color(rows, columns);
        else
            partition(rows, columns, workers);
        for (MeshIndex i = 0; i < count; ++i)
            position[sequence[i]] = i;
    }

  private:
    // Жадная раскраска: запрещаем цвета ранее обработанных соседей. Номер
    // текущей строки служит меткой и избавляет от очистки массива для каждого узла.
    void color(const std::vector<MeshIndex>& rows, const std::vector<MeshIndex>& columns) {
        const auto n = position.size();
        std::vector<MeshIndex> marks, colors(n);
        for (MeshIndex row = 0; row < n; ++row) {
            for (MeshIndex p = rows[row] + 1; p < rows[row + 1]; ++p) {
                const auto other = row_of_node[columns[p]];
                if (other < row) marks[colors[other]] = row;
            }
            MeshIndex c = 0;
            while (c < marks.size() && marks[c] == row)
                ++c;
            if (c == marks.size()) marks.push_back(n);
            colors[row] = c;
        }
        offsets.assign(marks.size() + 1, 0);
        for (auto c : colors)
            ++offsets[c + 1];
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        auto next = offsets;
        sequence.resize(n);
        for (MeshIndex i = 0; i < n; ++i)
            sequence[next[colors[i]]++] = i;
    }

    // Обход Катхилла–Макки посещает соседей в порядке возрастания степени.
    // Делим полученный порядок на соседние по графу участки для потоков CPU.
    void partition(const std::vector<MeshIndex>& rows, const std::vector<MeshIndex>& columns,
                   MeshIndex workers) {
        const auto n = position.size();
        std::vector<MeshIndex> roots(n);
        std::iota(roots.begin(), roots.end(), MeshIndex(0));
        // Упорядочивает узлы по степени графа для построения обхода CSGS.
        const auto degree_order = [&](MeshIndex i, MeshIndex j) {
            const auto a = rows[i + 1] - rows[i], b = rows[j + 1] - rows[j];
            return a == b ? i < j : a < b;
        };
        std::sort(roots.begin(), roots.end(), degree_order);
        std::vector<bool> seen(n, false);
        for (auto root : roots) {
            if (seen[root]) continue;
            MeshIndex head = sequence.size();
            sequence.push_back(root);
            seen[root] = true;
            while (head < sequence.size()) {
                const auto row = sequence[head++], first = sequence.size();
                for (MeshIndex p = rows[row] + 1; p < rows[row + 1]; ++p) {
                    const auto j = row_of_node[columns[p]];
                    if (j == n || seen[j]) continue;
                    sequence.push_back(j);
                    seen[j] = true;
                }
                std::sort(sequence.begin() + static_cast<std::ptrdiff_t>(first), sequence.end(), degree_order);
            }
        }
        // На малой системе один участок сохраняет последовательный SGS и снижает расходы на запуск.
        workers = std::min(workers, std::max<MeshIndex>(1, (n + 127) / 128));
        domain.resize(n);
        for (MeshIndex k = 0; k < workers; ++k) {
            offsets.push_back(k * (n / workers) + std::min(k, n % workers));
            const auto end = offsets.back() + n / workers + (k < n % workers ? 1 : 0);
            for (auto j = offsets.back(); j < end; ++j)
                domain[sequence[j]] = k;
        }
        offsets.push_back(n);
    }
};
// Граф CSGS строится на CPU один раз по тому же порядку, которым пользуется
// предобусловливатель. Все внедиагональные блоки попадают ровно в один проход.
struct CsgsLayout {
    std::vector<CsgsRow> rows;
    std::vector<MeshIndex> source, neighbours;
    unsigned lanes = 1;
    // Строит порядок строк и границы групп для проходов CSGS.
    void build(const CsgsOrder& order, const std::vector<MeshIndex>& starts,
               const std::vector<MeshIndex>& columns, bool gpu) {
        const auto n=order.sequence.size();
        rows.assign(n, CsgsRow{});
        // Определяет проход CSGS, к которому относится зависимость строки.
        const auto pass_for=[&](MeshIndex row, MeshIndex other, MeshIndex index) {
            return other<n && (order.domain.empty() || order.domain[row]==order.domain[other]) &&
                   order.position[other]<index ? 0u : 1u;
        };
        for (MeshIndex i=0;i<n;++i) {
            const auto row=order.sequence[i];
            for (auto p=starts[row]+1;p<starts[row+1];++p)
                ++rows[i].count[pass_for(row,order.row_of_node[columns[p]],i)];
        }
        // Для сильно неоднородного графа оставляем компактное хранение без
        // заполнителей. Допустимая добавка памяти для раскладки GPU — до половины числа связей.
        const auto offsets=[&](unsigned stride) {
            MeshIndex size=0;
            for (MeshIndex i=0;i<n;i+=stride) for (unsigned pass=0;pass<2;++pass) {
                MeshIndex longest=0;
                for (MeshIndex j=i;j<std::min(n,i+stride);++j) {
                    rows[j].first[pass]=size+j-i;
                    longest=std::max(longest,rows[j].count[pass]);
                }
                size+=stride*longest;
            }
            return size;
        };
        lanes=gpu ? 32 : 1;
        auto size=offsets(lanes);
        const auto entries=columns.size()-n;
        if (lanes>1 && size>entries+entries/2) { lanes=1; size=offsets(lanes); }
        source.assign(size,static_cast<MeshIndex>(-1));
        neighbours.resize(size);
        for (MeshIndex i=0;i<n;++i) {
            const auto row=order.sequence[i];
            MeshIndex next[]={rows[i].first[0],rows[i].first[1]};
            for (auto p=starts[row]+1;p<starts[row+1];++p) {
                const auto node=columns[p], other=order.row_of_node[node];
                const bool local=other<n && (order.domain.empty() || order.domain[row]==order.domain[other]);
                auto& pos=next[pass_for(row,other,i)];
                source[pos]=p;
                neighbours[pos]=local ? 2*other : 2*node+1;
                pos+=lanes;
            }
        }
    }
};
}
}
#endif
