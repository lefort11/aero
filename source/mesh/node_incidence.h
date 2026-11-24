#pragma once
#include "mesh.h"
#include "../parallel/parallel_for.h"
#include <atomic>

namespace aero {
// Обратная связность для подготовки сетки. В строке узла лежат номера всех
// содержащих его элементов. Порядок этих номеров не используется.
class NodeIncidence {
    MeshCsr rows_; // Одна строка на узел, одна запись на инцидентный ему элемент.
public:
    // Строит обратные связи от узлов к содержащим их элементам.
    template<class Vertices>
    NodeIncidence(MeshIndex nodes, MeshIndex elements, Vertices vertices) {
        // Соседние элементы могут увеличивать один счётчик. На первом проходе
        // считаем длины строк, чтобы сразу выделить непрерывный массив точного размера.
        std::vector<std::atomic<MeshIndex>> next(nodes);
        parallel_for(nodes, [&](MeshIndex n) { next[n].store(0, std::memory_order_relaxed); });
        parallel_for(elements, [&](MeshIndex e) {
            for (MeshIndex n : vertices(e)) next[n].fetch_add(1, std::memory_order_relaxed);
        });
        // После префиксной суммы каждый счётчик указывает на первое свободное место своей строки.
        rows_.offsets.resize(nodes + 1);
        for (MeshIndex n = 0; n < nodes; ++n) {
            rows_.offsets[n+1] = rows_.offsets[n] + next[n].load(std::memory_order_relaxed);
            next[n].store(rows_.offsets[n], std::memory_order_relaxed);
        }
        // Каждый поток резервирует одну позицию и записывает туда исходный номер элемента.
        rows_.indices.resize(rows_.offsets.back());
        parallel_for(elements, [&](MeshIndex e) {
            for (MeshIndex n : vertices(e))
                rows_.indices[next[n].fetch_add(1, std::memory_order_relaxed)] = e;
        });
    }
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    MeshIndexSpan operator[](MeshIndex node) const {
        return mesh_index_row(rows_.offsets.data(), rows_.indices.data(), node);
    }
};
}
