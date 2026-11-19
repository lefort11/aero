#ifndef AERO_MESHDEC_H
#define AERO_MESHDEC_H
#include "storage.h"

namespace aero {
namespace meshdec {
struct Options {
    // Исходная сетка, готовый архив и каталог промежуточных таблиц.
    std::string source, target, temp;
    // Предел памяти сортировки, целевой размер листа в байтах и желаемое число разделов.
    Id memory = 64 * 1024 * 1024, leaf_bytes = 4 * 1024 * 1024, target_ranks = 0;
    // Включает независимое сжатие данных каждого листа при упаковке.
    bool compress = false;
};
// Сводка исходной сетки, необходимая при построении временных таблиц и каталога архива.
struct Header {
    Id dimension, nodes, faces, face_entries, cells, cell_entries, boundaries, patches;
    double scale, low[3], high[3];
};
// Ключ пространственной сортировки и размер будущей записи узла, грани или ячейки.
struct Entity {
    Id morton, kind, id, bytes;
    bool operator<(const Entity& other) const {
        if (morton != other.morton)
            return morton < other.morton;
        if (kind != other.kind)
            return kind < other.kind;
        return id < other.id;
    }
};
// Номер листа и смещение записи в его распакованных данных.
struct Address {
    Id leaf, offset;
};
// Адрес ячейки или грани в архиве всегда состоит из двух последовательных 64-битных слов.
inline void write_address(std::ostream& out, Address a) {
    word(out, a.leaf);
    word(out, a.offset);
}
struct Boundary {
    Id id, patch;
};
struct Box {
    double low[3], high[3];
    static Box point(double x, double y, double z) { return {{x, y, z}, {x, y, z}}; }
    void include(const Box& other) {
        for (unsigned k = 0; k < 3; ++k) {
            low[k] = std::min(low[k], other.low[k]);
            high[k] = std::max(high[k], other.high[k]);
        }
    }
};
// Диапазон отсортированных записей одного листа и количества элементов для каталога.
struct Leaf {
    Id first, records, raw_bytes, nodes, faces, cells;
};
struct Chunk {
    Id begin, end;
};
// Делит последовательность без пересечений; первые процессы получают по одной лишней записи остатка.
inline Chunk divide(Id count, int rank, int size) {
    const Id n = count / static_cast<unsigned>(size), extra = count % static_cast<unsigned>(size);
    const Id r = static_cast<unsigned>(rank), start = r * n + std::min(r, extra);
    return {start, start + n + (r < extra ? 1u : 0u)};
}
inline std::string part(const std::string& base, int rank) {
    return base + "-" + std::to_string(rank);
}
// Распаковывает исходную сетку во временные таблицы с ограниченной памятью.
void unpack(const Options&, const std::string& dir);
// Строит обратные связи, по которым позднее выбираются соседние ячейки.
void index_topology(const std::string& dir, Id memory);
// Присваивает исходным записям пространственные ключи и сортирует свою часть.
void order_entities(const std::string& dir, Id memory, int rank, int size);
// По выборке ключей выбирает границы диапазонов для параллельной сортировки.
void choose_splitters(const std::string& dir, Id memory, int size);
// Сводит записи своего диапазона в окончательный пространственный порядок.
void sort_range(const std::string& dir, Id memory, int rank, int size);
// Распределяет отсортированные записи по листьям и назначает им адреса.
void plan_leaves(const std::string& dir, const Options&, int size);
// Готовит таблицы соседства, необходимые при чтении подобласти и построении шаблона.
void plan_halo(const std::string& dir, Id memory);
// Записывает полезные данные листьев процесса, при запросе применяя блочное сжатие.
void pack_leaves(const std::string& dir, bool compress, int rank, int size);
// Собирает заголовок и каталог в подготовленном файле и возвращает число листьев.
Id write_catalog(const std::string& dir, const std::string& staged, bool compress, int size);
// Помещает подготовленные байты данного процесса в его диапазон итогового архива.
void write_part(const std::string& dir, const std::string& staged, int rank);
}
}
#endif
