#include "meshdec.h"

namespace aero {
namespace meshdec {
namespace {
// Пара номера строки и значения для внешней сортировки halo-связей.
struct Pair {
    Id row, value;
    // Сравнивает ключи сущностей для упорядочения и поиска.
    bool operator<(const Pair& b) const { return row == b.row ? value < b.value : row < b.row; }
};

// Для каждой строки сохраняем ячейку один раз, даже когда она достижима через несколько граней.
// Временная таблица номеров нужна также для соединения с концами межлистовых рёбер.
void star_rows(const std::string& base, Sorter<Pair>& pairs, Id count) {
    Table<Pair> sorted(pairs.finish());
    auto starts = output(base + "-start"), cells = output(base + "-cells");
    Id pos = 0, entries = 0;
    for (Id row = 0; row < count; ++row) {
        put(starts, entries);
        Id previous = none;
        while (pos < sorted.count && sorted.at(pos).row == row) {
            const Id cell = sorted.at(pos++).value;
            if (cell == previous)
                continue;
            put(cells, cell);
            previous = cell;
            ++entries;
        }
    }
    put(starts, entries);
    require(pos == sorted.count, "Halo row out of range");
    starts.close();
    cells.close();
}

// Периодический представитель может получить ребро с противоположной границы.
// Нужно сохранить концы только тех рёбер, которые касаются физической границы.
void mark_boundary(const std::string& dir, Id count) {
    Table<Id> starts(dir + "/nf-start"), faces(dir + "/nf");
    Table<Boundary> labels(dir + "/face-bc");
    auto out = output(dir + "/boundary-nodes");
    for (Id n = 0; n < count; ++n) {
        std::uint8_t marked = 0;
        for (Id j = starts.at(n); j < starts.at(n + 1); ++j)
            if (labels.at(faces.at(j)).id != none) {
                marked = 1;
                break;
            }
        put(out, marked);
    }
    out.close();
}
// Выдаём вершины по одной, чтобы даже большой многоугольник или многогранник помещался в бюджет внешней сортировки.
struct CellNodes {
    Table<Id> cf, cf_start, fn, fn_start;
    // Открывает дисковые таблицы связей ячеек, граней и узлов.
    explicit CellNodes(const std::string& dir)
        : cf(dir + "/cf"), cf_start(dir + "/cf-start"), fn(dir + "/fn"), fn_start(dir + "/fn-start") {}
    // Перебирает узлы ячейки через дисковые связи ячейка–грани–узлы.
    template <class Visit> void each(Id c, Visit visit) {
        for (Id j = cf_start.at(c); j < cf_start.at(c + 1); ++j) {
            const Id f = cf.at(j);
            for (Id k = fn_start.at(f); k < fn_start.at(f + 1); ++k)
                visit(fn.at(k));
        }
    }
};
// Записывает строки смежности в два файла. Смещение строки — число уже записанных адресов.
class AddressRows {
public:
    // Открывает выходные таблицы смещений и ссылок строк halo.
    explicit AddressRows(const std::string& base)
        : offsets(output(base + "-start")), refs(output(base + "-refs")) {}
    // Запоминает начальное смещение новой строки ссылок.
    void start_row() { put(offsets, total); }
    // Добавляет очередную ссылку в текущую строку дисковой таблицы.
    void append(Address a) {
        write_address(refs, a);
        ++total;
    }
    // Записывает конечное смещение CSR и завершает таблицу ссылок.
    void finish() {
        // Конец последней строки нужен и при пустом последнем списке.
        start_row();
        offsets.close();
        refs.close();
    }

private:
    std::ofstream offsets, refs; // Начала строк и сами пары (лист, смещение).
    Id total = 0;                // Число пар во всех уже записанных строках.
};
// Строит звёзды ячеек для узлов на стыках листьев.
void boundary_stars(const std::string& dir, Table<Id>& needed, Table<Id>& starts, Table<Id>& incidence,
                    Table<Address>& cells) {
    AddressRows rows(dir + "/stars");
    Id i = 0;
    for (Id n = 0; n < starts.count - 1; ++n) {
        rows.start_row();
        if (i == needed.count || needed.at(i) != n)
            continue;
        while (i < needed.count && needed.at(i) == n)
            ++i;
        for (Id j = starts.at(n); j < starts.at(n + 1); ++j) {
            rows.append(cells.at(incidence.at(j)));
        }
    }
    rows.finish();
}

// У листа две непересекающиеся строки: ядро покрывает собственные вершины,
// а вторая строка содержит дополнительные ячейки для EBR3.
template <class Visit>
void leaf_rows(const std::string& dir, Table<Pair>& core, Table<Pair>& outer, Table<Address>& cells, Id count,
               Visit visit) {
    AddressRows rows(dir + "/halo");
    Id i = 0, j = 0;
    // Добавляет очередную геометрическую сущность или связь в собираемый список.
    const auto emit = [&](Id leaf, Id c) {
        visit(leaf, c);
        rows.append(cells.at(c));
    };
    for (Id leaf = 0; leaf < count; ++leaf) {
        Id k = i, previous = none;
        rows.start_row();
        while (i < core.count && core.at(i).row == leaf) {
            const Id c = core.at(i++).value;
            if (c != previous)
                emit(leaf, c);
            previous = c;
        }
        rows.start_row();
        previous = none;
        while (j < outer.count && outer.at(j).row == leaf) {
            const Id c = outer.at(j++).value;
            if (c == previous)
                continue;
            while (k < i && core.at(k).value < c)
                ++k;
            if (k == i || core.at(k).value != c)
                emit(leaf, c);
            previous = c;
        }
    }
    require(i == core.count && j == outer.count, "Halo leaf out of range");
    rows.finish();
}

// Для каждой запрошенной листом вершины находим список смежных ячеек один раз.
void expand_stars(Table<Pair>& centers, Table<Id>& starts, Table<Id>& incidence, Sorter<Pair>& output) {
    Pair previous = {none, none};
    for (Id i = 0; i < centers.count; ++i) {
        const auto p = centers.at(i);
        if (p.row == previous.row && p.value == previous.value)
            continue;
        for (Id j = starts.at(p.value); j < starts.at(p.value + 1); ++j)
            output.add({p.row, incidence.at(j)});
        previous = p;
    }
}
// Строит дополнительные строки дальнего halo для расширенных шаблонов.
void far_rows(const std::string& dir, Table<Pair>& core, Table<Pair>& outer, Table<Pair>& far,
              Table<Address>& cells, Id leaves) {
    AddressRows rows(dir + "/far");
    Id i = 0, j = 0, k = 0;
    for (Id leaf = 0; leaf < leaves; ++leaf) {
        rows.start_row();
        Id previous = none;
        while (k < far.count && far.at(k).row == leaf) {
            const auto p = far.at(k++);
            if (previous == p.value)
                continue;
            previous = p.value;
            while (i < core.count && core.at(i) < p)
                ++i;
            while (j < outer.count && outer.at(j) < p)
                ++j;
            if ((i < core.count && !(p < core.at(i))) || (j < outer.count && !(p < outer.at(j))))
                continue;
            rows.append(cells.at(p.value));
        }
    }
    rows.finish();
}
}

// Готовит ближние и дальние связи halo для архива разбиения.
void plan_halo(const std::string& dir, Id memory) {
    const auto h = Table<Header>(dir + "/header").at(0);
    CellNodes topology(dir);
    auto& cf = topology.cf;
    auto& cf_start = topology.cf_start;
    auto& fn = topology.fn;
    auto& fn_start = topology.fn_start;
    Table<Address> nodes(dir + "/address-0"), cells(dir + "/address-2");
    Sorter<Pair> stars(dir + "/stars.sort", memory);
    for (Id c = 0; c < h.cells; ++c)
        topology.each(c, [&](Id n) { stars.add({n, c}); });
    star_rows(dir + "/incidence", stars, h.nodes);
    Table<Id> start(dir + "/incidence-start"), incidence(dir + "/incidence-cells");
    Sorter<Pair> core(dir + "/core.sort", memory);
    for (Id n = 0; n < h.nodes; ++n) {
        const Id leaf = nodes.at(n).leaf;
        for (Id j = start.at(n); j < start.at(n + 1); ++j)
            core.add({leaf, incidence.at(j)});
    }
    Table<Pair> core_cells(core.finish());

    // EBR3 нужны все ячейки вокруг противоположного конца ребра. Для узлов того же листа они уже входят в ядро.
    mark_boundary(dir, h.nodes);
    Table<std::uint8_t> boundary(dir + "/boundary-nodes");
    Sorter<Pair> neighbors(dir + "/cross-edges.sort", memory / 2);
    Sorter<Id> near_patch(dir + "/near-patch.sort", memory / 2);
    // Добавляет связь концов ребра в построение halo.
    const auto edge = [&](Id a, Id b) {
        const Id left = nodes.at(a).leaf, right = nodes.at(b).leaf;
        if (left != right) {
            neighbors.add({left, b});
            neighbors.add({right, a});
        }
        if (boundary.at(a) || boundary.at(b)) {
            near_patch.add(a);
            near_patch.add(b);
        }
    };
    if (h.dimension == 1) {
        for (Id c = 0; c < h.cells; ++c) {
            const Id j = cf_start.at(c);
            edge(fn.at(fn_start.at(cf.at(j))), fn.at(fn_start.at(cf.at(j + 1))));
        }
    } else {
        for (Id f = 0; f < h.faces; ++f) {
            const Id first = fn_start.at(f), end = fn_start.at(f + 1);
            for (Id j = first + 1; j < end; ++j)
                edge(fn.at(j - 1), fn.at(j));
            if (h.dimension == 3)
                edge(fn.at(end - 1), fn.at(first));
        }
    }
    Table<Id> needed(near_patch.finish());
    Table<Pair> sorted(neighbors.finish());
    // Периодические пары выбираются при запуске. Сохраняем окружения вершин в той полосе у границы,
    // куда может попасть первый луч после перехода через будущий шов.
    Sorter<Id> band_cells(dir + "/band-cells.sort", memory);
    for (Id i = 0; i < needed.count; ++i) {
        const Id n = needed.at(i);
        if (i && n == needed.at(i - 1))
            continue;
        for (Id j = start.at(n); j < start.at(n + 1); ++j)
            band_cells.add(incidence.at(j));
    }
    Table<Id> band(band_cells.finish());
    Sorter<Id> band_nodes(dir + "/band-nodes.sort", memory);
    for (Id i = 0; i < band.count; ++i)
        if (!i || band.at(i) != band.at(i - 1))
            topology.each(band.at(i), [&](Id n) { band_nodes.add(n); });
    Table<Id> boundary_support(band_nodes.finish());
    boundary_stars(dir, boundary_support, start, incidence, cells);
    Sorter<Pair> halo(dir + "/halo.sort", memory);
    expand_stars(sorted, start, incidence, halo);
    Table<Pair> outer(halo.finish());
    const Id leaves = Table<Leaf>(dir + "/leaves").count;
    Sorter<Pair> far_nodes(dir + "/far-nodes.sort", memory);
    leaf_rows(dir, core_cells, outer, cells, leaves, [&](Id leaf, Id c) {
        // На первое пересечение луча может повлиять любая вершина ячейки, в том числе через диагональ грани.
        // Окружения собственных вершин уже собраны.
        topology.each(c, [&](Id n) {
            if (nodes.at(n).leaf != leaf)
                far_nodes.add({leaf, n});
        });
    });
    Table<Pair> centers(far_nodes.finish());
    Sorter<Pair> far_cells(dir + "/far-cells.sort", memory);
    expand_stars(centers, start, incidence, far_cells);
    Table<Pair> far(far_cells.finish());
    far_rows(dir, core_cells, outer, far, cells, leaves);
}
}
}
