#include "meshdec.h"
#include "../../source/mesh/archive_format.h"
#include "../../source/io/lzma_stream.h"

namespace aero {
namespace meshdec {
namespace {
// Размер упакованного фрагмента и его ограничивающий параллелепипед.
struct Packed {
    Id bytes;
    Box box;
};
// Записывает минимальные и максимальные координаты ограничивающего параллелепипеда.
void write_box(std::ostream& out, const Box& box) {
    for (double v : box.low)
        real(out, v);
    for (double v : box.high)
        real(out, v);
}

// Читает промежуточные дисковые таблицы координат и топологических связей.
class Geometry {
public:
    // Открывает набор промежуточных таблиц геометрии из рабочего каталога.
    explicit Geometry(const std::string& d)
        : x(d + "/X"), y(d + "/Y"), z(d + "/Z"), nf(d + "/nf"), fn(d + "/fn"), fc(d + "/fc"), cf(d + "/cf"),
          nf_start(d + "/nf-start"), fn_start(d + "/fn-start"), fc_start(d + "/fc-start"),
          cf_start(d + "/cf-start"), signs(d + "/signs"), bc(d + "/face-bc"), boxes(d + "/face-box"),
          nodes(d + "/address-0"), faces(d + "/address-1"), cells(d + "/address-2") {}
    // Сериализует геометрическую запись сущности из промежуточных дисковых таблиц.
    Box write(std::ostream& out, const Entity& e) {
        const Id i = e.id;
        word(out, e.kind);
        word(out, i);
        if (e.kind == 0) {
            const Box point = Box::point(x.at(i), y.at(i), z.at(i));
            for (double v : point.low)
                real(out, v);
            word(out, nf_start.at(i + 1) - nf_start.at(i));
            for (Id k = nf_start.at(i); k < nf_start.at(i + 1); ++k)
                write_address(out, faces.at(nf.at(k)));
            return point;
        }
        if (e.kind == 1) {
            const auto label = bc.at(i);
            word(out, label.id);
            word(out, label.patch);
            word(out, fn_start.at(i + 1) - fn_start.at(i));
            word(out, fc_start.at(i + 1) - fc_start.at(i));
            for (Id k = fn_start.at(i); k < fn_start.at(i + 1); ++k)
                write_address(out, nodes.at(fn.at(k)));
            for (Id k = fc_start.at(i); k < fc_start.at(i + 1); ++k)
                write_address(out, cells.at(fc.at(k)));
            return boxes.at(i);
        }
        const Id first = cf_start.at(i), end = cf_start.at(i + 1);
        word(out, end - first);
        Box box{};
        for (Id k = first; k < end; ++k) {
            const Id f = cf.at(k);
            write_address(out, faces.at(f));
            out.put(static_cast<char>(signs.at(k)));
            if (k == first)
                box = boxes.at(f);
            else
                box.include(boxes.at(f));
        }
        return box;
    }

private:
    Table<double> x, y, z;
    Table<Id> nf, fn, fc, cf, nf_start, fn_start, fc_start, cf_start;
    Table<std::int8_t> signs;
    Table<Boundary> bc;
    Table<Box> boxes;
    Table<Address> nodes, faces, cells;
};

// Запись ветви дерева разбиения с размерами и геометрическими границами.
struct Branch {
    Id index, bytes, nodes, faces, cells;
    Box box;
};
// Записываем дочерние узлы раньше родителя. В стеке хранится один путь по дереву,
// поэтому индекс может быть больше доступной оперативной памяти.
Branch tree(std::ostream& out, Table<Leaf>& leaves, Table<Packed>& packed, Id first, Id count, Id& next) {
    Branch b{};
    Id left = none, right = none;
    if (count == 1) {
        const auto leaf = leaves.at(first);
        b = {0, leaf.raw_bytes, leaf.nodes, leaf.faces, leaf.cells, packed.at(first).box};
    } else {
        const Id half = count / 2;
        b = tree(out, leaves, packed, first, half, next);
        const auto other = tree(out, leaves, packed, first + half, count - half, next);
        left = b.index;
        right = other.index;
        b.bytes = plus(b.bytes, other.bytes);
        b.nodes = plus(b.nodes, other.nodes);
        b.faces = plus(b.faces, other.faces);
        b.cells = plus(b.cells, other.cells);
        b.box.include(other.box);
    }
    b.index = next++;
    word(out, first);
    word(out, count);
    word(out, left);
    word(out, right);
    word(out, b.bytes);
    word(out, b.nodes);
    word(out, b.faces);
    word(out, b.cells);
    write_box(out, b.box);
    return b;
}
}

// Упаковывает сущности листьев и вычисляет их размеры и геометрические границы.
void pack_leaves(const std::string& dir, bool compress, int rank, int size) {
    Table<Leaf> leaves(dir + "/leaves");
    Table<Entity> ordered(dir + "/ordered");
    Geometry geometry(dir);
    const auto range = divide(leaves.count, rank, size);
    auto payload = output(part(dir + "/payload", rank)), index = output(part(dir + "/packed", rank));
    for (Id i = range.begin; i < range.end; ++i) {
        const auto leaf = leaves.at(i);
        const auto start = payload.tellp();
        Box box{};
        // Записывает подготовленный блок данных в часть архива.
        const auto write = [&](std::ostream& stream) {
            for (Id k = 0; k < leaf.records; ++k) {
                const auto extent = geometry.write(stream, ordered.at(leaf.first + k));
                if (!k)
                    box = extent;
                else
                    box.include(extent);
            }
        };
        if (compress) {
            io::LzmaOutputBuffer buffer(payload);
            std::ostream stream(&buffer);
            stream.exceptions(std::ios::failbit | std::ios::badbit);
            write(stream);
            buffer.finish();
        } else
            write(payload);
        const Id bytes = static_cast<Id>(payload.tellp() - start);
        require(compress || bytes == leaf.raw_bytes, "Leaf size mismatch");
        put(index, Packed{bytes, box});
    }
    payload.close();
    index.close();
}

// Записывает каталог листьев и дерево ветвей архива.
Id write_catalog(const std::string& dir, const std::string& staged, bool compress, int size) {
    const auto h = Table<Header>(dir + "/header").at(0);
    Table<Leaf> leaves(dir + "/leaves");
    auto merged = output(dir + "/packed");
    for (int r = 0; r < size; ++r) {
        const auto path = part(dir + "/packed", r);
        auto in = input(path);
        copy_bytes(in, merged, file_size(path));
    }
    merged.close();
    Table<Packed> packed(dir + "/packed");
    require(packed.count == leaves.count, "Missing packed leaves");
    const Id names = archive_format::header_bytes, index = plus(names, file_size(dir + "/names"));
    const Id branches = plus(index, times(archive_format::leaf_entry_bytes, leaves.count));
    const Id entries = leaves.count ? times(2, leaves.count) - 1 : 0;
    const Id directory = plus(branches, times(archive_format::tree_entry_bytes, entries));
    const Id star_bytes = file_size(dir + "/stars-refs"), halo_bytes = file_size(dir + "/halo-refs");
    const Id rows = plus(plus(h.nodes, times(2, leaves.count)), 2);
    const Id far_index = plus(plus(directory, 16), plus(times(8, rows), plus(star_bytes, halo_bytes)));
    const Id far_bytes = file_size(dir + "/far-refs");
    const Id payload = plus(plus(far_index, 64), plus(times(8, plus(leaves.count, 2)), far_bytes));
    auto out = output(staged);
    out.write(archive_format::signature, sizeof(archive_format::signature) - 1);
    word(out, archive_format::current_version);
    word(out, compress ? 1 : 0);
    word(out, h.dimension);
    word(out, h.nodes);
    word(out, h.faces);
    word(out, h.cells);
    word(out, h.boundaries);
    word(out, h.patches);
    word(out, leaves.count);
    word(out, entries);
    word(out, entries ? entries - 1 : none);
    word(out, names);
    word(out, index);
    word(out, branches);
    word(out, payload);
    auto strings = input(dir + "/names");
    copy_bytes(strings, out, index - names);
    Id offset = payload;
    for (Id i = 0; i < leaves.count; ++i) {
        const auto l = leaves.at(i);
        const Id length = packed.at(i).bytes;
        word(out, offset);
        word(out, length);
        word(out, l.raw_bytes);
        word(out, l.nodes);
        word(out, l.faces);
        word(out, l.cells);
        offset = plus(offset, length);
    }
    Id next = 0;
    if (leaves.count)
        tree(out, leaves, packed, 0, leaves.count, next);
    require(next == entries && out.tellp() == position(directory), "Catalog size mismatch");
    word(out, star_bytes / 16);
    word(out, halo_bytes / 16);
    // Сериализует диапазон строк индексированной таблицы архива.
    const auto write_rows = [&](const std::string& name) {
        const auto path = dir + "/" + name;
        Table<Id> starts(path + "-start");
        for (Id i = 0; i < starts.count; ++i)
            word(out, starts.at(i));
        auto refs = input(path + "-refs");
        copy_bytes(refs, out, file_size(path + "-refs"));
    };
    write_rows("stars");
    write_rows("halo");
    word(out, far_bytes / 16);
    write_rows("far");
    auto digest = input(dir + "/mesh-sha256");
    copy_bytes(digest, out, 64);
    require(out.tellp() == position(payload), "Halo directory size mismatch");
    if (offset > payload) {
        out.seekp(position(offset - 1));
        out.put(0);
    }
    out.close();
    auto starts = output(dir + "/output-start");
    Id begin = payload;
    for (int r = 0; r < size; ++r) {
        put(starts, begin);
        begin = plus(begin, file_size(part(dir + "/payload", r)));
    }
    starts.close();
    require(begin == offset, "Payload size mismatch");
    return leaves.count;
}
// Записывает назначенную рабочему процессу часть архива разбиения.
void write_part(const std::string& dir, const std::string& staged, int rank) {
    const auto path = part(dir + "/payload", rank);
    const Id offset = Table<Id>(dir + "/output-start").at(static_cast<unsigned>(rank));
    auto in = input(path);
    std::fstream out(staged.c_str(), std::ios::in | std::ios::out | std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.seekp(position(offset));
    copy_bytes(in, out, file_size(path));
    out.close();
}
}
}
