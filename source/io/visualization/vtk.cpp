#include "vtk.h"
#include "binary_columns.h"
#include "../number_codec.h"
#include "../../mesh/mesh.h"
#include <algorithm>
#include <array>
#include <limits>
#include <locale>
#include <ostream>
#include <set>

namespace aero {
namespace vtk {
namespace {
// Размеры списков и номера узлов Legacy 4.2 хранятся в знаковом 32-битном слове.
void check_size(MeshIndex n) {
    if (n > static_cast<MeshIndex>(std::numeric_limits<std::int32_t>::max()))
        throw FieldError("VTK Legacy: connectivity or entity count exceeds INT32_MAX");
}

// Двоичные массивы VTK имеют сетевой порядок байтов. Собираем их в блоки по 64 КиБ.
class Block {
    std::ostream& out_;            // Принимает завершённые блоки.
    std::array<char, 65536> data_; // Переиспользуется между координатами, связностью и полями.
    std::size_t used_ = 0;         // Число уже занятых байтов.
    // Кодирует одно слово заданной ширины, старшим байтом вперёд.
    template<unsigned bytes> void word(std::uint64_t value) {
        if (used_ + bytes > data_.size()) flush();
        for (unsigned k = 0; k < bytes; ++k)
            data_[used_++] = static_cast<char>(value >> (8 * (bytes - 1 - k)));
    }
    // Передаёт заполненную часть буфера потоку.
    void flush() {
        out_.write(data_.data(), static_cast<std::streamsize>(used_));
        used_ = 0;
    }
public:
    // Связывает буфер с открытым двоичным файлом.
    explicit Block(std::ostream& out) : out_(out) {}
    // Записывает идентификатор, размер строки или код ячейки после проверки общих размеров.
    void index(MeshIndex value) { word<4>(value); }
    // Сохраняет исходное представление binary64 и знак нуля.
    void real(double value) { word<8>(number_codec::real_bits(value)); }
    // Завершает бинарный массив перед следующим текстовым заголовком.
    void end() { flush(); out_.put('\n'); }
};

// Числа ячеек разных типов в одном UNSTRUCTURED_GRID, в порядке следования на диске.
struct Groups {
    MeshIndex cells = 0; // Элементы исходной сетки или точки стенки.
    MeshIndex edges = 0; // Отрезки для полей, заданных на рёбрах.
    MeshIndex faces = 0; // Грани для полей на гранях.
    // Возвращает число записей CELL_TYPES и длину каждого столбца CELL_DATA.
    MeshIndex total() const { return cells + edges + faces; }
};

// Вычисляет число слов описания ячейки без первого слова длины.
MeshIndex cell_size(const MeshCell& cell, unsigned dimension) {
    if (dimension != 3) return cell.node_count();
    MeshIndex words = 1 + cell.face_count();
    for (MeshIndex j = 0; j < cell.face_count(); ++j) words += cell.face(j).node_count();
    return words;
}

// Для многоугольника список граней может идти в произвольном порядке. Связываем начало
// каждого ориентированного ребра с концом. Рабочий массив вмещает только одну ячейку.
void polygon(Block& block, const MeshCell& cell, std::vector<MeshIndex>& next) {
    const auto nodes = cell.node_ids(); // Номера вершин в этой таблице отсортированы сеткой.
    next.resize(nodes.size);
    for (MeshIndex j = 0; j < cell.face_count(); ++j) {
        const auto face = cell.face(j);
        const MeshIndex start = cell.face_orientation(j) > 0 ? 0 : 1;
        const auto slot = std::lower_bound(nodes.begin(), nodes.end(), face.node_id(start)) - nodes.begin();
        next[static_cast<std::size_t>(slot)] = face.node_id(1 - start);
    }
    MeshIndex node = nodes[0];
    for (MeshIndex j = 0; j < nodes.size; ++j) {
        block.index(node);
        node = next[static_cast<std::size_t>(std::lower_bound(nodes.begin(), nodes.end(), node) - nodes.begin())];
    }
}

// Пишет отрезок, контур многоугольника или ориентированные грани многогранника.
void write_cell(Block& block, const MeshCell& cell, unsigned dimension, std::vector<MeshIndex>& next) {
    block.index(cell_size(cell, dimension));
    if (dimension == 1) {
        const MeshIndex first = cell.face_orientation(0) < 0 ? 0 : 1;
        block.index(cell.face(first).node_id(0));
        block.index(cell.face(1 - first).node_id(0));
    } else if (dimension == 2) {
        polygon(block, cell, next);
    } else {
        block.index(cell.face_count());
        for (MeshIndex j = 0; j < cell.face_count(); ++j) {
            const auto nodes = cell.face(j).node_ids();
            block.index(nodes.size);
            for (MeshIndex k = 0; k < nodes.size; ++k)
                block.index(nodes[cell.face_orientation(j) > 0 ? k : nodes.size - 1 - k]);
        }
    }
}

// Вычисляет длину блока CELLS до записи его заголовка, затем выводит все типы геометрии.
void topology(std::ostream& out, Block& block, const MeshView& mesh, const Groups& groups, bool wall) {
    MeshIndex words = groups.total() + 2 * groups.edges;
    if (wall) words += groups.cells;
    else for (MeshIndex c = 0; c < groups.cells; ++c) words += cell_size(mesh.cell(c), mesh.dimension);
    for (MeshIndex f = 0; f < groups.faces; ++f) words += mesh.face_nodes(f).size;
    check_size(words);
    out << "CELLS " << groups.total() << ' ' << words << '\n';
    std::vector<MeshIndex> next;
    for (MeshIndex c = 0; c < groups.cells; ++c) {
        if (wall) { block.index(1); block.index(c); }
        else write_cell(block, mesh.cell(c), mesh.dimension, next);
    }
    for (MeshIndex e = 0; e < groups.edges; ++e) {
        block.index(2);
        block.index(mesh.edge_begin[e]);
        block.index(mesh.edge_end[e]);
    }
    for (MeshIndex f = 0; f < groups.faces; ++f) {
        const auto nodes = mesh.face_nodes(f);
        block.index(nodes.size);
        for (auto node : nodes) block.index(node);
    }
    block.end();
    out << "CELL_TYPES " << groups.total() << '\n';
    // Коды Legacy: вершина 1, отрезок 3, многоугольник 7, многогранник 42.
    const unsigned types[] = {1, 3, 7, 42};
    for (MeshIndex c = 0; c < groups.cells; ++c) block.index(wall ? 1 : types[mesh.dimension]);
    for (MeshIndex e = 0; e < groups.edges; ++e) block.index(3);
    for (MeshIndex f = 0; f < groups.faces; ++f) block.index(types[mesh.dimension - 1]);
    block.end();
}

// Экранирует пробелы и специальные байты метки по правилам имён VTK Legacy.
std::string vtk_name(const std::string& name) {
    const char digits[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned char c : name) {
        if (c > 32 && c < 127 && c != '%') result += static_cast<char>(c);
        else { result += '%'; result += digits[c >> 4]; result += digits[c & 15]; }
    }
    return result;
}

// В CELL_DATA все массивы имеют общую длину. Для элементов другого размещения
// выводим NaN прямо в поток, без временного массива размера всей сетки.
void attributes(std::ostream& out, Block& block, const FieldSnapshot& snapshot, const Groups& groups,
                bool point_data, bool wall, BinaryColumns* columns) {
    bool first = true;
    std::set<std::string> names;
    for (const auto& field : snapshot.fields) {
        if ((field.location == FieldLocation::Node) != point_data) continue;
        if (wall && (field.name == "x" || field.name == "y" || field.name == "z")) continue;
        const MeshIndex total = point_data ? snapshot.counts[0] : groups.total();
        if (first) out << (point_data ? "POINT_DATA " : "CELL_DATA ") << total << '\n';
        first = false;
        const MeshIndex count = snapshot.entity_count(field.location);
        const MeshIndex start = field.location == FieldLocation::Face ? groups.cells + groups.edges :
                                field.location == FieldLocation::Edge ? groups.cells : 0;
        for (std::size_t k = 0; k < field.components.size(); ++k) {
            std::string name = field.name;
            if (field.components.size() > 1 || field.components[k] != "value") name += "." + field.components[k];
            if (field.location == FieldLocation::Face) name = "face." + name;
            if (field.location == FieldLocation::Edge) name = "edge." + name;
            while (!names.insert(name).second) name += "_";
            out << "SCALARS " << vtk_name(name) << " double 1\nLOOKUP_TABLE default\n";
            for (MeshIndex i = 0; i < start; ++i) block.real(std::numeric_limits<double>::quiet_NaN());
            if (columns) {
                columns->visit(field, count, [&](const char* bytes, std::size_t n) {
                    for (std::size_t i = 0; i < n; ++i) block.real(number_codec::read_real(bytes + 8*i));
                });
            } else {
                for (MeshIndex i = 0; i < count; ++i) block.real(field.values[k * count + i]);
            }
            for (MeshIndex i = start + count; i < total; ++i) block.real(std::numeric_limits<double>::quiet_NaN());
            block.end();
        }
    }
}
}

// Формирует файл последовательно, сохраняя порядок узлов и ячеек снимка.
void write(std::ostream& out, const Mesh* geometry, const FieldSnapshot* snapshot, BinaryColumns* columns) {
    const bool wall = !geometry;
    const MeshView mesh = geometry ? geometry->view() : MeshView{};
    const MeshIndex points = wall ? snapshot->counts[0] : mesh.node_count;
    Groups groups;
    groups.cells = wall ? points : mesh.cell_count;
    if (snapshot) for (const auto& field : snapshot->fields) {
        if (field.location == FieldLocation::Edge) groups.edges = mesh.edge_count;
        if (field.location == FieldLocation::Face) groups.faces = mesh.face_count;
    }
    check_size(points);
    check_size(groups.total());
    out.imbue(std::locale::classic());
    out << "# vtk DataFile Version 4.2\nAero";
    if (snapshot) out << " step " << snapshot->step;
    out << "\nBINARY\nDATASET UNSTRUCTURED_GRID\n";
    Block block(out);
    if (snapshot) {
        out << "FIELD FieldData 1\nTIME 1 1 double\n";
        block.real(snapshot->time);
        block.end();
    }
    out << "POINTS " << points << " double\n";
    for (MeshIndex n = 0; n < points; ++n) {
        if (wall) for (unsigned k = 1; k <= 3; ++k) block.real(snapshot->fields[k].values[n]);
        else { block.real(mesh.x[n]); block.real(mesh.y[n]); block.real(mesh.z[n]); }
    }
    block.end();
    topology(out, block, mesh, groups, wall);
    if (snapshot) {
        attributes(out, block, *snapshot, groups, true, wall, columns);
        attributes(out, block, *snapshot, groups, false, wall, columns);
    }
    if (!out) throw FieldError("Cannot finish VTK output");
}
}
}
