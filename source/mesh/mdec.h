#ifndef AERO_MDEC_H
#define AERO_MDEC_H
#include "mesh.h"
#include "archive_format.h"
#include <array>
#include <fstream>
#include <map>
#include <set>

namespace aero {
namespace mdec {
using Id = MeshIndex;
using Address = std::pair<Id, Id>; // Номер листа и байтовое смещение в распакованной записи.
const Id absent = ~Id(0);
enum Kind { Node, Face, Cell };
void check(bool condition, const char* message);
Id read_word(const std::string& data, std::size_t offset);
void append_word(std::string& data, Id value);

// Записи храним в двоичном представлении архива. В ответ на запрос соседей можно скопировать запись целиком.
struct Record {
    // Запись исходного элемента целиком, включая тип, исходный номер, свойства и ссылки.
    std::string bytes;
    Id word(std::size_t offset) const { return read_word(bytes, offset); }
    Id kind() const { return word(0); }
    Id id() const { return word(8); }
    // Только для записи грани: порядок в исходной таблице границы и номер участка.
    Id boundary_order() const { return word(16); }
    Id boundary_patch() const { return word(24); }
    double coordinate(unsigned axis) const;
    // Число ссылок первого списка; их смысл зависит от типа записи.
    Id primary_count() const;
    Id secondary_count() const;
    // Извлекает адрес связанного элемента. У грани первичные ссылки ведут к вершинам, вторичные — к ячейкам.
    Address link(Id i, bool secondary = false) const;
    // Ориентация грани с локальным номером i внутри записи ячейки.
    std::int8_t sign(Id i) const;
};
using Records = std::map<Address, Record>;
using Addresses = std::set<Address>;

class Archive {
public:
    explicit Archive(const std::string& path);
    // Читает только сигнатуру, чтобы выбрать обычную сетку или архив разбиения при запуске.
    static bool matches(const std::string& path);
    // Возвращает границы последовательных диапазонов листьев для заданного числа процессов.
    std::vector<Id> partition(int processes);
    struct Prefix {
        Id raw = 0, nodes = 0, halo = 0;
    };
    Prefix prefix(Id end_leaf, bool fifth);
    void read_leaf(Id leaf, Records& records);
    bool indexed_halo() const { return version_ >= 2; }
    bool far_halo() const { return version_ >= 3; }
    // Добавляет все ячейки, инцидентные узлу, по обратному индексу архива.
    void cell_star(Id node, Addresses& cells);
    // Читает ядро листа или его дополнительный слой смежных ячеек.
    void leaf_cells(Id leaf, bool outer, Addresses& cells);
    // Дополняет множество дальними ячейками для шаблона пятого порядка.
    void far_cells(Id leaf, Addresses& cells);
    void read_records(const Addresses& needed, Kind kind, Records& records);
    // Идентификатор исходной сетки. Пуст для старых архивов без этого поля.
    std::string mesh_sha256;
    unsigned dimension;
    // Исходные числа узлов, граней и ячеек, до разбиения.
    std::array<Id, 3> counts;
    Id boundaries, leaves;
    std::vector<std::string> patches;

private:
    std::ifstream file_;
    struct Rows {
        // Смещения таблиц начал и адресов в файле, число строк и общее число ссылок.
        Id offsets, refs, count, entries;
    };
    using Branch = std::array<Id, 8>;
    void read_patch_names(Id names);
    Id read_halo_directory(Id end_tree);
    Branch branch(Id entry);
    Branch root();
    void children(const Branch& parent, Branch& left, Branch& right);
    Id row_offset(const Rows& rows, Id row);
    void read_row(const Rows& rows, Id row, Addresses& result);
    // Таблицы смежности узлов, ближайших слоёв и дальнего слоя с адресами в файле.
    Rows stars_{}, halo_{}, far_{};
    // Версия и длина файла, смещения каталога, дерева и данных, затем размер дерева и индекс корня.
    Id version_, length_, index_, tree_, payload_, entries_, root_;
    bool compressed_;
};
}
}
#endif
