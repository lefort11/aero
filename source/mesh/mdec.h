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
// Проверяет условие целостности архива и выбрасывает MeshError при нарушении.
void check(bool condition, const char* message);
// Читает целое слово из двоичного представления.
Id read_word(const std::string& data, std::size_t offset);
// Добавляет 64-битную величину в сообщение в том же формате, что и запись на диске.
void append_word(std::string& data, Id value);

// Записи храним в двоичном представлении архива. В ответ на запрос соседей можно скопировать запись целиком.
struct Record {
    // Запись исходного элемента целиком, включая тип, исходный номер, свойства и ссылки.
    std::string bytes;
    // Извлекает целое слово по смещению в записи архива.
    Id word(std::size_t offset) const { return read_word(bytes, offset); }
    // Возвращает тип геометрической сущности, закодированный в записи.
    Id kind() const { return word(0); }
    // Возвращает исходный глобальный номер сущности из записи архива.
    Id id() const { return word(8); }
    // Только для записи грани: порядок в исходной таблице границы и номер участка.
    Id boundary_order() const { return word(16); }
    // Читает номер граничного участка из записи грани.
    Id boundary_patch() const { return word(24); }
    // Читает координату узла по смещению её битового представления.
    double coordinate(unsigned axis) const;
    // Число ссылок первого списка; их смысл зависит от типа записи.
    Id primary_count() const;
    // Возвращает число вторичных связей записи сущности архива.
    Id secondary_count() const;
    // Извлекает адрес связанного элемента. У грани первичные ссылки ведут к вершинам, вторичные — к ячейкам.
    Address link(Id i, bool secondary = false) const;
    // Ориентация грани с локальным номером i внутри записи ячейки.
    std::int8_t sign(Id i) const;
};
using Records = std::map<Address, Record>;
using Addresses = std::set<Address>;

// Читает каталог, геометрию и halo-связи архива разбиения сетки.
class Archive {
public:
    // Открывает архив и проверяет его каталог перед обращением к адресам отдельных листьев.
    explicit Archive(const std::string& path);
    // Читает только сигнатуру, чтобы выбрать обычную сетку или архив разбиения при запуске.
    static bool matches(const std::string& path);
    // Возвращает границы последовательных диапазонов листьев для заданного числа процессов.
    std::vector<Id> partition(int processes);
    // Накопленные размеры геометрии, узлов и halo до границы листа.
    struct Prefix {
        // Накопленные размеры исходных данных, узлов и ореола.
        Id raw = 0, nodes = 0, halo = 0;
    };
    // Читает накопленные размеры записей до указанной границы листа.
    Prefix prefix(Id end_leaf, bool fifth);
    // Загружает упакованные записи геометрии одного листа.
    void read_leaf(Id leaf, Records& records);
    // Проверяет наличие индексированного каталога ближнего halo.
    bool indexed_halo() const { return version_ >= 2; }
    // Проверяет наличие сохранённых связей дальнего halo.
    bool far_halo() const { return version_ >= 3; }
    // Добавляет все ячейки, инцидентные узлу, по обратному индексу архива.
    void cell_star(Id node, Addresses& cells);
    // Читает ядро листа или его дополнительный слой смежных ячеек.
    void leaf_cells(Id leaf, bool outer, Addresses& cells);
    // Дополняет множество дальними ячейками для шаблона пятого порядка.
    void far_cells(Id leaf, Addresses& cells);
    // Получает недостающие записи из архива по их адресам и добавляет в локальный набор.
    void read_records(const Addresses& needed, Kind kind, Records& records);
    // Идентификатор исходной сетки. Пуст для старых архивов без этого поля.
    std::string mesh_sha256;
    // Размерность сетки в архиве.
    unsigned dimension;
    // Исходные числа узлов, граней и ячеек, до разбиения.
    std::array<Id, 3> counts;
    // Количество граничных записей и листьев разбиения.
    Id boundaries, leaves;
    // Имена граничных патчей по номерам архива.
    std::vector<std::string> patches;

private:
    // Открытый поток для выборочного чтения разделов.
    std::ifstream file_;
    // Файловые смещения, число строк и количество ссылок дисковой CSR-таблицы.
    struct Rows {
        // Смещения таблиц начал и адресов в файле, число строк и общее число ссылок.
        Id offsets, refs, count, entries;
    };
    // До восьми дочерних записей узла дерева.
    using Branch = std::array<Id, 8>;
    // Таблица имён занимает точный диапазон между заголовком и началом индекса листьев.
    void read_patch_names(Id names);
    // Числа и смещения описывают строки CSR. Сами списки ячеек читаем позже только по запросу своего ранга.
    Id read_halo_directory(Id end_tree);
    // Читает запись ветви дерева разбиения.
    Branch branch(Id entry);
    // Возвращает корневую запись дерева или представителя множества.
    Branch root();
    // Возвращает дочерние записи ветви разбиения.
    void children(const Branch& parent, Branch& left, Branch& right);
    // Вычисляет файловое смещение строки индексированной таблицы.
    Id row_offset(const Rows& rows, Id row);
    // Читает ссылки одной строки дисковой таблицы архива.
    void read_row(const Rows& rows, Id row, Addresses& result);
    // Таблицы смежности узлов, ближайших слоёв и дальнего слоя с адресами в файле.
    Rows stars_{}, halo_{}, far_{};
    // Версия и длина файла, смещения каталога, дерева и данных, затем размер дерева и индекс корня.
    Id version_, length_, index_, tree_, payload_, entries_, root_;
    // Признак сжатого представления записей.
    bool compressed_;
};
}
}
#endif
