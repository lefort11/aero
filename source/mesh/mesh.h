#ifndef AERO_MESH_H
#define AERO_MESH_H
#include "../parallel/kernel_math.h"

#include "mesh_index.h"
#include "coordinate.h"
#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace aero {

enum class MeshFormat { Text, Binary, Lzma };

namespace mesh_format {
constexpr std::uint64_t legacy_version = 2, current_version = 3;
// Формат полей фиксирует SHA-256 потока v2. Для новой сетки его служебное
// слово равно представлению числа 1.0 в IEEE-754; геометрия его не использует.
constexpr std::uint64_t default_fingerprint_word = 0x3ff0000000000000ULL;
}

// Ошибка геометрии, топологии или формата сетки.
class MeshError : public std::runtime_error {
public:
    // Создаёт исключение с переданным диагностическим сообщением.
    explicit MeshError(const std::string& message) : std::runtime_error(message) {}
};

// Строка i в CSR занимает индексы от offsets[i] включительно до offsets[i+1] исключительно. Нумерация с нуля.
struct MeshCsr {
    // Начальные смещения строк и конец последней строки.
    std::vector<MeshIndex> offsets;
    // Номера сущностей, последовательно собранные по строкам.
    std::vector<MeshIndex> indices;
    // Инициализирует CSR-строки; пустая таблица содержит единственное нулевое смещение.
    MeshCsr() : offsets(1, 0) {}
    // Инициализирует CSR-строки; пустая таблица содержит единственное нулевое смещение.
    MeshCsr(std::vector<MeshIndex> row_offsets, std::vector<MeshIndex> entries)
        : offsets(std::move(row_offsets)), indices(std::move(entries)) {}
};

// Входные координаты, топологические связи и граничные метки сетки.
struct MeshData {
    // Топологическая размерность равна 1, 2 или 3. Координаты всегда хранятся по трём осям.
    unsigned dimension;
    // Покомпонентные координаты узлов.
    std::vector<double> x, y, z;
    // Грань в 1D — вершина, в 2D — направленное ребро, в 3D — циклически упорядоченный многоугольник.
    MeshCsr face_nodes;
    // Ячейки ссылаются на номера граней. Общая грань хранится один раз.
    MeshCsr cell_faces;
    // Знаки идут в том же порядке, что и cell_faces.indices. В 1D начало имеет знак −1, конец +1.
    // В 2D знак +1 означает обход ребра по его направлению, в 3D — наружную ориентацию цикла грани.
    // Знак −1 обращает соответствующее направление.
    std::vector<std::int8_t> cell_face_orientation;
    // Номера граней на физической границе.
    std::vector<MeshIndex> boundary_faces;
    // Номер патча каждой граничной грани.
    std::vector<MeshIndex> boundary_patch;
    // Здесь хранятся имена участков. Физические условия и периодические пары задаются в постановке.
    std::vector<std::string> patch_names;
    // Создаёт пустую трёхмерную заготовку сетки.
    MeshData() : dimension(3) {}
};

// Невладеющий диапазон последовательных индексов сетки.
struct MeshIndexSpan {
    // Начало непрерывного диапазона номеров.
    const MeshIndex* data;
    // Количество номеров в диапазоне.
    MeshIndex size;
    // Связывает указатель и длину невладеющего диапазона индексов.
    AERO_HOST_DEVICE MeshIndexSpan(const MeshIndex* values = nullptr, MeshIndex count = 0)
        : data(values), size(count) {}
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE MeshIndex operator[](MeshIndex i) const noexcept { return data[i]; }
    // Возвращает начало диапазона для последовательного обхода.
    AERO_HOST_DEVICE const MeshIndex* begin() const noexcept { return data; }
    // Возвращает границу диапазона после последнего элемента.
    AERO_HOST_DEVICE const MeshIndex* end() const noexcept { return size ? data + size : data; }
};

// Любая связь сетки задаётся одной строкой CSR. Смежные номера лежат подряд между двумя смещениями.
// В частности, у точечной грани в 1D строка рёбер пуста, а указатель на их массив может быть нулевым.
AERO_HOST_DEVICE inline MeshIndexSpan mesh_index_row(const MeshIndex* offsets, const MeshIndex* indices,
                                                     MeshIndex row) noexcept {
    const MeshIndex first = offsets[row], count = offsets[row + 1] - first;
    return MeshIndexSpan(count ? indices + first : nullptr, count);
}

class MeshNode;
class MeshEdge;
class MeshFace;
class MeshCell;

// Представление ссылается на массивы владельца. Инициализируйте его как MeshView{}.
// Копирование указателей не переносит данные на GPU. Адреса должны быть доступны выбранному исполнителю.
// Индексы должны быть проверены вызывающим кодом. Имена остаются на CPU.
struct MeshView {
    // Число активных координат сетки.
    unsigned dimension;
    // Размеры таблиц топологических сущностей и патчей.
    MeshIndex node_count, face_count, cell_count, edge_count, boundary_face_count, patch_count;
    // Первые координаты узлов.
    const double* x;
    // Вторые координаты узлов.
    const double* y;
    // Третьи координаты узлов.
    const double* z;
    // Возвращает координаты одного узла из трёх массивов сетки.
    AERO_HOST_DEVICE Coordinate coordinate(MeshIndex node) const { return {{x[node], y[node], z[node]}}; }
    // Связь «грань → вершины»: смещения и номера узлов в порядке обхода грани.
    const MeshIndex* face_offsets;
    // Номера узлов по граням.
    const MeshIndex* face_indices;
    // Связь «ячейка → грани». Знак в той же позиции задаёт ориентацию грани в ячейке.
    const MeshIndex* cell_offsets;
    // Номера граней по ячейкам.
    const MeshIndex* cell_indices;
    // Знак обхода каждой грани относительно ячейки.
    const signed char* cell_orientation;
    // Параллельные массивы: номер внешней грани и номер её граничного участка.
    const MeshIndex* boundary_faces;
    // Соответствие граничной грани патчу.
    const MeshIndex* boundary_patch;
    // Концы ориентированного ребра. Положительный поток идёт от begin к end.
    const MeshIndex* edge_begin;
    // Конечный узел ориентированного ребра.
    const MeshIndex* edge_end;
    // Отдельная CSR-таблица рёбер грани: ноль рёбер в 1D, одно в 2D, цикл в 3D.
    const MeshIndex* face_edge_offsets;
    // Список рёбер каждой грани.
    const MeshIndex* face_edge_indices;
    // Производные строки узлов и рёбер ячейки. Их строят один раз при загрузке сетки.
    const MeshIndex* cell_node_offsets;
    // Список узлов каждой ячейки.
    const MeshIndex* cell_node_indices;
    // Начальные смещения списков рёбер ячеек.
    const MeshIndex* cell_edge_offsets;
    // Рёбра ячеек в порядке строк.
    const MeshIndex* cell_edge_indices;

    // Методы Mesh проверяют номер элемента и выбрасывают MeshError при ошибке.
    // Во вложенных обращениях проверка локальной позиции остаётся обязанностью вызывающего кода.
    AERO_HOST_DEVICE MeshNode node(MeshIndex id) const noexcept;
    // Возвращает дескриптор ребра в соответствующей нумерации сетки.
    AERO_HOST_DEVICE MeshEdge edge(MeshIndex id) const noexcept;
    // Возвращает дескриптор грани в соответствующей нумерации сетки.
    AERO_HOST_DEVICE MeshFace face(MeshIndex id) const noexcept;
    // Возвращает дескриптор ячейки в соответствующей нумерации сетки.
    AERO_HOST_DEVICE MeshCell cell(MeshIndex id) const noexcept;

    // Возвращает строку узлов указанной грани.
    AERO_HOST_DEVICE MeshIndexSpan face_nodes(MeshIndex face) const noexcept {
        return mesh_index_row(face_offsets, face_indices, face);
    }
    // Возвращает строку граней указанной ячейки.
    AERO_HOST_DEVICE MeshIndexSpan cell_faces(MeshIndex cell) const noexcept {
        return mesh_index_row(cell_offsets, cell_indices, cell);
    }
    // Возвращает знак ориентации грани относительно ячейки.
    AERO_HOST_DEVICE int face_orientation(MeshIndex cell, MeshIndex local_face) const noexcept {
        return cell_orientation[cell_offsets[cell] + local_face];
    }
    // Возвращает строку рёбер указанной грани.
    AERO_HOST_DEVICE MeshIndexSpan face_edges(MeshIndex face) const noexcept {
        return mesh_index_row(face_edge_offsets, face_edge_indices, face);
    }
    // Возвращает строку узлов указанной ячейки.
    AERO_HOST_DEVICE MeshIndexSpan cell_nodes(MeshIndex cell) const noexcept {
        return mesh_index_row(cell_node_offsets, cell_node_indices, cell);
    }
    // Возвращает строку рёбер указанной ячейки.
    AERO_HOST_DEVICE MeshIndexSpan cell_edges(MeshIndex cell) const noexcept {
        return mesh_index_row(cell_edge_offsets, cell_edge_indices, cell);
    }
};

static_assert(std::is_trivially_copyable<MeshView>::value, "MeshView must be trivially copyable");
static_assert(std::is_standard_layout<MeshView>::value, "MeshView must have standard layout");

// Дескрипторы дают доступ только для чтения. Они копируют представление, а массивы остаются у владельца.
// Можно брать дескриптор из временного mesh.view(), пока существует память сетки или устройства.
// Аргументы обхода — локальные позиции внутри элемента. Методы id() и ..._id() возвращают номера в сетке.
// На CPU и CUDA вложенные обращения предполагают корректные индексы.
class MeshEntity {
public:
    // Возвращает номер сущности в исходном представлении сетки.
    AERO_HOST_DEVICE MeshIndex id() const noexcept { return id_; }

protected:
    // Связывает дескриптор с представлением сетки и номером сущности.
    AERO_HOST_DEVICE MeshEntity(MeshView view, MeshIndex id) : view_(view), id_(id) {}
    // Невладеющее представление сетки для доступа к смежным сущностям.
    MeshView view_;
    // Номер сущности в таблице своего типа.
    MeshIndex id_;
};

// Дескриптор узла, читающий координаты через представление сетки.
class MeshNode : public MeshEntity {
public:
    // Связывает дескриптор с представлением сетки и номером сущности.
    AERO_HOST_DEVICE MeshNode(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    // Координаты в единицах файла сетки.
    AERO_HOST_DEVICE double x() const noexcept { return view_.x[id_]; }
    // Возвращает координату y узла в единицах исходной сетки.
    AERO_HOST_DEVICE double y() const noexcept { return view_.y[id_]; }
    // Возвращает координату z узла в единицах исходной сетки.
    AERO_HOST_DEVICE double z() const noexcept { return view_.z[id_]; }
};

// Дескриптор ребра с доступом к его упорядоченным концам.
class MeshEdge : public MeshEntity {
public:
    // Связывает дескриптор с представлением сетки и номером сущности.
    AERO_HOST_DEVICE MeshEdge(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    // Возвращает число узлов в текущем представлении.
    AERO_HOST_DEVICE MeshIndex node_count() const noexcept { return 2; }
    // Концы упорядочены по номерам узлов. Направление обхода грани или ячейки может быть обратным.
    AERO_HOST_DEVICE MeshIndex node_id(MeshIndex local) const noexcept {
        return local == 0 ? view_.edge_begin[id_] : view_.edge_end[id_];
    }
    // Методы Mesh проверяют номер элемента и выбрасывают MeshError при ошибке.
    // Во вложенных обращениях проверка локальной позиции остаётся обязанностью вызывающего кода.
    AERO_HOST_DEVICE MeshNode node(MeshIndex local) const noexcept { return view_.node(node_id(local)); }
};

// Дескриптор грани с доступом к её узлам и рёбрам.
class MeshFace : public MeshEntity {
public:
    // Связывает дескриптор с представлением сетки и номером сущности.
    AERO_HOST_DEVICE MeshFace(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    // Возвращает диапазон номеров узлов сущности.
    AERO_HOST_DEVICE MeshIndexSpan node_ids() const noexcept { return view_.face_nodes(id_); }
    // Возвращает диапазон номеров рёбер сущности.
    AERO_HOST_DEVICE MeshIndexSpan edge_ids() const noexcept { return view_.face_edges(id_); }
    // Возвращает число узлов в текущем представлении.
    AERO_HOST_DEVICE MeshIndex node_count() const noexcept { return node_ids().size; }
    // Возвращает число рёбер в текущем представлении.
    AERO_HOST_DEVICE MeshIndex edge_count() const noexcept { return edge_ids().size; }
    // Переводит локальную позицию вершины в номер узла сетки.
    AERO_HOST_DEVICE MeshIndex node_id(MeshIndex local) const noexcept { return node_ids()[local]; }
    // Переводит локальную позицию ребра в номер ребра сетки.
    AERO_HOST_DEVICE MeshIndex edge_id(MeshIndex local) const noexcept { return edge_ids()[local]; }
    // Методы Mesh проверяют номер элемента и выбрасывают MeshError при ошибке.
    // Во вложенных обращениях проверка локальной позиции остаётся обязанностью вызывающего кода.
    AERO_HOST_DEVICE MeshNode node(MeshIndex local) const noexcept { return view_.node(node_id(local)); }
    // Возвращает дескриптор ребра в соответствующей нумерации сетки.
    AERO_HOST_DEVICE MeshEdge edge(MeshIndex local) const noexcept { return view_.edge(edge_id(local)); }
};

// Дескриптор ячейки с доступом к граням, узлам и ориентациям.
class MeshCell : public MeshEntity {
public:
    // Связывает дескриптор с представлением сетки и номером сущности.
    AERO_HOST_DEVICE MeshCell(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    // Возвращает диапазон номеров граней сущности.
    AERO_HOST_DEVICE MeshIndexSpan face_ids() const noexcept { return view_.cell_faces(id_); }
    // Узлы и рёбра перечисляются по возрастанию номера без повторов. Геометрический обход имеет другой порядок.
    AERO_HOST_DEVICE MeshIndexSpan node_ids() const noexcept { return view_.cell_nodes(id_); }
    // Возвращает диапазон номеров рёбер сущности.
    AERO_HOST_DEVICE MeshIndexSpan edge_ids() const noexcept { return view_.cell_edges(id_); }
    // Возвращает число граней в текущем представлении.
    AERO_HOST_DEVICE MeshIndex face_count() const noexcept { return face_ids().size; }
    // Возвращает число узлов в текущем представлении.
    AERO_HOST_DEVICE MeshIndex node_count() const noexcept { return node_ids().size; }
    // Возвращает число рёбер в текущем представлении.
    AERO_HOST_DEVICE MeshIndex edge_count() const noexcept { return edge_ids().size; }
    // Переводит локальную позицию грани в номер грани сетки.
    AERO_HOST_DEVICE MeshIndex face_id(MeshIndex local) const noexcept { return face_ids()[local]; }
    // Переводит локальную позицию вершины в номер узла сетки.
    AERO_HOST_DEVICE MeshIndex node_id(MeshIndex local) const noexcept { return node_ids()[local]; }
    // Переводит локальную позицию ребра в номер ребра сетки.
    AERO_HOST_DEVICE MeshIndex edge_id(MeshIndex local) const noexcept { return edge_ids()[local]; }
    // Возвращает знак ориентации грани относительно ячейки.
    AERO_HOST_DEVICE int face_orientation(MeshIndex local) const noexcept {
        return view_.face_orientation(id_, local);
    }
    // Возвращает дескриптор грани в соответствующей нумерации сетки.
    AERO_HOST_DEVICE MeshFace face(MeshIndex local) const noexcept { return view_.face(face_id(local)); }
    // Методы Mesh проверяют номер элемента и выбрасывают MeshError при ошибке.
    // Во вложенных обращениях проверка локальной позиции остаётся обязанностью вызывающего кода.
    AERO_HOST_DEVICE MeshNode node(MeshIndex local) const noexcept { return view_.node(node_id(local)); }
    // Возвращает дескриптор ребра в соответствующей нумерации сетки.
    AERO_HOST_DEVICE MeshEdge edge(MeshIndex local) const noexcept { return view_.edge(edge_id(local)); }
};

// Методы Mesh проверяют номер элемента и выбрасывают MeshError при ошибке.
// Во вложенных обращениях проверка локальной позиции остаётся обязанностью вызывающего кода.
AERO_HOST_DEVICE inline MeshNode MeshView::node(MeshIndex id) const noexcept {
    return MeshNode(*this, id);
}
// Возвращает дескриптор ребра в соответствующей нумерации сетки.
AERO_HOST_DEVICE inline MeshEdge MeshView::edge(MeshIndex id) const noexcept {
    return MeshEdge(*this, id);
}
// Возвращает дескриптор грани в соответствующей нумерации сетки.
AERO_HOST_DEVICE inline MeshFace MeshView::face(MeshIndex id) const noexcept {
    return MeshFace(*this, id);
}
// Возвращает дескриптор ячейки в соответствующей нумерации сетки.
AERO_HOST_DEVICE inline MeshCell MeshView::cell(MeshIndex id) const noexcept {
    return MeshCell(*this, id);
}

static_assert(std::is_trivially_copyable<MeshCell>::value, "Entity handles must be trivially copyable");

// Ограничения объёма входных данных и длины имён при чтении сетки.
struct MeshReadLimits {
    // Память входных массивов и строк. Рабочая память проверок и производные связи учитываются отдельно.
    std::uint64_t max_payload_bytes, max_name_bytes;
    // Задаёт предел входных данных и максимальную длину имени участка.
    MeshReadLimits(std::uint64_t bytes = 1024ULL * 1024ULL * 1024ULL, std::uint64_t name_bytes = 4096)
        : max_payload_bytes(bytes), max_name_bytes(name_bytes) {}
};

// Формат сетки версии 3. При чтении сигнатура различает текст, двоичные данные и двоичные блоки AEROLZMA.
// Версия 1 отклоняется. Контейнер LZMA описан в source/io/README.txt.
//
// Текст начинается с «AEROMESH 3 TEXT». Далее идут DIMENSION, COUNTS, X, Y, Z,
// FACE_OFFSETS, FACE_NODES, CELL_OFFSETS, CELL_FACES, CELL_ORIENTATIONS, BOUNDARY_FACES, BOUNDARY_PATCH, PATCH_NAMES, END.
// COUNTS перечисляет узлы, грани, записи «грань–узел», ячейки, записи «ячейка–грань», внешние грани и участки.
// Длина массива смещений равна числу строк плюс один. Номера с нуля, размерность 1–3, массивов координат всегда три.
// Чтение файлов v2 сохраняется во всех размерностях. Токены разделяют пробелами, дробную часть — точкой.
// Имена заключают в двойные кавычки. Обратная косая черта экранирует кавычку или другую обратную косую черту.
//
// Двоичный поток начинается с восьми байтов AEROMSHB и uint64 со значением 3. Далее те же данные без меток и END.
// Целые числа записываются как uint64 младшим байтом вперёд. Для ориентации один байт: 0x01 (+1) или 0xff (−1).
// Вещественные числа — IEEE-754 binary64 младшим байтом вперёд. Имя содержит длину uint64 и байты UTF-8 без завершающего нуля.
// Имена непустые и уникальные, управляющие символы ASCII запрещены. Лишние данные в конце файла вызывают ошибку.
// Производные рёбра и списки узлов и рёбер ячеек восстанавливаются при чтении.
//
// Ячейка 1D имеет две точечные грани с противоположными знаками. В 2D нужен один простой ориентированный контур
// хотя бы из трёх рёбер, он может быть вогнутым. В 3D требуется связная замкнутая ориентированная поверхность.
// У грани одна или две ячейки. Две соседние ячейки используют противоположные знаки. Каждая внешняя грань входит в список один раз.
// Число вершин многоугольника и граней многогранника произвольно. Повторы граней и ячеек, повтор вершины,
// грани без ячеек и неполная граница отклоняются. На этом этапе проверяется конечность координат.
// Плоскостность, длины, площади, объёмы, пересечения и наложения одной проверкой топологии установить нельзя.
class Mesh {
public:
    // Создаёт хранилище сетки; при переданных данных проверяет их и строит производные связи.
    Mesh() : face_edge_offsets_(1, 0) {}
    // Создаёт хранилище сетки; при переданных данных проверяет их и строит производные связи.
    explicit Mesh(MeshData data);
    // Возвращает исходные координаты, топологию и метки сетки.
    const MeshData& data() const noexcept { return data_; }
    // Возвращает номера начальных узлов ориентированных рёбер.
    const std::vector<MeshIndex>& edge_begin() const noexcept { return edge_begin_; }
    // Возвращает номера конечных узлов ориентированных рёбер.
    const std::vector<MeshIndex>& edge_end() const noexcept { return edge_end_; }
    // Указатели на память CPU действительны, пока существует исходное хранилище и его не заменили.
    MeshView view() const noexcept;
    // Методы Mesh проверяют номер элемента и выбрасывают MeshError при ошибке.
    // Во вложенных обращениях проверка локальной позиции остаётся обязанностью вызывающего кода.
    MeshNode node(MeshIndex id) const;
    // Возвращает дескриптор ребра в соответствующей нумерации сетки.
    MeshEdge edge(MeshIndex id) const;
    // Возвращает дескриптор грани в соответствующей нумерации сетки.
    MeshFace face(MeshIndex id) const;
    // Возвращает дескриптор ячейки в соответствующей нумерации сетки.
    MeshCell cell(MeshIndex id) const;
    // Проверяет геометрию, индексы, связность и граничные метки сетки.
    void validate() const;

    // Файловый ввод и вывод выполняются на CPU. Путь передаётся в узкой кодировке системы.
    static Mesh read(const std::string& path, const MeshReadLimits& limits = MeshReadLimits());
    // Файловый ввод и вывод выполняются на CPU. Путь передаётся в узкой кодировке системы.
    static Mesh read(std::istream& input, const MeshReadLimits& limits = MeshReadLimits());
    // При ошибке записи часть выходного файла уже может находиться на диске.
    void write(const std::string& path, MeshFormat format = MeshFormat::Binary) const;
    // При ошибке записи часть выходного файла уже может находиться на диске.
    void write(std::ostream& output, MeshFormat format = MeshFormat::Binary) const;
    // Подаёт канонические байты в SHA-256 полей и recovery, включая старые файлы.
    void write_fingerprint_payload(std::ostream& output) const;

private:
    // Нумерует рёбра по возрастанию пары номеров их концов.
    void build_edges();
    // Строит рёбра одномерной сетки из концов её ячеек.
    void build_line_edges(); // В 1D концы определяются двумя точечными гранями ячейки.
    // Делит рёбра по меньшему узлу и возвращает начала этих групп в edge_end_.
    std::vector<MeshIndex> group_edges_by_node();
    // Записывает для каждой грани номера уже построенных рёбер её контура.
    void number_face_edges(const std::vector<MeshIndex>& node_offsets);
    // Собирает уникальные узлы и рёбра одной ячейки по контурам её граней.
    void collect_cell_connectivity(MeshIndex cell, std::vector<MeshIndex>& nodes, std::vector<MeshIndex>& edges) const;
    // Строит производные CSR-связи ячеек с узлами и рёбрами.
    void build_cell_connectivity();
    // Исходные массивы координат и связности.
    MeshData data_;
    // Только для совпадения SHA-256 со снимками, ранее записанными на исходной сетке v2.
    std::uint64_t fingerprint_word_ = mesh_format::default_fingerprint_word;
    // Концы рёбер и производная таблица соответствия граней рёбрам.
    std::vector<MeshIndex> edge_begin_, edge_end_, face_edge_offsets_, face_edge_indices_;
    // Производные списки узлов и рёбер каждой ячейки.
    MeshCsr cell_nodes_, cell_edges_;
};

} // namespace aero

#endif
