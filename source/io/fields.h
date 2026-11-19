#ifndef AERO_IO_FIELDS_H
#define AERO_IO_FIELDS_H

#include <array>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace aero {
class Mesh;
enum class FieldFormat { Text, Binary, Lzma };
enum class FieldLocation : std::uint8_t { Node = 0, Edge = 1, Face = 2, Cell = 3 };

class FieldError : public std::runtime_error {
public:
    explicit FieldError(const std::string& message) : std::runtime_error(message) {}
};

struct Field {
    // Имя физической величины в файле полей.
    std::string name;
    // Тип элемента сетки, на котором хранятся значения.
    FieldLocation location;
    // Имена столбцов в том же порядке, в котором идут данные каждой компоненты.
    std::vector<std::string> components;
    // Единицы измерения для подписи при экспорте.
    std::string units;
    // Значения идут по компонентам. Индекс равен component * entity_count + entity_id.
    // В каждом столбце находятся все элементы данного типа в нумерации сетки с нуля.
    std::vector<double> values;
    Field() : location(FieldLocation::Cell) {}
};

// Запрос подмножества узлов из снимка той же глобальной сетки для локального расчёта.
struct FieldNodeSelection {
    const std::vector<std::uint64_t>& nodes; // Исходные номера узлов строго возрастают.
    // Идентификатор исходной сетки для проверки до загрузки локальных узлов.
    const std::string& mesh_sha256;
    unsigned dimension;
    std::array<std::uint64_t, 4> counts;
};

struct FieldReadLimits {
    // Предел памяти для сохраняемых векторов и строк. Служебные данные аллокатора и кодека учитываются отдельно.
    std::uint64_t max_payload_bytes;
    // Максимальный размер распакованного файла вместе с метаданными и пробелами текстового представления.
    std::uint64_t max_decoded_bytes;
    std::uint64_t max_string_bytes, max_fields, max_components;
    FieldReadLimits()
        : max_payload_bytes(1024ULL * 1024 * 1024), max_decoded_bytes(2ULL * 1024 * 1024 * 1024),
          max_string_bytes(4096), max_fields(1024), max_components(1024) {}
};

// Снимок содержит поля и ссылку на сетку по хешу. Формат определяем по сигнатуре файла.
// Описание двоичного представления AEROFIELDS v1 приведено в FIELDS.txt.
class FieldSnapshot {
public:
    unsigned dimension;
    // Число узлов, рёбер, граней и ячеек именно в этом порядке.
    std::array<std::uint64_t, 4> counts;
    // SHA-256 двоичной записи сетки v2 в виде строчных шестнадцатеричных цифр.
    std::string mesh_sha256;
    // Номер полностью принятого шага снимка.
    std::uint64_t step;
    // Физическое время состояния на этом шаге.
    double time;
    // Поля снимка в порядке записи в файл.
    std::vector<Field> fields;

    // Пустой снимок нужно связать с сеткой через метаданные или использовать конструктор от Mesh.
    FieldSnapshot() : dimension(0), counts{{0, 0, 0, 0}}, step(0), time(0) {}
    explicit FieldSnapshot(const Mesh& mesh, std::uint64_t step_number = 0, double time_seconds = 0);
    std::uint64_t entity_count(FieldLocation location) const;
    void validate() const;
    // Сверяет размерность, числа элементов и SHA-256 сетки перед использованием значений.
    void check_mesh(const Mesh& mesh) const;
    static std::string mesh_fingerprint(const Mesh& mesh);

    // Читает значения узловых полей в порядке запрошенных исходных номеров.
    static std::vector<Field> read_nodes(const std::string& path, const FieldNodeSelection& selection);
    static FieldSnapshot read(const std::string& path, const FieldReadLimits& limits = FieldReadLimits());
    static FieldSnapshot read(std::istream& input, const FieldReadLimits& limits = FieldReadLimits());
    // Проверяет снимок до открытия файла. Ошибка самой записи может оставить файл частично заполненным.
    void write(const std::string& path, FieldFormat format = FieldFormat::Binary) const;
    void write(std::ostream& output, FieldFormat format = FieldFormat::Binary) const;
    // Записывает описания полей и резервирует диапазоны для двоичных данных. Массивы значений должны быть пусты.
    std::vector<std::uint64_t> write_binary_layout(const std::string& path) const;

private:
    void validate_fields(bool layout_only) const;
};
} // namespace aero
#endif
