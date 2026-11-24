#ifndef AERO_INDEXED_IO_H
#define AERO_INDEXED_IO_H
#include "../parallel/processes.h"
#include <functional>

namespace aero {
// Одна строка файла соответствует исходному узлу; периодическая копия читает состояние своего представителя.
struct OwnedNode {
    std::uint64_t id, representative,
        slot; // Исходный номер узла, его периодический представитель и смещение узла в локальном поле.
};
namespace indexed_io {
enum class Access { Read, Write };
// Поле j текущего узла начинается по адресу data + j*pitch. Все поля узла обрабатываем за один вызов.
using Row = std::function<void(const OwnedNode&, char* data, std::size_t pitch)>;

// Перед вызовом проверяют размер файла и сортируют узлы по исходному номеру.
void transfer(const Processes& world, const std::string& file, Access access, unsigned stride,
              const std::vector<OwnedNode>& nodes, const std::vector<std::uint64_t>& offsets, const Row& row);
// Записывает двоичное представление вещественного числа без потери точности.
void put_real(char* dst, double value);
}
}
#endif
