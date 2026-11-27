#pragma once
#include "../fields.h"
#include <algorithm>
#include <array>
#include <fstream>

namespace aero {
// Читает узловые столбцы уже заполненного MPI-файла. Заголовок и смещения
// принадлежат вызывающему коду и соответствуют тому же вызову write_binary_layout.
class BinaryColumns {
    std::ifstream input_;                      // Общий источник для обоих представлений.
    const Field* first_;                       // Начало таблицы полей для поиска смещения по ссылке.
    const std::vector<std::uint64_t>& offsets_; // Байтовые позиции первых значений узловых полей.
    std::array<char, 65536> buffer_;            // Один блок на весь экспорт независимо от размера снимка.
public:
    // Открывает временный файл после завершения коллективной записи всех владельцев.
    BinaryColumns(const std::string& path, const FieldSnapshot& snapshot, const std::vector<std::uint64_t>& offsets)
        : input_(path, std::ios::binary), first_(snapshot.fields.data()), offsets_(offsets) {
        if (!input_) throw FieldError("Cannot open MPI snapshot for visualization: " + path);
    }
    // Передаёт числовые блоки потребителю в исходном little-endian порядке файла полей.
    template<class Consumer> void visit(const Field& field, std::uint64_t count, const Consumer& consume) {
        const auto index = static_cast<std::size_t>(&field - first_);
        input_.seekg(static_cast<std::streamoff>(offsets_.at(index)));
        while (count) {
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(count, buffer_.size() / 8));
            if (!input_.read(buffer_.data(), static_cast<std::streamsize>(8 * n)))
                throw FieldError("Incomplete MPI field column during visualization");
            consume(buffer_.data(), n);
            count -= n;
        }
    }
};
}
