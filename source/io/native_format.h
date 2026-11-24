#pragma once
#include "number_codec.h"

namespace aero {
namespace io {
// Сетка и поля используют одинаковые целые числа и строки, но по-разному
// ограничивают ввод. Чтение байтов и токенов поручаем читателю конкретного файла.
struct NativeFileCodec {
    // Читает беззнаковое число формата и проверяет диапазон значения.
    template<class Reader, class Fail>
    static number_codec::Word read_unsigned(Reader& reader, Fail fail) {
        if (reader.text_) return number_codec::parse_decimal(reader.token(),fail);
        char bytes[sizeof(number_codec::Word)];
        reader.bytes(bytes,sizeof(bytes));
        return number_codec::read_word(bytes);
    }

    // В тексте строку заключаем в кавычки и ставим разделитель своего формата.
    // В двоичном файле перед байтами строки записывается только её длина.
    template<class Fail>
    static void write_string(std::ostream& output, bool text, const std::string& value,
                             char separator, Fail fail, const char* size_error) {
        if (text) {
            output.put('"');
            for (char c : value) {
                if (c=='"' || c=='\\') output.put('\\');
                output.put(c);
            }
            output.put('"');
            output.put(separator);
            return;
        }
        if (value.size()>static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            fail(size_error);
        number_codec::write_word(output,value.size(),false);
        output.write(value.data(),static_cast<std::streamsize>(value.size()));
    }
};
}
}
