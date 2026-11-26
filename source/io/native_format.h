#pragma once
#include "number_codec.h"

namespace aero {
namespace io {
// Сетка и поля используют одинаковые целые числа и строки, но по-разному
// ограничивают ввод. Чтение байтов и токенов поручаем читателю конкретного файла.
struct NativeFileCodec {
    // Записывает готовую текстовую метку без преобразования локалью.
    static void write_literal(std::ostream& output, const char* label) {
        output.write(label, static_cast<std::streamsize>(std::strlen(label)));
    }
    // Двоичный формат не содержит текстовых меток. Для текста сетка и поля
    // передают свой завершающий разделитель, чтобы сохранить формат обоих файлов.
    static void write_block(std::ostream& output, bool text, const char* label, char separator) {
        if (text) {
            output.put('\n');
            write_literal(output, label);
            output.put(separator);
        }
    }
    // Проверяет метку конца, допустимый хвост текстового файла и состояние потока.
    // Читатель формата пропускает пробелы по собственным правилам и задаёт класс ошибки.
    template<class Reader, class Fail>
    static void finish(Reader& reader, Fail fail, const char* trailing, const char* failed) {
        reader.block("END");
        if (reader.text_) reader.skip_space();
        if (reader.input_->peek() != std::char_traits<char>::eof()) fail(trailing);
        if (reader.input_->bad()) fail(failed);
    }

    // Проверяет адресуемую длину и списывает байты до выделения памяти.
    // Каждый формат сохраняет свой класс ошибки и свой счётчик бюджета.
    template<class Array, class Charge, class Fail>
    static void resize_array(Array& values, number_codec::Word size, Charge charge, Fail fail, const char* message) {
        if (size > values.max_size()) fail(message);
        charge(size, sizeof(typename Array::value_type));
        values.resize(static_cast<std::size_t>(size));
    }

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
