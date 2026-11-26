#ifndef AERO_IO_UTF8_LABEL_H
#define AERO_IO_UTF8_LABEL_H
#include <cstdint>
#include <string>
#include <cctype>

namespace aero {
// Приводит байты ключевого слова к нижнему регистру. Передача по значению
// позволяет переиспользовать буфер временной строки без второго выделения.
inline std::string lower_keyword(std::string value) {
    for (char& c : value)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

namespace io {
// Имена границ, полей и единицы измерения в файлах используют печатный UTF-8.
// Пустая строка допустима для единиц измерения. Для обязательного имени это проверяет вызывающая сторона.
inline bool valid_label_text(const std::string& text) {
    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char first = static_cast<unsigned char>(text[i++]);
        if (first < 0x80) {
            if (first < 0x20 || first == 0x7f)
                return false;
            continue;
        }
        // Старший байт определяет число продолжений и минимальный код для такой длины.
        unsigned tail;
        std::uint32_t code, minimum;
        if (first >= 0xc2 && first <= 0xdf) {
            tail = 1;
            code = first & 31;
            minimum = 0x80;
        } else if (first >= 0xe0 && first <= 0xef) {
            tail = 2;
            code = first & 15;
            minimum = 0x800;
        } else if (first >= 0xf0 && first <= 0xf4) {
            tail = 3;
            code = first & 7;
            minimum = 0x10000;
        } else
            return false;
        if (tail > text.size() - i)
            return false;
        for (unsigned k = 0; k < tail; ++k) {
            const unsigned char byte = static_cast<unsigned char>(text[i++]);
            if ((byte & 0xc0) != 0x80)
                return false;
            code = (code << 6) | (byte & 63);
        }
        // Исключаем избыточные кодировки, суррогаты UTF-16 и символы за пределами Unicode.
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
            return false;
    }
    return true;
}
}
}
#endif
