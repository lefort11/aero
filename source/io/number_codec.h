#ifndef AERO_IO_NUMBER_CODEC_H
#define AERO_IO_NUMBER_CODEC_H

#include <cstdint>
#include <cstring>
#include <limits>
#include <ostream>
#include <string>

namespace aero {
namespace number_codec {
using Word = std::uint64_t;
static_assert(sizeof(double) == sizeof(Word) && std::numeric_limits<double>::is_iec559,
              "Requires IEEE-754 binary64");

// Сетка, поля и контрольные точки используют один формат: восемь байтов, младший первым.
// memcpy допускает невыровненный адрес. На обычном little-endian CPU это одна загрузка или запись.
#if defined(_WIN32) || (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) &&                       \
                        __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
constexpr bool native_little_endian = true;
#else
constexpr bool native_little_endian = false;
#endif

inline Word read_word(const char* bytes) {
    Word value = 0;
    if (native_little_endian) {
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    for (unsigned i = 0; i < sizeof(value); ++i)
        value |= Word(static_cast<unsigned char>(bytes[i])) << (8 * i);
    return value;
}

inline void store_word(char* bytes, Word value) {
    if (native_little_endian) {
        std::memcpy(bytes, &value, sizeof(value));
        return;
    }
    for (unsigned i = 0; i < sizeof(value); ++i)
        bytes[i] = static_cast<char>(value >> (8 * i));
}

// Копируем битовое представление double, сохраняя в том числе знак нуля.
inline Word real_bits(double value) {
    Word bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
inline double real_from_bits(Word bits) {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
inline double read_real(const char* bytes) {
    return real_from_bits(read_word(bytes));
}
inline void store_real(char* bytes, double value) {
    store_word(bytes, real_bits(value));
}

// Пишет ровно восемь байтов либо десятичное число с пробелом, независимо от локали потока.
inline void write_word(std::ostream& output, Word value, bool text) {
    if (!text) {
        char bytes[sizeof(value)];
        store_word(bytes, value);
        output.write(bytes, sizeof(bytes));
        return;
    }
    char digits[20]; // Наибольшее uint64 содержит ровно 20 десятичных цифр.
    char* end = digits + sizeof(digits);
    char* first = end;
    do {
        *--first = static_cast<char>('0' + value % 10);
        value /= 10;
    } while (value);
    output.write(first, static_cast<std::streamsize>(end - first));
    output.put(' ');
}

// Разбирает уже выделенный непустой токен. Вызывающий читатель задаёт свой тип ошибки.
template <class Fail> Word parse_decimal(const std::string& token, Fail fail) {
    Word value = 0;
    for (char c : token) {
        if (c < '0' || c > '9')
            fail("Invalid unsigned integer");
        const unsigned digit = static_cast<unsigned>(c - '0');
        // Проверяем до умножения: переполненное беззнаковое число уже неотличимо от малого.
        if (value > (std::numeric_limits<Word>::max() - digit) / 10)
            fail("Unsigned integer overflow");
        value = value * 10 + digit;
    }
    return value;
}
}
}
#endif
