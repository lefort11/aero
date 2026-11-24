#ifndef AERO_MESH_READER_H
#define AERO_MESH_READER_H
#include "mesh.h"
#include "../io/lzma_stream.h"
#include "../io/native_format.h"
#include "../io/utf8_label.h"
#include <cstring>
#include <istream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>

namespace aero {
namespace mesh_input {
// Сообщает ошибку чтения или несогласованности сетки.
[[noreturn]] inline void fail(const std::string& message) {
    throw MeshError(message);
}
// Метка границы обязательна. Допустимые символы те же, что у имён полей снимка.
inline bool valid_name(const std::string& name) {
    return !name.empty() && io::valid_label_text(name);
}

// Проверяет пробельный разделитель текстового формата сетки.
inline bool whitespace(int c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
}

// Ведёт общий остаток байтов для выделений при чтении одного файла сетки.
class Budget {
public:
    // Начинает учёт разрешённого объёма памяти чтения.
    explicit Budget(std::uint64_t bytes) : remaining_(bytes) {}
    // Уменьшает остаток разрешённой памяти и отклоняет превышение лимита.
    void consume(std::uint64_t count, std::uint64_t width = 1) {
        // Делим до умножения, чтобы счётчик проверял и переполнение размера.
        if (width == 0 || count > remaining_ / width)
            fail("Mesh exceeds the configured payload limit");
        remaining_ -= count * width;
    }

private:
    // Число байтов, которые ещё можно занять в массивах и строках.
    std::uint64_t remaining_;
};

// Приводит текстовый, двоичный и сжатый поток к одним операциям чтения полей сетки.
class Reader {
    friend struct io::NativeFileCodec;
public:
    // Подключает входной поток и ограничения безопасного чтения сетки.
    Reader(std::istream& input, const MeshReadLimits& limits)
        : input_(&input), limits_(limits), budget_(limits.max_payload_bytes) {
        char magic[8];
        bytes(magic, sizeof magic);
        if (std::memcmp(magic, "AEROLZMA", 8) == 0) {
            const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
            // Двоичный заголовок занимает 88 байт. Остальные прочитанные байты учитывает Budget.
            const std::uint64_t limit =
                limits.max_payload_bytes > maximum - 88 ? maximum : limits.max_payload_bytes + 88;
            // Распаковка идёт по мере чтения. Последующие операции используют уже внутренний поток.
            compressed_.reset(new io::LzmaInputBuffer(input, limit));
            decoded_.reset(new std::istream(compressed_.get()));
            decoded_->exceptions(std::ios::badbit);
            input_ = decoded_.get();
            bytes(magic, sizeof magic);
            if (std::memcmp(magic, "AEROMSHB", 8) != 0)
                fail("LZMA mesh must contain native binary mesh data");
        }
        if (std::memcmp(magic, "AEROMESH", 8) == 0)
            text_ = true;
        else if (std::memcmp(magic, "AEROMSHB", 8) == 0)
            text_ = false;
        else
            fail("Unrecognized mesh signature");
        version_ = u64();
        if (version_ != mesh_format::legacy_version && version_ != mesh_format::current_version)
            fail("Unsupported mesh format version; expected 2 or 3");
        if (text_)
            block("TEXT");
    }

    // Передаёт блок байтов формата сетки с проверкой состояния потока.
    void block(const char* expected) {
        if (text_ && token() != expected)
            fail(std::string("Expected block: ") + expected);
    }

    // В v2 между COUNTS и X было ещё одно число. Сохраняем его биты для
    // проверки старых файлов полей. В v3 сразу за счётчиками идут координаты.
    std::uint64_t read_fingerprint_word() {
        if (text_) {
            // В тексте можно удалить устаревшую строку вручную. Если следующим
            // идёт X, оставляем его в потоке для обычного чтения координат.
            skip_space();
            if (input_->peek() != 'L')
                return mesh_format::default_fingerprint_word;
        } else if (version_ == mesh_format::current_version) {
            // В бинарном потоке нет названий секций. Их расположение задаёт версия.
            return mesh_format::default_fingerprint_word;
        }
        block("LENGTH_SCALE_TO_METERS");
        return number_codec::real_bits(real());
    }

    // Читает или записывает беззнаковое 64-битное значение формата сетки.
    std::uint64_t u64() {
        return io::NativeFileCodec::read_unsigned(*this,fail);
    }

    // Знак определяет направление обхода грани в ячейке. В двоичном файле −1 записан байтом 255.
    std::int8_t orientation() {
        if (text_) {
            const std::string value = token();
            if (value == "1" || value == "+1")
                return 1;
            if (value == "-1")
                return -1;
        } else {
            char value;
            bytes(&value, 1);
            const unsigned char byte = static_cast<unsigned char>(value);
            if (byte == 1)
                return 1;
            if (byte == 255)
                return -1;
        }
        fail("Cell face orientation must be +1 or -1");
    }

    // Читает или записывает координату в вещественном представлении формата.
    double real() {
        if (!text_) {
            return number_codec::real_from_bits(u64());
        }
        std::istringstream number(token());
        number.imbue(std::locale::classic());
        double result = 0;
        if (!(number >> result) || number.peek() != std::char_traits<char>::eof())
            fail("Invalid floating-point value");
        return result;
    }

    // В native v2 числа занимают восемь байтов. На little-endian машине
    // массив уже имеет файловое представление: читаем его одним блоком.
    template<class T, class ReadOne>
    void array(const char* name, std::vector<T>& values, ReadOne one) {
        block(name);
        if (!text_ && (sizeof(T) == 1 || number_codec::native_little_endian)) {
            bytes(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
        } else {
            for (auto& value : values) value = (this->*one)();
        }
    }

    // Проверяем предел контейнера и общий бюджет до выделения памяти из размеров файла.
    template <class T> void resize(std::vector<T>& values, std::uint64_t count) {
        if (count > values.max_size())
            fail("Array is too large for this platform");
        budget_.consume(count, sizeof(T));
        values.resize(static_cast<std::size_t>(count));
    }

    // В двоичном файле длина задана явно. В тексте имя заключено в кавычки и может содержать пробелы.
    std::string name() {
        std::string result;
        if (!text_) {
            const auto length = u64();
            if (length > limits_.max_name_bytes || length > result.max_size())
                fail("Boundary name is too long");
            budget_.consume(length);
            result.resize(static_cast<std::size_t>(length));
            if (!result.empty())
                bytes(&result[0], result.size());
            return result;
        }
        skip_space();
        if (input_->get() != '"')
            fail("Expected a quoted boundary name");
        while (true) {
            int c = input_->get();
            if (c == std::char_traits<char>::eof())
                fail("Truncated boundary name");
            if (c == '"')
                break;
            if (c == '\\') {
                c = input_->get();
                if (c != '\\' && c != '"')
                    fail("Invalid escape in boundary name");
            }
            if (result.size() >= limits_.max_name_bytes)
                fail("Boundary name is too long");
            budget_.consume(1);
            result.push_back(static_cast<char>(c));
        }
        if (!whitespace(input_->peek()) && input_->peek() != std::char_traits<char>::eof())
            fail("Boundary names must be separated by whitespace");
        return result;
    }

    // Проверяет корректное завершение обработки сеточного файла.
    void finish() {
        block("END");
        if (text_)
            skip_space();
        if (input_->peek() != std::char_traits<char>::eof())
            fail("Trailing data after the mesh");
        if (input_->bad())
            fail("Mesh input failed");
    }

private:
    // Читает точный блок байтов с проверкой преждевременного конца файла.
    void bytes(char* destination, std::size_t size) {
        if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            fail("Input block is too large");
        if (size != 0 && !input_->read(destination, static_cast<std::streamsize>(size)))
            fail("Truncated mesh input");
    }
    // Пропускает пробельные разделители текстового формата.
    void skip_space() {
        while (whitespace(input_->peek()))
            input_->get();
    }
    // Читает следующий токен текстового сеточного файла.
    std::string token() {
        skip_space();
        std::string result;
        while (true) {
            const int c = input_->peek();
            if (c == std::char_traits<char>::eof() || whitespace(c))
                break;
            if (result.size() == 128)
                fail("Mesh token is too long");
            result.push_back(static_cast<char>(input_->get()));
        }
        if (result.empty())
            fail("Unexpected end of mesh input");
        return result;
    }
    // Распаковщик и поток над ним живут до конца чтения сжатой сетки.
    std::unique_ptr<io::LzmaInputBuffer> compressed_;
    std::unique_ptr<std::istream> decoded_;
    // Текущий источник байтов — исходный поток или распакованные данные.
    std::istream* input_;
    MeshReadLimits limits_;
    Budget budget_;
    bool text_ = false;
    std::uint64_t version_; // Определяет расположение координат в потоке.
};

}
} // namespace aero::mesh_input
#endif
