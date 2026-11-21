#include "fields.h"
#include "utf8_label.h"
#include "../parallel/parallel_for.h"
#include "native_format.h"
#include <algorithm>
#include "lzma_stream.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <utility>

namespace aero {
namespace {
typedef std::uint64_t U64;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "Requires IEEE-754 binary64");
[[noreturn]] void fail(const std::string& message) {
    throw FieldError(message);
}
bool space(int c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}
const char* location_name(FieldLocation location) {
    switch (location) {
    case FieldLocation::Node:
        return "NODE";
    case FieldLocation::Edge:
        return "EDGE";
    case FieldLocation::Face:
        return "FACE";
    case FieldLocation::Cell:
        return "CELL";
    }
    fail("Invalid field location");
}
U64 product(U64 a, U64 b) {
    if (b && a > std::numeric_limits<U64>::max() / b)
        fail("Field value count overflow");
    return a * b;
}
bool valid_hash(const std::string& hash) {
    if (hash.size() != 64)
        return false;
    for (char c : hash)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}
unsigned hex_value(char c) {
    return static_cast<unsigned>(c <= '9' ? c - '0' : c - 'a' + 10);
}
void check_format(FieldFormat format) {
    if (format != FieldFormat::Text && format != FieldFormat::Binary && format != FieldFormat::Lzma)
        fail("Unsupported field encoding");
}

class Reader {
    friend struct io::NativeFileCodec;
public:
    Reader(std::istream& input, const FieldReadLimits& limits)
        : input_(&input), limits_(limits), budget_(limits.max_payload_bytes),
          available_(limits.max_decoded_bytes), text_(false) {
        char magic[8];
        bytes(magic, 8);
        if (std::memcmp(magic, "AEROLZMA", 8) == 0) {
            compressed_.reset(new io::LzmaInputBuffer(input, limits.max_decoded_bytes));
            decoded_.reset(new std::istream(compressed_.get()));
            decoded_->exceptions(std::ios::badbit);
            input_ = decoded_.get();
            available_ = limits.max_decoded_bytes;
            bytes(magic, 8);
            if (std::memcmp(magic, "AEROFLDB", 8) != 0)
                fail("LZMA fields must contain binary AEROFIELDS v1");
        }
        if (std::memcmp(magic, "AEROFIEL", 8) == 0) {
            char suffix[2];
            bytes(suffix, 2);
            if (std::memcmp(suffix, "DS", 2) != 0 || !space(peek()))
                fail("Unrecognized field signature");
            text_ = true;
        } else if (std::memcmp(magic, "AEROFLDB", 8) != 0)
            fail("Unrecognized field signature");
        if (u64() != 1)
            fail("Unsupported field format version");
        block("TEXT");
    }
    void block(const char* label) {
        if (text_ && token() != label)
            fail(std::string("Expected block: ") + label);
    }
    U64 u64() {
        return io::NativeFileCodec::read_unsigned(*this,fail);
    }
    double real() {
        if (!text_) {
            return number_codec::real_from_bits(u64());
        }
        const std::string value = token();
        if (value == "nan")
            return std::numeric_limits<double>::quiet_NaN();
        if (value == "inf")
            return std::numeric_limits<double>::infinity();
        if (value == "-inf")
            return -std::numeric_limits<double>::infinity();
        std::istringstream number(value);
        number.imbue(std::locale::classic());
        double result;
        if (!(number >> result) || number.peek() != std::char_traits<char>::eof() || !std::isfinite(result))
            fail("Invalid floating-point value");
        return result;
    }
    std::string string() {
        std::string result;
        if (!text_) {
            const U64 length = u64();
            if (length > limits_.max_string_bytes || length > result.max_size())
                fail("Field string exceeds limit");
            charge(length, 1);
            result.resize(static_cast<std::size_t>(length));
            if (length)
                bytes(&result[0], result.size());
        } else {
            skip_space();
            if (get() != '"')
                fail("Expected quoted field string");
            for (;;) {
                int c = get();
                if (c == std::char_traits<char>::eof())
                    fail("Truncated field string");
                if (c == '"')
                    break;
                if (c == '\\') {
                    c = get();
                    if (c != '\\' && c != '"')
                        fail("Invalid field string escape");
                }
                if (result.size() >= limits_.max_string_bytes)
                    fail("Field string exceeds limit");
                charge(1, 1);
                result += static_cast<char>(c);
            }
            if (!space(peek()) && peek() != std::char_traits<char>::eof())
                fail("Missing separator after string");
        }
        if (!io::valid_label_text(result))
            fail("Invalid UTF-8 or control character in field string");
        return result;
    }
    std::string hash() {
        if (text_)
            return string();
        unsigned char data[32];
        bytes(reinterpret_cast<char*>(data), 32);
        const char* digits = "0123456789abcdef";
        std::string result;
        for (unsigned char c : data) {
            result += digits[c >> 4];
            result += digits[c & 15];
        }
        return result;
    }
    FieldLocation location() {
        if (!text_) {
            char value;
            bytes(&value, 1);
            const FieldLocation result = static_cast<FieldLocation>(static_cast<unsigned char>(value));
            location_name(result);
            return result;
        }
        const std::string value = token();
        for (FieldLocation loc :
             {FieldLocation::Node, FieldLocation::Edge, FieldLocation::Face, FieldLocation::Cell})
            if (value == location_name(loc))
                return loc;
        fail("Invalid field location");
    }
    template <class T> void resize(std::vector<T>& array, U64 size) {
        if (size > array.max_size())
            fail("Field array is too large");
        charge(size, sizeof(T));
        array.resize(static_cast<std::size_t>(size));
    }
    // При совпадении формата файла с представлением double читаем непрерывный диапазон одним блоком.
    // bytes учитывает лимит распаковки и обнаруживает усечённый файл также на этом быстром пути.
    void read_values(double* destination, std::size_t count) {
        if (!text_ && number_codec::native_little_endian && count) {
            bytes(reinterpret_cast<char*>(destination), count * sizeof(double));
            return;
        }
        for (std::size_t i = 0; i < count; ++i)
            destination[i] = real();
    }

    void skip_values(U64 count) {
        if (!count)
            return;
        if (text_) {
            for (U64 i = 0; i < count; ++i)
                real();
            return;
        }
        U64 size = product(count, 8);
        if (size > available_)
            fail("Fields exceed decoded byte limit");
        // Небольшой зазор обычно уже в буфере потока. Последовательное чтение сохраняет его,
        // а для длинного пропуска в обычном файле быстрее сразу перейти к концу диапазона.
        constexpr std::size_t discard_bytes = 8192;
        if (!compressed_ && size > discard_bytes) {
            // Читаем также последний байт: успешное перемещение указателя само по себе не доказывает, что данные существуют.
            if (size - 1 > static_cast<U64>(std::numeric_limits<std::streamoff>::max()))
                fail("Field offset too large");
            input_->seekg(static_cast<std::streamoff>(size - 1), std::ios::cur);
            available_ -= size - 1;
            char last;
            bytes(&last, 1);
        } else {
            char block[discard_bytes];
            while (size) {
                const auto n = static_cast<std::size_t>(std::min<U64>(size, sizeof block));
                bytes(block, n);
                size -= n;
            }
        }
    }
    void finish() {
        block("END");
        if (text_)
            skip_space();
        if (peek() != std::char_traits<char>::eof())
            fail("Trailing data after fields");
        if (input_->bad())
            fail("Field input failed");
    }

private:
    void charge(U64 size, U64 width) {
        if (size > budget_ / width)
            fail("Fields exceed payload limit");
        budget_ -= size * width;
    }
    int peek() { return input_->peek(); }
    int get() {
        if (!available_) {
            if (peek() == std::char_traits<char>::eof())
                return peek();
            fail("Fields exceed decoded byte limit");
        }
        const int c = input_->get();
        if (c != std::char_traits<char>::eof())
            --available_;
        return c;
    }
    void bytes(char* data, std::size_t size) {
        if (size > available_)
            fail("Fields exceed decoded byte limit");
        if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            fail("Field block too large");
        if (!input_->read(data, static_cast<std::streamsize>(size)))
            fail("Truncated field input");
        available_ -= size;
    }
    void skip_space() {
        while (space(peek()))
            get();
    }
    std::string token() {
        skip_space();
        std::string result;
        while (peek() != std::char_traits<char>::eof() && !space(peek())) {
            if (result.size() == 128)
                fail("Field token too long");
            result += static_cast<char>(get());
        }
        if (result.empty())
            fail("Unexpected end of field input");
        return result;
    }
    std::unique_ptr<io::LzmaInputBuffer> compressed_;
    std::unique_ptr<std::istream> decoded_;
    std::istream* input_;
    const FieldReadLimits& limits_;
    U64 budget_, available_;
    bool text_;
};

void text_real(std::ostream& stream, double value) {
    if (std::isnan(value))
        stream << "nan";
    else if (std::isinf(value))
        stream << (value < 0 ? "-inf" : "inf");
    else
        stream << value;
    stream.put(' ');
}

class Writer {
public:
    Writer(std::ostream& output, bool text) : output_(output), text_(text) {
        numeric_.imbue(std::locale::classic());
        numeric_ << std::setprecision(std::numeric_limits<double>::max_digits10);
        literal(text_ ? "AEROFIELDS " : "AEROFLDB");
        u64(1);
        if (text_)
            literal("TEXT\n");
    }
    void literal(const char* text) { output_.write(text, static_cast<std::streamsize>(std::strlen(text))); }
    void block(const char* label) {
        if (text_) {
            output_.put('\n');
            literal(label);
            output_.put(' ');
        }
    }
    void u64(U64 value) { number_codec::write_word(output_, value, text_); }
    void real(double value) {
        if (!text_) {
            u64(number_codec::real_bits(value));
            return;
        }
        numeric_.clear();
        numeric_.str("");
        text_real(numeric_, value);
        const auto s = numeric_.str();
        output_.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
    void values(const std::vector<double>& data, U64 count) {
        if (!text_) {
            // Двоичный формат хранит компоненты целыми столбцами в том же порядке, что и Field::values.
            if (number_codec::native_little_endian) {
                // Ограничиваем один вызов write, чтобы длина помещалась в streamsize на всех платформах.
                for (std::size_t first = 0; first < data.size();) {
                    const auto count_in_block = std::min<std::size_t>(data.size() - first, 65536);
                    output_.write(reinterpret_cast<const char*>(data.data() + first),
                                  static_cast<std::streamsize>(count_in_block * sizeof(double)));
                    first += count_in_block;
                }
            } else {
                for (double value : data)
                    real(value);
            }
            return;
        }
        // Форматируем независимые блоки параллельно, а записываем в исходной последовательности.
        const std::size_t chunk = 512, batch = 32 * chunk;
        for (std::size_t start = 0; start < data.size();) {
            const auto size = std::min(batch, data.size() - start);
            std::vector<std::string> parts((size + chunk - 1) / chunk);
            parallel_for(
                parts.size(),
                [&](std::size_t part) {
                    std::ostringstream s;
                    s.imbue(std::locale::classic());
                    s << std::setprecision(std::numeric_limits<double>::max_digits10);
                    const auto end = start + std::min((part + 1) * chunk, size);
                    for (std::size_t i = start + part * chunk; i < end; ++i) {
                        text_real(s, data[i]);
                        if ((i + 1) % 8 == 0 || (i + 1) % count == 0)
                            s.put('\n');
                    }
                    parts[part] = s.str();
                },
                8);
            for (const auto& part : parts)
                output_.write(part.data(), static_cast<std::streamsize>(part.size()));
            start += size;
        }
    }
    void string(const std::string& value) {
        io::NativeFileCodec::write_string(output_,text_,value,' ',fail,"Field string too large");
    }
    void hash(const std::string& value) {
        if (text_)
            string(value);
        else
            for (std::size_t i = 0; i < 64; i += 2)
                output_.put(static_cast<char>((hex_value(value[i]) << 4) | hex_value(value[i + 1])));
    }
    void location(FieldLocation value) {
        if (text_) {
            literal(location_name(value));
            output_.put(' ');
        } else
            output_.put(static_cast<char>(value));
    }
    void finish() {
        block("END");
        if (text_)
            output_.put('\n');
        if (!output_)
            fail("Field output failed");
    }

private:
    std::ostream& output_;
    bool text_;
    std::ostringstream numeric_;
};
void write_payload(const FieldSnapshot& snapshot, std::ostream& out, bool text,
                   std::vector<U64>* offsets = nullptr) {
    Writer writer(out, text);
    writer.block("DIMENSION");
    writer.u64(snapshot.dimension);
    writer.block("MESH_SHA256");
    writer.hash(snapshot.mesh_sha256);
    writer.block("COUNTS");
    for (U64 value : snapshot.counts)
        writer.u64(value);
    writer.block("STEP");
    writer.u64(snapshot.step);
    writer.block("TIME");
    writer.real(snapshot.time);
    writer.block("FIELDS");
    writer.u64(snapshot.fields.size());
    for (const Field& field : snapshot.fields) {
        writer.block("FIELD");
        writer.string(field.name);
        writer.block("LOCATION");
        writer.location(field.location);
        writer.block("COMPONENTS");
        writer.u64(field.components.size());
        for (const std::string& c : field.components)
            writer.string(c);
        writer.block("UNITS");
        writer.string(field.units);
        writer.block("VALUES");
        writer.u64(snapshot.entity_count(field.location));
        writer.block("DATA");
        const U64 count = snapshot.entity_count(field.location);
        if (offsets) {
            const std::streamoff pos = out.tellp();
            const U64 size = product(product(count, field.components.size()), 8);
            if (pos < 0 || size > static_cast<U64>(std::numeric_limits<std::streamoff>::max() - pos))
                fail("Binary field offset exceeds file size limit");
            offsets->push_back(static_cast<U64>(pos));
            if (size) {
                out.seekp(static_cast<std::streamoff>(size - 1), std::ios::cur);
                out.put(0);
            }
        } else
            writer.values(field.values, count);
        writer.block("END_FIELD");
    }
    writer.finish();
}
void write_encoded(const FieldSnapshot& snapshot, std::ostream& out, FieldFormat format) {
    if (format == FieldFormat::Lzma) {
        io::LzmaOutputBuffer buffer(out);
        std::ostream compressed(&buffer);
        compressed.exceptions(std::ios::badbit | std::ios::failbit);
        write_payload(snapshot, compressed, false);
        buffer.finish();
    } else
        write_payload(snapshot, out, format == FieldFormat::Text);
}
} // namespace

U64 FieldSnapshot::entity_count(FieldLocation location) const {
    location_name(location);
    return counts[static_cast<std::size_t>(location)];
}
void FieldSnapshot::validate() const {
    validate_fields(false);
}
void FieldSnapshot::validate_fields(bool layout_only) const {
    if (dimension < 1 || dimension > 3)
        fail("Field dimension must be 1, 2 or 3");
    if (!valid_hash(mesh_sha256))
        fail("Invalid mesh SHA-256; expected 64 lowercase hex characters");
    if (!std::isfinite(time))
        fail("Field snapshot time must be finite");
    std::set<std::pair<FieldLocation, std::string>> names;
    for (const Field& field : fields) {
        const U64 count = entity_count(field.location);
        if (field.name.empty() || !io::valid_label_text(field.name) || !io::valid_label_text(field.units))
            fail("Invalid field name or units");
        if (!names.insert(std::make_pair(field.location, field.name)).second)
            fail("Duplicate field name at location");
        if (field.components.empty())
            fail("Field must have components");
        std::set<std::string> components;
        for (const std::string& c : field.components) {
            if (c.empty() || !io::valid_label_text(c))
                fail("Invalid component name");
            if (!components.insert(c).second)
                fail("Duplicate component name");
        }
        const U64 size = product(count, field.components.size());
        if ((layout_only ? 0 : size) != field.values.size())
            fail("Field values size mismatch");
    }
}
std::vector<U64> FieldSnapshot::write_binary_layout(const std::string& path) const {
    if (path.find('\0') != std::string::npos)
        fail("NUL in field path");
    validate_fields(true);
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out)
        fail("Cannot create binary field layout: " + path);
    std::vector<U64> offsets;
    write_payload(*this, out, false, &offsets);
    out.close();
    if (!out)
        fail("Cannot finish binary field layout: " + path);
    return offsets;
}
// Загружает одно поле. При выборочном чтении MPI нужны только запрошенные строки узлов.
// Прочие данные пропускаем до начала следующего поля.
static void read_field_values(Reader& reader, Field& field, U64 count, U64 components,
                              const FieldNodeSelection* selection) {
    if (!selection) {
        reader.resize(field.values, product(count, components));
        reader.read_values(field.values.data(), field.values.size());
    } else if (field.location == FieldLocation::Node) {
        const auto& ids = selection->nodes;
        reader.resize(field.values, product(ids.size(), components));
        std::size_t written = 0;
        for (U64 c = 0; c < components; ++c) {
            U64 next = 0;
            for (std::size_t begin = 0; begin < ids.size();) {
                // Идентификаторы отсортированы. Смежные узлы читаем подряд без лишних перемещений по файлу.
                std::size_t end = begin + 1;
                while (end < ids.size() && ids[end] == ids[end - 1] + 1)
                    ++end;
                reader.skip_values(ids[begin] - next);
                reader.read_values(field.values.data() + written, end - begin);
                written += end - begin;
                next = ids[end - 1] + 1;
                begin = end;
            }
            reader.skip_values(count - next);
        }
    } else
        reader.skip_values(product(count, components));
}

// Читает метаданные до значений. Это позволяет сверить сетку и размеры до выделения памяти.
static FieldSnapshot read_snapshot(std::istream& input, const FieldReadLimits& limits,
                                   const FieldNodeSelection* selection) try {
    Reader reader(input, limits);
    FieldSnapshot result;
    reader.block("DIMENSION");
    const U64 dimension = reader.u64();
    if (dimension < 1 || dimension > 3)
        fail("Field dimension must be 1, 2 or 3");
    result.dimension = static_cast<unsigned>(dimension);
    reader.block("MESH_SHA256");
    result.mesh_sha256 = reader.hash();
    if (!valid_hash(result.mesh_sha256))
        fail("Invalid mesh SHA-256");
    reader.block("COUNTS");
    for (U64& count : result.counts)
        count = reader.u64();
    if (selection) {
        if (selection->mesh_sha256.empty())
            fail("File initialization requires a mesh fingerprint; regenerate the .mdec with meshdec");
        if (selection->mesh_sha256 != result.mesh_sha256)
            fail("Field mesh SHA-256 mismatch");
        if (dimension != selection->dimension || result.counts != selection->counts)
            fail("Field mesh dimension/counts mismatch");
        const auto& ids = selection->nodes;
        if (!ids.empty() &&
            (ids.back() >= result.counts[0] ||
             std::adjacent_find(ids.begin(), ids.end(), std::greater_equal<U64>()) != ids.end()))
            fail("Invalid selected node IDs");
    }
    reader.block("STEP");
    result.step = reader.u64();
    reader.block("TIME");
    result.time = reader.real();
    if (!std::isfinite(result.time))
        fail("Field snapshot time must be finite");
    reader.block("FIELDS");
    const U64 field_count = reader.u64();
    if (field_count > limits.max_fields)
        fail("Field count exceeds limit");
    reader.resize(result.fields, field_count);
    for (Field& field : result.fields) {
        reader.block("FIELD");
        field.name = reader.string();
        reader.block("LOCATION");
        field.location = reader.location();
        reader.block("COMPONENTS");
        const U64 components = reader.u64();
        if (!components || components > limits.max_components)
            fail("Field component count invalid or exceeds limit");
        reader.resize(field.components, components);
        for (std::string& c : field.components)
            c = reader.string();
        reader.block("UNITS");
        field.units = reader.string();
        reader.block("VALUES");
        const U64 count = reader.u64();
        if (count != result.entity_count(field.location))
            fail("Field entity count mismatch");
        reader.block("DATA");
        read_field_values(reader, field, count, components, selection);
        reader.block("END_FIELD");
    }
    if (selection)
        result.counts = {{selection->nodes.size(), 0, 0, 0}};
    reader.finish();
    result.validate();
    return result;
} catch (const io::CompressionError& error) {
    throw FieldError(error.what());
} catch (const std::ios_base::failure& error) {
    throw FieldError(std::string("Field input failed: ") + error.what());
}
FieldSnapshot FieldSnapshot::read(std::istream& input, const FieldReadLimits& limits) {
    return read_snapshot(input, limits, nullptr);
}
// Для полной загрузки и выборки узлов выполняем одинаковую проверку пути и открытия.
static std::ifstream open_field_input(const std::string& path) {
    if (path.find('\0') != std::string::npos)
        fail("NUL in field path");
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
        fail("Cannot open fields for reading: " + path);
    return input;
}
std::vector<Field> FieldSnapshot::read_nodes(const std::string& path, const FieldNodeSelection& selection) {
    auto input = open_field_input(path);
    FieldReadLimits limits;
    // Число элементов проверено по архиву. Весь файл может превышать память процесса,
    // поэтому выделяем место только под его локальные узлы NODE.
    limits.max_decoded_bytes = std::numeric_limits<U64>::max();
    limits.max_payload_bytes = std::numeric_limits<U64>::max();
    return read_snapshot(input, limits, &selection).fields;
}
FieldSnapshot FieldSnapshot::read(const std::string& path, const FieldReadLimits& limits) {
    auto input = open_field_input(path);
    return read(input, limits);
}
void FieldSnapshot::write(std::ostream& output, FieldFormat format) const try {
    validate();
    check_format(format);
    write_encoded(*this, output, format);
} catch (const io::CompressionError& error) {
    throw FieldError(error.what());
} catch (const std::ios_base::failure& error) {
    throw FieldError(std::string("Field output failed: ") + error.what());
}
void FieldSnapshot::write(const std::string& path, FieldFormat format) const {
    validate();
    check_format(format);
    if (path.find('\0') != std::string::npos)
        fail("NUL in field path");
    std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!output)
        fail("Cannot open fields for writing: " + path);
    write(output, format);
    output.close();
    if (!output)
        fail("Cannot finish writing fields: " + path);
}
} // namespace aero
