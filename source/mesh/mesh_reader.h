#ifndef AERO_MESH_READER_H
#define AERO_MESH_READER_H
#include "mesh.h"
#include "../io/lzma_stream.h"
#include <cstring>
#include <istream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>

namespace aero { namespace mesh_input {
[[noreturn]] inline void fail(const std::string& message) { throw MeshError(message); }
inline bool valid_name(const std::string& name) {
    if (name.empty()) return false;
    std::size_t i = 0;
    while (i < name.size()) {
        const auto c = static_cast<unsigned char>(name[i++]);
        if (c < 0x20 || c == 0x7f) return false;
        if (c < 0x80) continue;
        unsigned code = 0, count = 0, minimum = 0;
        if (c >= 0xc2 && c <= 0xdf) { code = c & 0x1f; count = 1; minimum = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { code = c & 0x0f; count = 2; minimum = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { code = c & 0x07; count = 3; minimum = 0x10000; }
        else return false;
        if (name.size() - i < count) return false;
        for (unsigned j = 0; j < count; ++j) {
            const auto tail = static_cast<unsigned char>(name[i++]);
            if ((tail & 0xc0) != 0x80) return false;
            code = (code << 6) | (tail & 0x3f);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
    }
    return true;
}

inline bool whitespace(int c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v'; }

class Budget {
public:
    explicit Budget(std::uint64_t bytes) : remaining_(bytes) {}
    void consume(std::uint64_t count, std::uint64_t width = 1) {
        if (width == 0 || count > remaining_ / width) fail("Mesh exceeds the configured payload limit");
        remaining_ -= count * width;
    }
private:
    std::uint64_t remaining_;
};

class Reader {
public:
    Reader(std::istream& input, const MeshReadLimits& limits)
        : input_(&input), limits_(limits), budget_(limits.max_payload_bytes) {
        char magic[8];
        bytes(magic, sizeof magic);
        if (std::memcmp(magic, "AEROLZMA", 8) == 0) {
            const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
            // Binary header is 88 bytes; remaining bytes are charged to Budget.
            const std::uint64_t limit = limits.max_payload_bytes > maximum - 88
                ? maximum : limits.max_payload_bytes + 88;
            compressed_.reset(new io::LzmaInputBuffer(input, limit));
            decoded_.reset(new std::istream(compressed_.get()));
            decoded_->exceptions(std::ios::badbit);
            input_ = decoded_.get();
            bytes(magic, sizeof magic);
            if (std::memcmp(magic, "AEROMSHB", 8) != 0)
                fail("LZMA mesh must contain native binary v2");
        }
        if (std::memcmp(magic, "AEROMESH", 8) == 0) text_ = true;
        else if (std::memcmp(magic, "AEROMSHB", 8) == 0) text_ = false;
        else fail("Unrecognized mesh signature");
        if (u64() != 2) fail("Unsupported mesh format version; expected face-based version 2");
        if (text_) block("TEXT");
    }

    void block(const char* expected) {
        if (text_ && token() != expected) fail(std::string("Expected block: ") + expected);
    }

    std::uint64_t u64() {
        if (!text_) {
            unsigned char data[8];
            bytes(reinterpret_cast<char*>(data), sizeof data);
            std::uint64_t value = 0;
            for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(data[i]) << (8 * i);
            return value;
        }
        const std::string value = token();
        std::uint64_t result = 0;
        for (const char c : value) {
            if (c < '0' || c > '9') fail("Invalid unsigned integer");
            const unsigned digit = static_cast<unsigned>(c - '0');
            if (result > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
                fail("Unsigned integer overflow");
            result = result * 10 + digit;
        }
        return result;
    }

    std::int8_t orientation() {
        if (text_) {
            const std::string value = token();
            if (value == "1" || value == "+1") return 1;
            if (value == "-1") return -1;
        } else {
            char value;
            bytes(&value, 1);
            const unsigned char byte = static_cast<unsigned char>(value);
            if (byte == 1) return 1;
            if (byte == 255) return -1;
        }
        fail("Cell face orientation must be +1 or -1");
    }

    double real() {
        if (!text_) {
            const auto bits = u64();
            double value;
            std::memcpy(&value, &bits, sizeof value);
            return value;
        }
        std::istringstream number(token());
        number.imbue(std::locale::classic());
        double result = 0;
        if (!(number >> result) || number.peek() != std::char_traits<char>::eof())
            fail("Invalid floating-point value");
        return result;
    }

    template<class T> void resize(std::vector<T>& values, std::uint64_t count) {
        if (count > values.max_size()) fail("Array is too large for this platform");
        budget_.consume(count, sizeof(T));
        values.resize(static_cast<std::size_t>(count));
    }

    std::string name() {
        std::string result;
        if (!text_) {
            const auto length = u64();
            if (length > limits_.max_name_bytes || length > result.max_size()) fail("Boundary name is too long");
            budget_.consume(length);
            result.resize(static_cast<std::size_t>(length));
            if (!result.empty()) bytes(&result[0], result.size());
            return result;
        }
        skip_space();
        if (input_->get() != '"') fail("Expected a quoted boundary name");
        while (true) {
            int c = input_->get();
            if (c == std::char_traits<char>::eof()) fail("Truncated boundary name");
            if (c == '"') break;
            if (c == '\\') {
                c = input_->get();
                if (c != '\\' && c != '"') fail("Invalid escape in boundary name");
            }
            if (result.size() >= limits_.max_name_bytes) fail("Boundary name is too long");
            budget_.consume(1);
            result.push_back(static_cast<char>(c));
        }
        if (!whitespace(input_->peek()) && input_->peek() != std::char_traits<char>::eof())
            fail("Boundary names must be separated by whitespace");
        return result;
    }

    void finish() {
        block("END");
        if (text_) skip_space();
        if (input_->peek() != std::char_traits<char>::eof()) fail("Trailing data after the mesh");
        if (input_->bad()) fail("Mesh input failed");
    }

private:
    void bytes(char* destination, std::size_t size) {
        if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) fail("Input block is too large");
        if (size != 0 && !input_->read(destination, static_cast<std::streamsize>(size))) fail("Truncated mesh input");
    }
    void skip_space() { while (whitespace(input_->peek())) input_->get(); }
    std::string token() {
        skip_space();
        std::string result;
        while (true) {
            const int c = input_->peek();
            if (c == std::char_traits<char>::eof() || whitespace(c)) break;
            if (result.size() == 128) fail("Mesh token is too long");
            result.push_back(static_cast<char>(input_->get()));
        }
        if (result.empty()) fail("Unexpected end of mesh input");
        return result;
    }
    std::unique_ptr<io::LzmaInputBuffer> compressed_;
    std::unique_ptr<std::istream> decoded_;
    std::istream* input_;
    MeshReadLimits limits_;
    Budget budget_;
    bool text_ = false;
};

} } // namespace aero::mesh_input
#endif
