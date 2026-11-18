#include "lzma_stream.h"
#include "../../third_party/lzma/LzmaDec.h"
#include "../../third_party/lzma/LzmaEnc.h"

#include <array>
#include <cstdlib>
#include <istream>
#include <ostream>

namespace aero { namespace io {
namespace {
const std::uint32_t block_size = 1024 * 1024;
const std::uint32_t packed_limit = block_size + block_size / 8 + 65536;
void* allocate(ISzAllocPtr, size_t size) { return std::malloc(size); }
void release(ISzAllocPtr, void* address) { std::free(address); }
const ISzAlloc allocator = {allocate, release};

void read_bytes(std::istream& in, void* data, std::size_t size) {
    if (!in.read(static_cast<char*>(data), static_cast<std::streamsize>(size)))
        throw CompressionError("Truncated LZMA container");
}
void write_bytes(std::ostream& out, const void* data, std::size_t size) {
    if (!out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size)))
        throw CompressionError("LZMA output failed");
}
std::uint32_t read_u32(std::istream& in) {
    unsigned char bytes[4]; read_bytes(in, bytes, 4);
    std::uint32_t result = 0;
    for (unsigned i = 0; i < 4; ++i) result |= std::uint32_t(bytes[i]) << (8 * i);
    return result;
}
void write_u32(std::ostream& out, std::uint32_t value) {
    unsigned char bytes[4];
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<unsigned char>(value >> (8 * i));
    write_bytes(out, bytes, 4);
}
} // namespace
std::uint32_t crc32(const char* data, std::size_t size) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> values = {};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (unsigned bit = 0; bit < 8; ++bit) c = (c >> 1) ^ ((c & 1) ? 0xedb88320U : 0);
            values[i] = c;
        }
        return values;
    }();
    std::uint32_t c = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i)
        c = table[(c ^ static_cast<unsigned char>(data[i])) & 255] ^ (c >> 8);
    return c ^ 0xffffffffU;
}

LzmaOutputBuffer::LzmaOutputBuffer(std::ostream& output)
    : output_(output), raw_(block_size), finished_(false) {
    write_bytes(output_, "AEROLZMA", 8);
    write_u32(output_, 1);
    setp(raw_.data(), raw_.data() + raw_.size());
}

void LzmaOutputBuffer::flush_block() {
    const std::size_t count = static_cast<std::size_t>(pptr() - pbase());
    if (!count) return;
    std::vector<Byte> packed(packed_limit);
    SizeT packed_size = packed.size();
    Byte properties[LZMA_PROPS_SIZE]; SizeT properties_size = sizeof properties;
    CLzmaEncProps options; LzmaEncProps_Init(&options);
    options.level = 5; options.dictSize = block_size; options.numThreads = 1;
    const SRes result = LzmaEncode(packed.data(), &packed_size,
        reinterpret_cast<const Byte*>(raw_.data()), count, &options,
        properties, &properties_size, 1, nullptr, &allocator, &allocator);
    if (result != SZ_OK || properties_size != sizeof properties)
        throw CompressionError("LZMA encoding failed");
    write_u32(output_, static_cast<std::uint32_t>(count));
    write_u32(output_, static_cast<std::uint32_t>(packed_size));
    write_u32(output_, crc32(raw_.data(), count));
    write_bytes(output_, properties, sizeof properties);
    write_bytes(output_, packed.data(), packed_size);
    setp(raw_.data(), raw_.data() + raw_.size());
}

LzmaOutputBuffer::int_type LzmaOutputBuffer::overflow(int_type value) {
    if (finished_) throw CompressionError("LZMA stream is already finished");
    flush_block();
    if (!traits_type::eq_int_type(value, traits_type::eof())) {
        *pptr() = traits_type::to_char_type(value); pbump(1);
    }
    return traits_type::not_eof(value);
}
int LzmaOutputBuffer::sync() {
    if (!finished_) flush_block();
    output_.flush();
    if (!output_) throw CompressionError("LZMA output failed");
    return 0;
}
void LzmaOutputBuffer::finish() {
    if (finished_) return;
    flush_block(); write_u32(output_, 0);
    finished_ = true;
    setp(raw_.data(), raw_.data());
}

LzmaInputBuffer::LzmaInputBuffer(std::istream& input, std::uint64_t max_decoded_bytes)
    : input_(input), remaining_(max_decoded_bytes), finished_(false) {
    if (read_u32(input_) != 1) throw CompressionError("Unsupported LZMA container version");
}
LzmaInputBuffer::int_type LzmaInputBuffer::underflow() {
    if (gptr() && gptr() < egptr()) return traits_type::to_int_type(*gptr());
    if (finished_) return traits_type::eof();
    const std::uint32_t raw_size = read_u32(input_);
    if (!raw_size) {
        if (input_.peek() != traits_type::eof()) throw CompressionError("Trailing data after LZMA container");
        if (input_.bad()) throw CompressionError("LZMA input failed");
        finished_ = true;
        return traits_type::eof();
    }
    if (raw_size > block_size) throw CompressionError("LZMA block exceeds size limit");
    if (raw_size > remaining_) throw CompressionError("LZMA decoded payload exceeds configured limit");
    const std::uint32_t packed_size = read_u32(input_);
    const std::uint32_t checksum = read_u32(input_);
    if (!packed_size || packed_size > packed_limit) throw CompressionError("Invalid LZMA packed block size");
    Byte properties[LZMA_PROPS_SIZE]; read_bytes(input_, properties, sizeof properties);
    CLzmaProps props;
    if (LzmaProps_Decode(&props, properties, sizeof properties) != SZ_OK || props.dicSize > block_size)
        throw CompressionError("Invalid LZMA properties or dictionary size");
    std::vector<Byte> packed(packed_size); read_bytes(input_, packed.data(), packed.size());
    raw_.resize(raw_size);
    SizeT dest = raw_size, source = packed_size;
    ELzmaStatus status = LZMA_STATUS_NOT_SPECIFIED;
    const SRes result = LzmaDecode(reinterpret_cast<Byte*>(raw_.data()), &dest,
        packed.data(), &source, properties, sizeof properties, LZMA_FINISH_END, &status, &allocator);
    if (result != SZ_OK || dest != raw_size || source != packed_size || status != LZMA_STATUS_FINISHED_WITH_MARK)
        throw CompressionError("Invalid or incomplete LZMA block");
    if (crc32(raw_.data(), raw_.size()) != checksum) throw CompressionError("LZMA block checksum mismatch");
    remaining_ -= raw_size;
    setg(raw_.data(), raw_.data(), raw_.data() + raw_.size());
    return traits_type::to_int_type(*gptr());
}
} } // namespace aero::io
