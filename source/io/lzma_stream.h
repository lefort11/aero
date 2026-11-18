#ifndef AERO_IO_LZMA_STREAM_H
#define AERO_IO_LZMA_STREAM_H

#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <streambuf>
#include <vector>

namespace aero { namespace io {

std::uint32_t crc32(const char* data,std::size_t size);

class CompressionError : public std::runtime_error {
public:
    explicit CompressionError(const char* message) : std::runtime_error(message) {}
};

// Host-only, bounded-memory LZMA container. See README.txt for the wire format.
// Streams must outlive their buffers. Neither buffer owns the underlying stream.
class LzmaOutputBuffer : public std::streambuf {
public:
    explicit LzmaOutputBuffer(std::ostream& output);
    // Required; writes the final block and terminator. Destructor does no I/O.
    void finish();
protected:
    int_type overflow(int_type value) override;
    int sync() override;
private:
    void flush_block();
    std::ostream& output_;
    std::vector<char> raw_;
    bool finished_;
    LzmaOutputBuffer(const LzmaOutputBuffer&) = delete;
    LzmaOutputBuffer& operator=(const LzmaOutputBuffer&) = delete;
};

class LzmaInputBuffer : public std::streambuf {
public:
    // Caller has already consumed the eight-byte AEROLZMA signature.
    LzmaInputBuffer(std::istream& input, std::uint64_t max_decoded_bytes);
protected:
    int_type underflow() override;
private:
    std::istream& input_;
    std::vector<char> raw_;
    std::uint64_t remaining_;
    bool finished_;
    LzmaInputBuffer(const LzmaInputBuffer&) = delete;
    LzmaInputBuffer& operator=(const LzmaInputBuffer&) = delete;
};

} } // namespace aero::io
#endif
