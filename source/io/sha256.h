#ifndef AERO_IO_SHA256_H
#define AERO_IO_SHA256_H
#include <cstddef>
#include <cstdint>
#include <streambuf>
#include <string>

namespace aero { namespace io_detail {
// Internal streaming SHA-256, FIPS 180-4 section 6.2.
class Sha256Buffer : public std::streambuf {
public:
    Sha256Buffer();
    std::string hex_digest() const;
protected:
    std::streamsize xsputn(const char* data, std::streamsize size) override;
    int_type overflow(int_type value) override;
private:
    void append(const unsigned char* data, std::size_t size);
    void transform(const unsigned char* data);
    std::uint32_t state_[8];
    unsigned char block_[64];
    std::size_t used_;
    std::uint64_t bytes_;
};
} }
#endif
