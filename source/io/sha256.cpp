#include "sha256.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#include <immintrin.h>
#define AERO_X86_SHA256
#endif

namespace aero {
namespace io_detail {
namespace {
std::uint32_t rotate(std::uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}
const std::uint32_t constants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#ifdef AERO_X86_SHA256
bool has_sha_instructions() {
    unsigned a,b,c,d;
    return __get_cpuid(1,&a,&b,&c,&d) && (c & bit_SSSE3) && (c & bit_SSE4_1) &&
           __get_cpuid_count(7,0,&a,&b,&c,&d) && (b & (1u << 29));
}
// На процессорах с SHA-NI четыре раунда обрабатывают те же слова по стандарту SHA-256.
// Состояние остаётся в регистрах между соседними 64-байтовыми блоками сетки.
__attribute__((target("sha,ssse3,sse4.1")))
void sha_blocks(std::uint32_t* state, const unsigned char* data, std::size_t blocks) {
    const __m128i reverse = _mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
    const auto abcd = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(state)),0xb1);
    const auto efgh = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(state+4)),0x1b);
    auto abef = _mm_alignr_epi8(abcd,efgh,8);
    auto cdgh = _mm_blend_epi16(efgh,abcd,0xf0);
    for (std::size_t block = 0; block < blocks; ++block, data += 64) {
        const auto old_abef = abef, old_cdgh = cdgh;
        __m128i words[4];
        for (unsigned group = 0; group < 16; ++group) {
            const unsigned k = group % 4;
            if (group < 4) {
                words[k] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data+16*k)),reverse);
            } else {
                const auto tail = words[(k+3)%4];
                const auto first = _mm_sha256msg1_epu32(words[k],words[(k+1)%4]);
                words[k] = _mm_sha256msg2_epu32(_mm_add_epi32(first,_mm_alignr_epi8(tail,words[(k+2)%4],4)),tail);
            }
            const auto wk = _mm_add_epi32(words[k],_mm_loadu_si128(reinterpret_cast<const __m128i*>(constants+4*group)));
            cdgh = _mm_sha256rnds2_epu32(cdgh,abef,wk);
            abef = _mm_sha256rnds2_epu32(abef,cdgh,_mm_shuffle_epi32(wk,0x0e));
        }
        abef = _mm_add_epi32(abef,old_abef);
        cdgh = _mm_add_epi32(cdgh,old_cdgh);
    }
    const auto first = _mm_shuffle_epi32(abef,0x1b), second = _mm_shuffle_epi32(cdgh,0xb1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state),_mm_blend_epi16(first,second,0xf0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state+4),_mm_alignr_epi8(second,first,8));
}
#endif
}
Sha256Buffer::Sha256Buffer()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
      block_{}, used_(0), bytes_(0) {}

// Обрабатывает один блок из 64 байтов и прибавляет результат раундов к накопленному состоянию SHA-256.
void Sha256Buffer::transform(const unsigned char* data) {
    std::uint32_t w[64];
    for (unsigned i = 0; i < 16; ++i)
        w[i] = (std::uint32_t(data[4 * i]) << 24) | (std::uint32_t(data[4 * i + 1]) << 16) |
               (std::uint32_t(data[4 * i + 2]) << 8) | std::uint32_t(data[4 * i + 3]);
    for (unsigned i = 16; i < 64; ++i) {
        const std::uint32_t a = w[i - 15], b = w[i - 2];
        w[i] = w[i - 16] + (rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3)) + w[i - 7] +
               (rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10));
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4], f = state_[5],
                  g = state_[6], h = state_[7];
    for (unsigned i = 0; i < 64; ++i) {
        const std::uint32_t t1 =
            h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + constants[i] + w[i];
        const std::uint32_t t2 =
            (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}
// Дополняет неполный блок, обрабатывает целые блоки и сохраняет остаток до следующей записи.
void Sha256Buffer::append(const unsigned char* data, std::size_t size) {
    if (size > std::numeric_limits<std::uint64_t>::max() / 8 - bytes_)
        throw std::length_error("SHA-256 input too long");
    bytes_ += size;
    while (size) {
        if (!used_ && size >= 64) {
            const std::size_t blocks = size / 64;
#ifdef AERO_X86_SHA256
            static const bool accelerated = has_sha_instructions();
            if (accelerated) sha_blocks(state_,data,blocks);
            else
#endif
                for (std::size_t i=0; i<blocks; ++i) transform(data+64*i);
            data += blocks*64;
            size -= blocks*64;
            continue;
        }
        const std::size_t count = std::min(size, 64 - used_);
        std::memcpy(block_ + used_, data, count);
        used_ += count;
        data += count;
        size -= count;
        if (used_ == 64) {
            transform(block_);
            used_ = 0;
        }
    }
}
std::streamsize Sha256Buffer::xsputn(const char* data, std::streamsize size) {
    if (size < 0)
        return 0;
    append(reinterpret_cast<const unsigned char*>(data), static_cast<std::size_t>(size));
    return size;
}
Sha256Buffer::int_type Sha256Buffer::overflow(int_type value) {
    if (!traits_type::eq_int_type(value, traits_type::eof())) {
        const unsigned char byte = static_cast<unsigned char>(traits_type::to_char_type(value));
        append(&byte, 1);
    }
    return traits_type::not_eof(value);
}
// Финализирует копию состояния, чтобы исходный поток можно было продолжить после чтения хеша.
std::string Sha256Buffer::hex_digest() const {
    Sha256Buffer copy;
    std::copy(state_, state_ + 8, copy.state_);
    std::copy(block_, block_ + 64, copy.block_);
    copy.used_ = used_;
    const std::uint64_t bits = bytes_ * 8;
    // После данных добавляем единичный бит. Последние восемь байтов займёт длина сообщения в битах.
    copy.block_[copy.used_++] = 0x80;
    if (copy.used_ > 56) {
        std::fill(copy.block_ + copy.used_, copy.block_ + 64, static_cast<unsigned char>(0));
        copy.transform(copy.block_);
        copy.used_ = 0;
    }
    std::fill(copy.block_ + copy.used_, copy.block_ + 56, static_cast<unsigned char>(0));
    for (unsigned i = 0; i < 8; ++i)
        copy.block_[63 - i] = static_cast<unsigned char>(bits >> (8 * i));
    copy.transform(copy.block_);
    const char* digits = "0123456789abcdef";
    std::string result;
    for (unsigned i = 0; i < 8; ++i)
        for (unsigned j = 0; j < 8; ++j)
            result += digits[(copy.state_[i] >> (28 - 4 * j)) & 15];
    return result;
}
}
}
