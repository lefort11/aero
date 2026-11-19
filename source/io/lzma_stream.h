#ifndef AERO_IO_LZMA_STREAM_H
#define AERO_IO_LZMA_STREAM_H

#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <streambuf>
#include <vector>

namespace aero {
namespace io {

std::uint32_t crc32(const char* data, std::size_t size);

class CompressionError : public std::runtime_error {
public:
    explicit CompressionError(const char* message) : std::runtime_error(message) {}
};

// Контейнер LZMA читает и пишет блоками с ограниченной памятью. Формат описан в README.txt.
// Буферы ссылаются на потоки, которые должны существовать дольше буферов.
class LzmaOutputBuffer : public std::streambuf {
public:
    explicit LzmaOutputBuffer(std::ostream& output);
    // Обязательное завершение записи: сбрасывает последний блок и признак конца. Деструктор не пишет в поток.
    void finish();

protected:
    int_type overflow(int_type value) override;
    int sync() override;

private:
    void flush_block();
    // Поток назначения должен существовать до завершения записи последнего блока.
    std::ostream& output_;
    std::vector<char> raw_;
    bool finished_;
    LzmaOutputBuffer(const LzmaOutputBuffer&) = delete;
    LzmaOutputBuffer& operator=(const LzmaOutputBuffer&) = delete;
};

class LzmaInputBuffer : public std::streambuf {
public:
    // Перед созданием буфера вызывающий код должен прочитать восьмибайтовую сигнатуру AEROLZMA.
    LzmaInputBuffer(std::istream& input, std::uint64_t max_decoded_bytes);

protected:
    int_type underflow() override;

private:
    std::istream& input_;
    std::vector<char> raw_;
    // Сколько байтов ещё можно распаковать в пределах лимита вызывающего кода.
    std::uint64_t remaining_;
    bool finished_;
    LzmaInputBuffer(const LzmaInputBuffer&) = delete;
    LzmaInputBuffer& operator=(const LzmaInputBuffer&) = delete;
};

}
} // namespace aero::io
#endif
