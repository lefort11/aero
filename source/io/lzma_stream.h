#ifndef AERO_IO_LZMA_STREAM_H
#define AERO_IO_LZMA_STREAM_H

#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <streambuf>
#include <vector>

namespace aero {
namespace io {

// Контрольная сумма проверяет несжатые байты блока. Таблица для всех вызовов строится один раз.
std::uint32_t crc32(const char* data, std::size_t size);

// Исключение при чтении или записи сжатого контейнера.
class CompressionError : public std::runtime_error {
public:
    // Создаёт исключение с переданным диагностическим сообщением.
    explicit CompressionError(const char* message) : std::runtime_error(message) {}
};

// Контейнер LZMA читает и пишет блоками с ограниченной памятью. Формат описан в README.txt.
// Буферы ссылаются на потоки, которые должны существовать дольше буферов.
class LzmaOutputBuffer : public std::streambuf {
public:
    // Подготавливает блочное сжатие и записывает начало контейнера AEROLZMA.
    explicit LzmaOutputBuffer(std::ostream& output);
    // Обязательное завершение записи: сбрасывает последний блок и признак конца. Деструктор не пишет в поток.
    void finish();

protected:
    // Сбрасывает заполненный выходной блок и принимает следующий символ.
    int_type overflow(int_type value) override;
    // Сжимает накопленные данные и синхронизирует базовый выходной поток.
    int sync() override;

private:
    // Сжимает заполненную часть буфера и записывает её размеры, контрольную сумму, свойства и данные.
    void flush_block();
    // Поток назначения должен существовать до завершения записи последнего блока.
    std::ostream& output_;
    std::vector<char> raw_;
    bool finished_;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    LzmaOutputBuffer(const LzmaOutputBuffer&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    LzmaOutputBuffer& operator=(const LzmaOutputBuffer&) = delete;
};

// Потоковый буфер чтения и проверки блоков контейнера AEROLZMA.
class LzmaInputBuffer : public std::streambuf {
public:
    // Перед созданием буфера вызывающий код должен прочитать восьмибайтовую сигнатуру AEROLZMA.
    LzmaInputBuffer(std::istream& input, std::uint64_t max_decoded_bytes);

protected:
    // Распаковывает следующий блок при исчерпании входного буфера.
    int_type underflow() override;

private:
    std::istream& input_;
    std::vector<char> raw_;
    // Сколько байтов ещё можно распаковать в пределах лимита вызывающего кода.
    std::uint64_t remaining_;
    bool finished_;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    LzmaInputBuffer(const LzmaInputBuffer&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    LzmaInputBuffer& operator=(const LzmaInputBuffer&) = delete;
};

}
} // namespace aero::io
#endif
