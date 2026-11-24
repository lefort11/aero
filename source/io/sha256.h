#ifndef AERO_IO_SHA256_H
#define AERO_IO_SHA256_H
#include <cstddef>
#include <cstdint>
#include <streambuf>
#include <string>

namespace aero {
namespace io_detail {
// Потоковое вычисление SHA-256 по разделу 6.2 FIPS 180-4.
class Sha256Buffer : public std::streambuf {
public:
    // Инициализирует состояние SHA-256 стандартными начальными словами.
    Sha256Buffer();
    // Финализирует копию состояния, чтобы исходный поток можно было продолжить после чтения хеша.
    std::string hex_digest() const;

protected:
    // Добавляет блок байтов в поток вычисления SHA-256.
    std::streamsize xsputn(const char* data, std::streamsize size) override;
    // Добавляет одиночный символ в хеширующий поток.
    int_type overflow(int_type value) override;

private:
    // Дополняет неполный блок, обрабатывает целые блоки и сохраняет остаток до следующей записи.
    void append(const unsigned char* data, std::size_t size);
    // Обрабатывает один блок из 64 байтов и прибавляет результат раундов к накопленному состоянию SHA-256.
    void transform(const unsigned char* data);
    // Накопленные восемь слов хеша после последнего полного блока.
    std::uint32_t state_[8];
    // Байты текущего неполного блока.
    unsigned char block_[64];
    // Сколько мест блока уже занято входными байтами.
    std::size_t used_;
    // Длина исходного сообщения для его завершающего поля длины.
    std::uint64_t bytes_;
};
}
}
#endif
