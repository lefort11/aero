#ifndef AERO_CONSOLE_LOG_H
#define AERO_CONSOLE_LOG_H
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace aero {
// У stdout и stderr остаются свои приёмники в терминале. В файл оба пишут в порядке вывода.
class ConsoleMirror : public std::streambuf {
public:
    // Запоминает поток и его исходный буфер перед подключением журнала.
    explicit ConsoleMirror(std::ostream& stream) : stream_(stream), terminal_(stream.rdbuf()) {}
    // Подключает файл к потоку, сохраняя одновременный вывод в терминал.
    void attach(std::streambuf& file) {
        file_ = &file;
        stream_.rdbuf(this);
    }
    // Сбрасывает вывод и восстанавливает исходный буфер потока.
    ~ConsoleMirror() {
        if (file_) {
            pubsync();
            stream_.rdbuf(terminal_);
        }
    }
private:
    // Поток, чей буфер временно перенаправлен.
    std::ostream& stream_;
    // Исходный буфер консоли для вывода и последующего восстановления.
    std::streambuf* terminal_;
    // Второй приёмник вывода после открытия журнала.
    std::streambuf* file_ = nullptr;
    // Записывает блок символов в терминал и журнал и возвращает общий успешный размер.
    std::streamsize xsputn(const char* data, std::streamsize size) override {
        const auto shown = terminal_->sputn(data, size);
        const auto saved = file_->sputn(data, size);
        return std::min(shown, saved);
    }
    // Передаёт одиночный символ через зеркальную запись потока.
    int_type overflow(int_type value) override {
        if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
        const char c = traits_type::to_char_type(value);
        return xsputn(&c, 1) == 1 ? value : traits_type::eof();
    }
    // Сбрасывает буферы терминала и журнала и проверяет оба результата.
    int sync() override {
        const int shown = terminal_->pubsync();
        const int saved = file_->pubsync();
        return shown == 0 && saved == 0 ? 0 : -1;
    }
};

// Оба потока восстанавливаются до закрытия файла, в том числе при ошибке расчёта.
class ConsoleLog {
public:
    // Открывает журнал для добавления и подключает к нему stdout и stderr.
    void open(const std::string& path) {
        file_.open(path.c_str(), std::ios::app);
        if (!file_) throw std::runtime_error("Cannot open console log " + path);
        output_.attach(*file_.rdbuf());
        errors_.attach(*file_.rdbuf());
    }
private:
    // Владеет открытым файлом журнала.
    std::ofstream file_;
    // Дублируют обычный и диагностический вывод в журнал.
    ConsoleMirror output_{std::cout}, errors_{std::cerr};
};
}
#endif
