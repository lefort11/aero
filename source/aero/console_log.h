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
    explicit ConsoleMirror(std::ostream& stream) : stream_(stream), terminal_(stream.rdbuf()) {}
    void attach(std::streambuf& file) {
        file_ = &file;
        stream_.rdbuf(this);
    }
    ~ConsoleMirror() {
        if (file_) {
            pubsync();
            stream_.rdbuf(terminal_);
        }
    }
private:
    std::ostream& stream_;
    std::streambuf* terminal_;
    std::streambuf* file_ = nullptr;
    std::streamsize xsputn(const char* data, std::streamsize size) override {
        const auto shown = terminal_->sputn(data, size);
        const auto saved = file_->sputn(data, size);
        return std::min(shown, saved);
    }
    int_type overflow(int_type value) override {
        if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
        const char c = traits_type::to_char_type(value);
        return xsputn(&c, 1) == 1 ? value : traits_type::eof();
    }
    int sync() override {
        const int shown = terminal_->pubsync();
        const int saved = file_->pubsync();
        return shown == 0 && saved == 0 ? 0 : -1;
    }
};

// Оба потока восстанавливаются до закрытия файла, в том числе при ошибке расчёта.
class ConsoleLog {
public:
    void open(const std::string& path) {
        file_.open(path.c_str(), std::ios::app);
        if (!file_) throw std::runtime_error("Cannot open console log " + path);
        output_.attach(*file_.rdbuf());
        errors_.attach(*file_.rdbuf());
    }
private:
    std::ofstream file_;
    ConsoleMirror output_{std::cout}, errors_{std::cerr};
};
}
#endif
