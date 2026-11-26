#pragma once
#include "test_checks.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>

namespace aero_test {
// Владеет именем файла теста. Удаление выполняется один раз при выходе из области.
// Конструктор с проверкой сначала убеждается, что путь свободен.
class TestFile {
public:
    std::string path; // Полный путь теста относительно корня рабочей папки.

    // Принимает уже сформированное имя, например содержащее пробелы и знак решётки.
    explicit TestFile(std::string name) : path(std::move(name)) {}

    // Порождает имя с меткой времени и счётчиком и при необходимости проверяет коллизию.
    TestFile(const char* prefix, const char* suffix, const char* collision)
        : path(unique_path(prefix, suffix)) {
        if (collision) {
            std::ifstream existing(path.c_str(), std::ios::binary);
            require(!existing, collision);
        }
    }
    // Освобождает файл после успешной проверки и при исключении в тесте.
    ~TestFile() { std::remove(path.c_str()); }
    // У одного временного пути должен оставаться единственный владелец.
    TestFile(const TestFile&) = delete;
    // Запрещает подменять путь другого владельца присваиванием.
    TestFile& operator=(const TestFile&) = delete;

private:
    // Счётчик различает файлы, созданные за один тик часов в данном наборе тестов.
    static std::string unique_path(const char* prefix, const char* suffix) {
        static unsigned sequence = 0;
        return std::string("build/") + prefix +
               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
               "-" + std::to_string(sequence++) + suffix;
    }
};

// Заменяет ровно первое вхождение в заготовке и сохраняет диагностическое сообщение теста.
inline std::string replace_first(std::string text, const std::string& from, const std::string& to,
                                 const char* missing, bool show_token = true) {
    const auto pos = text.find(from);
    if (pos == std::string::npos)
        throw std::runtime_error(show_token ? missing + from : std::string(missing));
    text.replace(pos, from.size(), to);
    return text;
}
}
