#pragma once
#include "test_checks.h"
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace aero_test {
// Реестр именованных тестов одного исполняемого файла.
using TestSuite = std::vector<std::pair<std::string, std::function<void()>>>;

// Перемещает функцию в реестр, сохраняя имя для диагностики упавшей проверки.
inline void add_test(TestSuite& suite, const std::string& name, std::function<void()> test) {
    suite.emplace_back(name, std::move(test));
}

// Проверяет тип исключения и необязательную часть сообщения об ошибке входных данных.
template<class Error, class Work>
void expect_error(const Work& work, const std::string& part, const char* missing) {
    try {
        work();
    } catch (const Error& error) {
        require(part.empty() || std::string(error.what()).find(part) != std::string::npos,
                "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error(missing);
}

// Выполняет проверки независимо и печатает итог в прежнем формате данного набора.
inline int run_tests(const TestSuite& suite, const char* summary) {
    std::size_t failed = 0;
    for (const auto& test : suite) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << suite.size() - failed << '/' << suite.size() << summary << '\n';
    return failed ? 1 : 0;
}
}
