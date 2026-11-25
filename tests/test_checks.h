#ifndef AERO_TESTS_TEST_CHECKS_H
#define AERO_TESTS_TEST_CHECKS_H

#include <cstring>
#include <stdexcept>

namespace aero_test {
inline bool require_gpu_argument(const char* value) {
    return std::strcmp(value, "--require-gpu") == 0 || std::strcmp(value, "--require-cuda") == 0;
}

// Останавливает текущий сценарий с его собственным пояснением. Строка нужна только при ошибке.
template <class Message> inline void require(bool condition, const Message& message) {
    if (!condition)
        throw std::runtime_error(message);
}
}
#endif
