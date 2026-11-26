#ifndef AERO_TESTS_TEST_CHECKS_H
#define AERO_TESTS_TEST_CHECKS_H

#include <cstring>
#include <cmath>
#include <string>
#include <stdexcept>

namespace aero_test {
// Распознаёт требование запустить тестовое ядро на видимом устройстве.
inline bool require_gpu_argument(const char* value) {
    return std::strcmp(value, "--require-gpu") == 0 || std::strcmp(value, "--require-cuda") == 0;
}

// Останавливает текущий сценарий с его собственным пояснением. Строка нужна только при ошибке.
template <class Message> inline void require(bool condition, const Message& message) {
    if (!condition)
        throw std::runtime_error(message);
}
// Проверяет относительный допуск с абсолютной частью. Текст с числами строится
// только при расхождении. Проверка конечности нужна в тестах газодинамического потока.
inline void require_close(double actual, double expected, double tolerance, bool require_finite) {
    if ((!require_finite || std::isfinite(actual)) &&
        std::abs(actual - expected) <= tolerance * (1 + std::abs(expected))) return;
    throw std::runtime_error("Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

}
#endif
