#ifndef AERO_QA_TEST_CHECKS_H
#define AERO_QA_TEST_CHECKS_H

#include <stdexcept>

namespace aero_test {
// Останавливает текущий сценарий с его собственным пояснением. Строка нужна только при ошибке.
template <class Message> inline void require(bool condition, const Message& message) {
    if (!condition)
        throw std::runtime_error(message);
}
}
#endif
