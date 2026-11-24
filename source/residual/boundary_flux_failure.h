#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

namespace aero {
// Номер площадки нужен владельцу узла для отката шага и понижения порядка.
// На GPU достаточно передать на CPU один индекс из уже существующей проверки.
struct BoundaryFluxFailure : std::domain_error {
    // Номер площадки, на которой вычисление потока завершилось ошибкой.
    std::size_t face;
    // Сохраняет номер внешней площадки с недопустимым потоком.
    explicit BoundaryFluxFailure(std::size_t index)
        : std::domain_error("Invalid boundary flux at local face " + std::to_string(index)), face(index) {}
};
}
