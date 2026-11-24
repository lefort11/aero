#include "euler.h"

namespace aero {
// Возвращает диагностический текст кода газодинамического ядра.
const char* euler_status_message(EulerStatus status) {
    switch (status) {
    case EulerStatus::Success:
        return "Success";
    case EulerStatus::InvalidDimension:
        return "Euler dimension must be 1, 2 or 3";
    case EulerStatus::NullPointer:
        return "Null Euler input/output pointer";
    case EulerStatus::InvalidGamma:
        return "Ideal-gas gamma must be finite and greater than one";
    case EulerStatus::InvalidDirection:
        return "Euler flux direction must be nonzero";
    case EulerStatus::InvalidDensity:
        return "Euler density must be positive";
    case EulerStatus::InvalidPressure:
        return "Euler pressure and sound speed must be positive";
    case EulerStatus::NonFinite:
        return "Non-finite Euler input or intermediate result";
    case EulerStatus::InvalidRoeState:
        return "Nonpositive Roe sound speed squared";
    }
    return "Unknown gas-dynamics status";
}
namespace euler_detail {
namespace {
// Размерность определяется направлением, поэтому одна и та же проверка подходит для всех схем.
unsigned dimension_of(const std::vector<double>& direction) {
    if (direction.empty() || direction.size() > 3)
        throw EulerError(EulerStatus::InvalidDimension);
    return static_cast<unsigned>(direction.size());
}
// Проверяет размерность и число компонент перед вызовом потокового ядра.
void check_components(const std::vector<double>& values, unsigned dimension) {
    if (values.size() != dimension + 2)
        throw std::invalid_argument("Expected dimension+2 Euler components");
}
// Преобразует неуспешный код газодинамического ядра в исключение.
void require_success(EulerStatus status) {
    if (status != EulerStatus::Success)
        throw EulerError(status);
}
}
// Вычисляет поток по состояниям и сообщает ошибку через исключение.
EulerFluxResult checked_state_flux(StateFluxKernel kernel, const std::vector<double>& left,
                                   const std::vector<double>& right, const std::vector<double>& direction,
                                   double gamma) {
    const unsigned dimension = dimension_of(direction);
    check_components(left, dimension);
    check_components(right, dimension);
    EulerFluxResult result;
    require_success(kernel(dimension, left.data(), right.data(), direction.data(), gamma, &result));
    return result;
}
// Вычисляет численный поток с заданными физическими потоками и проверяет статус.
EulerFluxResult checked_supplied_flux(SuppliedFluxKernel kernel, const std::vector<double>& flux_left,
                                      const std::vector<double>& flux_right, const std::vector<double>& left,
                                      const std::vector<double>& right, const std::vector<double>& direction,
                                      double gamma) {
    const unsigned dimension = dimension_of(direction);
    // Потоки и состояния независимы при реконструкции потока. Проверяем все четыре массива.
    check_components(flux_left, dimension);
    check_components(flux_right, dimension);
    check_components(left, dimension);
    check_components(right, dimension);
    EulerFluxResult result;
    require_success(kernel(dimension, flux_left.data(), flux_right.data(), left.data(), right.data(),
                           direction.data(), gamma, &result));
    return result;
}
}
} // namespace aero
