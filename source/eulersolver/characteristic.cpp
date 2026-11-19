#include "characteristic.h"
namespace aero {
// Обёртка для одного вектора CPU. Выделяет выход и передаёт компоненты общей формуле CPU/CUDA.
std::vector<double> characteristic_to_conservative(const EulerRoeBasis& basis,
                                                   const std::vector<double>& characteristic) {
    if (basis.dimension < 1 || basis.dimension > 3)
        throw EulerError(EulerStatus::InvalidDimension);
    if (characteristic.size() != basis.dimension + 2)
        throw std::invalid_argument("Expected dimension+2 characteristic components");
    std::vector<double> result(characteristic.size());
    const EulerStatus status = characteristic_to_conservative(basis, characteristic.data(), result.data());
    if (status != EulerStatus::Success)
        throw EulerError(status);
    return result;
}
} // namespace aero
