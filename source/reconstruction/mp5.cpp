#include "mp5.h"
#include "scheme.h"
#include <cmath>

namespace aero {
namespace {
constexpr double alpha_max = 4.0;
constexpr double safety_factor = 1.0;
constexpr double c_ssp = 1.0; // Радиус SSP равен единице у обоих поддерживаемых интеграторов.
} // namespace

double mp5_alpha_from_cfl(double cfl) {
    if (!(cfl > 0) || !std::isfinite(cfl))
        throw ReconstructionError("MP5 requires a positive finite CFL");
    return std::fmin(alpha_max, safety_factor * std::fmax(0.0, c_ssp / cfl - 1.0));
}

void validate_mp5_alpha(double alpha) {
    if (!(alpha >= 0 && alpha <= alpha_max))
        throw ReconstructionError("MP5 alpha must be in [0, 4]");
}
} // namespace aero
