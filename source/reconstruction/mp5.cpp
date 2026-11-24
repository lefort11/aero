#include "mp5.h"
#include "scheme.h"
#include <cmath>

namespace aero {
namespace {
constexpr double alpha_max = 4.0;
constexpr double safety_factor = 1.0;
constexpr double c_ssp = 1.0; // Радиус SSP равен единице у Эйлера и SSPRK3.
} // namespace

// Выбирает параметр ограничителя MP5 по заданному числу Куранта.
double mp5_alpha_from_cfl(double cfl) {
    return mp5_alpha_for_time_scheme(cfl, false);
}

// Согласует ограничитель MP5 с CFL и временной схемой.
double mp5_alpha_for_time_scheme(double cfl, bool implicit_time) {
    if (!(cfl > 0) || !std::isfinite(cfl))
        throw ReconstructionError("MP5 requires a positive finite CFL");
    // В BDF и при выходе на стационар ограничение на явный шаг SSP не применяется.
    if (implicit_time) return alpha_max;
    return std::fmin(alpha_max, safety_factor * std::fmax(0.0, c_ssp / cfl - 1.0));
}

// Проверяет допустимость параметра ограничителя MP5.
void validate_mp5_alpha(double alpha) {
    if (!(alpha >= 0 && alpha <= alpha_max))
        throw ReconstructionError("MP5 alpha must be in [0, 4]");
}
} // namespace aero
