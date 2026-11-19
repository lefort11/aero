#include "lax_friedrichs.h"

namespace aero {
// Обёртка Русанова для двух векторов состояния в памяти CPU.
EulerFluxResult lax_friedrichs_conservative(const std::vector<double>& left, const std::vector<double>& right,
                                            const std::vector<double>& direction, double gamma) {
    return euler_detail::checked_state_flux(lax_friedrichs_conservative, left, right, direction, gamma);
}
// Передаёт готовые физические потоки и состояния в общую проверенную обёртку.
EulerFluxResult lax_friedrichs_fluxes(const std::vector<double>& fl, const std::vector<double>& fr,
                                      const std::vector<double>& ul, const std::vector<double>& ur,
                                      const std::vector<double>& direction, double gamma) {
    return euler_detail::checked_supplied_flux(lax_friedrichs_fluxes, fl, fr, ul, ur, direction, gamma);
}
} // namespace aero
