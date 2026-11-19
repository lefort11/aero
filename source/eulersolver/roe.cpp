#include "roe.h"
namespace aero {
// Вход — два восстановленных состояния. Проверку размеров и вызов ядра выполняет общая обёртка.
EulerFluxResult roe_conservative(const std::vector<double>& left, const std::vector<double>& right,
                                 const std::vector<double>& direction, double gamma) {
    return euler_detail::checked_state_flux(roe_conservative, left, right, direction, gamma);
}
// При реконструкции потоков передаём также сами состояния для диссипации Роу.
EulerFluxResult roe_fluxes(const std::vector<double>& fl, const std::vector<double>& fr,
                           const std::vector<double>& left, const std::vector<double>& right,
                           const std::vector<double>& direction, double gamma) {
    return euler_detail::checked_supplied_flux(roe_fluxes, fl, fr, left, right, direction, gamma);
}
} // namespace aero
