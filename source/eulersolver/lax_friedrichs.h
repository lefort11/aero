#ifndef AERO_LAX_FRIEDRICHS_H
#define AERO_LAX_FRIEDRICHS_H
#include "../parallel/kernel_math.h"
#include "euler.h"

namespace aero {
namespace euler_detail {
struct RusanovFlux {
    AERO_HOST_DEVICE static EulerStatus combine(const euler_detail::FaceStates& face,
                                                const euler_detail::InterfaceGas&, const double* flux_left,
                                                const double* flux_right, double alpha,
                                                EulerFluxResult* result) {
        const unsigned dimension = face.dimension;
        EulerFluxResult value = {};
        value.components = dimension + 2;
        value.max_wave_speed = alpha;
        for (unsigned k = 0; k < dimension + 2; ++k) {
            if (!finite(flux_left[k]) || !finite(flux_right[k]))
                return EulerStatus::NonFinite;
            // К центральному потоку добавляем затухание скачка с максимальной скоростью волн.
            value.flux[k] =
                0.5 * flux_left[k] + 0.5 * flux_right[k] - ((0.5 * face.upw_coeff) * alpha) * (face.right[k] - face.left[k]);
            if (!finite(value.flux[k]))
                return EulerStatus::NonFinite;
        }
        *result = value;
        return EulerStatus::Success;
    }
};
}

// Поток Русанова для пары состояний. Площадь грани применяет сборщик невязки.
AERO_HOST_DEVICE inline EulerStatus lax_friedrichs_conservative(unsigned dimension, const double* left,
                                                                const double* right, const double* direction,
                                                                double gamma, EulerFluxResult* result) {
    return euler_detail::flux_from_states<euler_detail::RusanovFlux>(
        {dimension, left, right, direction, gamma, 1}, result);
}

// Потоки уже спроецированы на общую единичную нормаль. Состояния нужны для диссипации.
AERO_HOST_DEVICE inline EulerStatus lax_friedrichs_fluxes(unsigned dimension, const double* flux_left,
                                                          const double* flux_right, const double* state_left,
                                                          const double* state_right, const double* direction,
                                                          double gamma, EulerFluxResult* result) {
    return euler_detail::flux_from_reconstructed_fluxes<euler_detail::RusanovFlux>(
        {dimension, state_left, state_right, direction, gamma, 1}, flux_left, flux_right, result);
}

} // namespace aero

#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
namespace aero {
// Обёртки для векторов проверяют размеры и сообщают причину ошибки через исключение.
EulerFluxResult lax_friedrichs_conservative(const std::vector<double>& left, const std::vector<double>& right,
                                            const std::vector<double>& direction, double gamma);
EulerFluxResult lax_friedrichs_fluxes(const std::vector<double>& fl, const std::vector<double>& fr,
                                      const std::vector<double>& ul, const std::vector<double>& ur,
                                      const std::vector<double>& direction, double gamma);

}
#endif
#endif
