#ifndef AERO_EULER_ROE_H
#define AERO_EULER_ROE_H
#include "../parallel/kernel_math.h"
#include "characteristic.h"
namespace aero {
namespace roe_detail {
// Поток Роу равен полусумме физических потоков минус половина волновой диссипации.
struct RoeFlux {
    AERO_HOST_DEVICE static EulerStatus combine(const euler_detail::FaceStates& face,
                                                const euler_detail::InterfaceGas& gas, const double* fl,
                                                const double* fr, double speed, EulerFluxResult* result) {
        const unsigned dimension = face.dimension;
        EulerRoeBasis basis;
        EulerStatus status = euler_detail::build_roe_basis(dimension, gas.left, gas.right, face.direction,
                                                           gas.normal, face.gamma, &basis);
        if (status != EulerStatus::Success)
            return status;
        const unsigned count = dimension + 2;
        double jump[5] = {}, strength[5] = {};
        for (unsigned k = 0; k < count; ++k) {
            jump[k] = face.right[k] - face.left[k];
            if (!euler_detail::finite(jump[k]) || !euler_detail::finite(fl[k]) ||
                !euler_detail::finite(fr[k]))
                return EulerStatus::NonFinite;
        }
        // Раскладываем скачок U_R − U_L по волнам в уже построенном базисе грани.
        status = euler_detail::project_characteristic(basis, gas.normal, face.gamma, basis.kinetic_energy(),
                                                       jump, strength);
        if (status != EulerStatus::Success)
            return status;
        EulerFluxResult output = {};
        output.components = count;
        output.max_wave_speed = speed;
        for (unsigned c = 0; c < count; ++c) {
            if (!euler_detail::finite(strength[c]))
                return EulerStatus::NonFinite;
            const double wave = euler_detail::absolute(basis.eigenvalues[c]);
            if (wave > output.max_wave_speed)
                output.max_wave_speed = wave;
        }
        for (unsigned r = 0; r < count; ++r) {
            // Каждая волна вносит вклад в компоненту потока с весом |λ|.
            double dissipation = 0;
            for (unsigned c = 0; c < count; ++c)
                dissipation +=
                    basis.right[5 * r + c] * euler_detail::absolute(basis.eigenvalues[c]) * strength[c];
            output.flux[r] = 0.5 * fl[r] + 0.5 * fr[r] - 0.5 * dissipation;
            if (!euler_detail::finite(output.flux[r]))
                return EulerStatus::NonFinite;
        }
        *result = output;
        return EulerStatus::Success;
    }
};
}

// При реконструкции состояний вычисляем оба физических потока прямо на грани.
AERO_HOST_DEVICE inline EulerStatus roe_conservative(unsigned dimension, const double* left,
                                                     const double* right, const double* direction,
                                                     double gamma, EulerFluxResult* result) {
    return euler_detail::flux_from_states<roe_detail::RoeFlux>({dimension, left, right, direction, gamma},
                                                               result);
}

// При реконструкции потоков полусумму берём из входных потоков, скачок — из состояний.
AERO_HOST_DEVICE inline EulerStatus roe_fluxes(unsigned dimension, const double* fl, const double* fr,
                                               const double* left, const double* right,
                                               const double* direction, double gamma,
                                               EulerFluxResult* result) {
    return euler_detail::flux_from_reconstructed_fluxes<roe_detail::RoeFlux>(
        {dimension, left, right, direction, gamma}, fl, fr, result);
}
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
EulerFluxResult roe_conservative(const std::vector<double>& left, const std::vector<double>& right,
                                 const std::vector<double>& direction, double gamma);
EulerFluxResult roe_fluxes(const std::vector<double>& fl, const std::vector<double>& fr,
                           const std::vector<double>& left, const std::vector<double>& right,
                           const std::vector<double>& direction, double gamma);
#endif
} // namespace aero
#endif
