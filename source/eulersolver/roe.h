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
        EulerStatus status = euler_detail::build_roe_basis<false>(dimension, gas.left, gas.right, face.direction,
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
        status = euler_detail::project_characteristic<0, false>(basis, gas.normal, face.gamma, basis.kinetic_energy(),
                                                       jump, strength);
        if (status != EulerStatus::Success)
            return status;
        EulerFluxResult output = {};
        output.components = count;
        const unsigned plus = count - 1;
        const double transport = euler_detail::absolute(basis.eigenvalues[1]);
        const double left_speed = euler_detail::absolute(basis.eigenvalues[0]);
        const double right_speed = euler_detail::absolute(basis.eigenvalues[plus]);
        // u·n лежит между u·n−c и u·n+c. Для CFL достаточно сравнить модули этих двух
        // звуковых скоростей с оценкой по левому и правому состояниям.
        output.max_wave_speed = left_speed > speed ? left_speed : speed;
        if (right_speed > output.max_wave_speed)
            output.max_wave_speed = right_speed;

        // Энтропия и поперечный импульс переносятся с одинаковой скоростью |u·n|.
        // Берём эту скорость для всего скачка. Затем доводим вклад двух звуковых волн
        // до их собственных скоростей. Это та же сумма R·|Λ|·L·ΔU в любой размерности.
        const double left_correction = (left_speed - transport) * strength[0];
        const double right_correction = (right_speed - transport) * strength[plus];
        for (unsigned r = 0; r < count; ++r) {
            const double dissipation = transport * jump[r] +
                                       basis.right[5 * r] * left_correction +
                                       basis.right[5 * r + plus] * right_correction;
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
