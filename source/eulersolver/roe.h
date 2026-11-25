#ifndef AERO_EULER_ROE_H
#define AERO_EULER_ROE_H
#include "../parallel/kernel_math.h"
#include "characteristic.h"
namespace aero {
namespace roe_detail {
// Поток Роу равен полусумме физических потоков минус половина волновой диссипации.
struct RoeFlux {
    // На полном двумерном шаблоне размерность уже известна. Компилятор может
    // развернуть циклы по четырём компонентам и оставить состояние в регистрах.
    template <unsigned fixed_dimension = 0>
    KOKKOS_FORCEINLINE_FUNCTION static EulerStatus combine(const euler_detail::FaceStates& face,
                                                const euler_detail::InterfaceGas& gas, const double* fl,
                                                const double* fr, double speed, EulerFluxResult* result, double entropy_fix=0.) {
        const unsigned dimension = fixed_dimension ? fixed_dimension : face.dimension;
        euler_detail::RoeAverage average;
        const auto status = euler_detail::average_roe_state(dimension, gas.left, gas.right, gas.normal,
                                                            face.gamma, average);
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
        // Для диссипации нужны только две звуковые поправки. Полную матрицу на этом пути не храним.
        double momentum[3] = {};
        const double inverse_sound = inverse_value(average.sound_speed);
        euler_detail::project_normal_waves(dimension, average.velocity, gas.normal, face.gamma, average.kinetic,
                                           average.sound_speed, inverse_sound, jump, momentum, strength);
        if (!euler_detail::finite(strength[1]))
            return EulerStatus::NonFinite;
        EulerFluxResult output = {};
        output.components = count;
        const unsigned plus = count - 1;
        const double transport = euler_detail::absolute(average.normal_speed);
        const double left_speed = euler_detail::acoustic_modulus(average.normal_speed - average.sound_speed,entropy_fix*average.sound_speed);
        const double right_speed = euler_detail::acoustic_modulus(average.normal_speed + average.sound_speed,entropy_fix*average.sound_speed);
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
            double minus, plus_wave;
            average.acoustic_row(r, dimension, gas.normal, minus, plus_wave);
            const double dissipation = transport * jump[r] +
                                       minus * left_correction + plus_wave * right_correction;
            output.flux[r] = 0.5 * fl[r] + 0.5 * fr[r] - (0.5 * face.upw_coeff) * dissipation;
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
    return euler_detail::flux_from_states<roe_detail::RoeFlux>({dimension, left, right, direction, gamma, 1},
                                                               result);
}

// При реконструкции потоков полусумму берём из входных потоков, скачок — из состояний.
AERO_HOST_DEVICE inline EulerStatus roe_fluxes(unsigned dimension, const double* fl, const double* fr,
                                               const double* left, const double* right,
                                               const double* direction, double gamma,
                                               EulerFluxResult* result) {
    return euler_detail::flux_from_reconstructed_fluxes<roe_detail::RoeFlux>(
        {dimension, left, right, direction, gamma, 1}, fl, fr, result);
}
#if !defined(AERO_EULERSOLVER_KERNEL_ONLY)
// Вычисляет поток Роу по консервативным состояниям двух сторон.
EulerFluxResult roe_conservative(const std::vector<double>& left, const std::vector<double>& right,
                                 const std::vector<double>& direction, double gamma);
// Вычисляет поток Роу с использованием заранее вычисленных физических потоков.
EulerFluxResult roe_fluxes(const std::vector<double>& fl, const std::vector<double>& fr,
                           const std::vector<double>& left, const std::vector<double>& right,
                           const std::vector<double>& direction, double gamma);
#endif
} // namespace aero
#endif
