#pragma once
#include "../parallel/kernel_math.h"
#include "../mesh/mesh_index.h"

namespace aero {
// У каждого владельца три числа: средние невязки газа и SA и доля местного CFL.
// Обновление выполняет тот же поток CPU/GPU, который выбирает шаг этого узла.
struct LocalRateLimit {
    enum Slot { GasMean, SaMean, CflFraction, Width }; // Три соседних double принадлежат одному узлу.
    static constexpr unsigned interval = 32;
    double* history;
    const double* gas_rate; // max_k |R_ik| / V_i в компактном порядке собственных строк.
    const double* sa_rate;  // |d(nu_tilde)/dt| берётся по номеру узла в полном поле.
    double gas_max, sa_max, previous_minimum;
    bool review; // История обновляется каждый принятый шаг, потолок — раз за interval шагов.

    const double* stencil_growth; // Признаки роста газа и SA по всей области реконструкции.
    unsigned stencil_stride; // При выводе диагностики следом идут номера двух узлов.
    static bool review_due(std::uint64_t samples) {
        return samples != 0 && samples % interval == 0;
    }

    AERO_HOST_DEVICE bool has_cap(MeshIndex row) const {
        return history && history[Width*row + CflFraction] < 1.;
    }

    AERO_HOST_DEVICE static bool rising(double rate, double mean, double domain_max) {
        // Не сравниваем относительные изменения почти нулевой невязки.
        return rate > .01*domain_max && rate > 1.05*mean;
    }

    AERO_HOST_DEVICE double sa_magnitude(MeshIndex node) const {
        return sa_rate ? ::fabs(sa_rate[node]) : 0.;
    }

    AERO_HOST_DEVICE double limit(MeshIndex row, MeshIndex node, double cfl_step,
                                  double previous_dt, double proposed, bool initialize) const {
        if (!history || !(proposed > 0.)) return proposed;
        double* entry = history + Width*row;
        double& gas_mean = entry[GasMean];
        double& sa_mean = entry[SaMean];
        double& fraction = entry[CflFraction];
        // Среднее всегда принадлежит одному объёму. В режиме stencil признаки
        // роста его соседей уже собраны отдельно, до включения текущего слоя в историю.
        const double gas = gas_rate[row];
        const double sa = sa_magnitude(node);
        if (initialize) {
            gas_mean = gas;
            sa_mean = sa;
            fraction = 1.;
        } else {
            if (review) {
                // Владелец сначала проверяет каждую строку на её собственном среднем.
                // Для stencil к этому моменту по MPI уже собраны признаки всей области.
                const double* patch = stencil_growth ? stencil_growth + stencil_stride*node : nullptr;
                const bool growth = patch ? (patch[0] > 0. || patch[1] > 0.)
                                          : (rising(gas,gas_mean,gas_max) || rising(sa,sa_mean,sa_max));
                if (growth) {
                    // Отсчитываем уменьшение от реально принятого шага. Узел с
                    // минимальным шагом не вынуждает всю сетку бесконечно замедляться.
                    const double cap = ::fmax(previous_minimum,.75*previous_dt);
                    fraction = ::fmin(fraction, cap/cfl_step);
                } else {
                    // В спокойной области возвращаем разрешённую долю CFL постепенно.
                    fraction = ::fmin(1.,1.25*fraction);
                }
            }
            gas_mean += (gas-gas_mean)/interval;
            sa_mean += (sa-sa_mean)/interval;
        }
        return ::fmin(proposed,cfl_step*fraction);
    }
};
}
