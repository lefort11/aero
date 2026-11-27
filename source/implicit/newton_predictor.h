#pragma once
#include "implicit_settings.h"
#include "newton_vectors.h"
#include <functional>
#include <algorithm>
#include <cmath>

namespace aero { namespace implicit {
// Предиктор пробует поправку от временной диагонали либо от готового предобусловливателя.
// Каждая проба использует те же потоки, тот же обмен и ту же норму, что и Ньютон.
template<class Update, class Share, class Residual>
double predict_initial_field(double norm, double initial_norm, const NewtonSettings& settings, const Update& update,
                             const Share& share, const Residual& residual,
                             const std::function<void()>& refresh, const std::function<bool()>& trial_flux,
                             bool viscous, bool use_history, bool damp, const std::function<bool()>& confirm_prediction,
                             bool local_rans, unsigned limit) {
    bool filtered=false; // После медленной подстановки применяем общий двухчленный фильтр.
    bool have_history=false; // Первое фильтрованное изменение не читает поле предыдущей пробы.
    if (norm == 0.) return confirm_prediction && confirm_prediction() ? residual() : norm;
    // При быстрой сходимости последовательные подстановки дешевле сборки AES
    // и решения линейной системы. Для вязкого шага оставляем запас по допуску,
    // чтобы точность поля была близка к получаемой после поправки Ньютона.
    const double target_multiplier = local_rans ? 1. : (viscous ? .02 : (use_history ? 1. : 4.));
    const double target = initial_norm * std::min(.8, target_multiplier * settings.newton_relative);
    // На первом приближении уточняем связанные уравнения сразу. Затем проверяем их
    // на полпути к допуску по логарифмической шкале и обязательно перед принятием.
    const double coupled_reduction = confirm_prediction ? std::sqrt(settings.newton_relative) : 1.;
    double next_coupled_check = local_rans ? target : norm;
    // После медленной подстановки слегка демпфируем следующую пробу. Это гасит
    // знакочередующиеся пространственные моды, а невязка остаётся полной.
    double fraction = 1.;
    for (unsigned k = 0; k < limit && norm > target; ++k) {
        if (filtered) {
            update(NewtonUpdate::FilterTemporal,have_history ? 1. : 0.);
            have_history=true;
        } else {
            update(NewtonUpdate::Predict, 1.);
            if (fraction != 1.) update(NewtonUpdate::RelaxPrediction, fraction);
        }
        share();
        double trial = trial_flux() ? residual() : HUGE_VAL;
        if (settings.damp_correction && !filtered && std::isfinite(trial) && !(trial < .8*norm)) {
            // Первую демпфированную пробу строим между уже известными полями.
            update(NewtonUpdate::RelaxPrediction,CorrectionFilter::weight());
            filtered=have_history=true;
            share();
            trial=trial_flux() ? residual() : HUGE_VAL;
        } else if (!settings.damp_correction && (viscous || damp) && std::isfinite(trial) && !(trial < .8 * norm)) {
            // На жёстком вязком члене полная подстановка может перебросить поле
            // через корень. При противоположных знаках одной линейной моды
            // доля norm/(norm+trial) даёт нуль этой моды между двумя полями.
            // Для системы это только проба, её проверяем по полной невязке.
            const double relaxation = 1. / (1. + trial/norm);
            update(NewtonUpdate::RelaxPrediction, relaxation);
            share();
            trial = trial_flux() ? residual() : HUGE_VAL;
        }
        if (!(trial < .8 * norm)) {
            // После отката нужны и потоки, и правая часть для восстановленного поля.
            update(NewtonUpdate::UndoPrediction, 1.);
            share();
            refresh();
            return residual();
        }
        if (settings.damp_correction && CorrectionFilter::slow(trial,norm)) filtered=true;
        if (!settings.damp_correction && damp && trial > .2 * norm) fraction = .75;
        norm = trial;
        // Пока подбирается газовое приближение, SA может использовать последнее
        // рабочее поле. Перед принятием ещё раз проверяем оба уравнения совместно.
        if (confirm_prediction && (norm <= next_coupled_check || norm <= target)) {
            if (confirm_prediction()) norm = residual();
            next_coupled_check = coupled_reduction * norm;
        }
    }
    if (confirm_prediction && confirm_prediction()) norm = residual();
    return norm;
}
} }
