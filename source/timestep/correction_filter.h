#pragma once
#include "../parallel/kernel_math.h"

namespace aero {
// Общая двухчленная релаксация нелинейной поправки газа и SA. История всегда
// относится к одному решению при фиксированном временном шаге.
struct CorrectionFilter {
    // Определяет момент включения фильтра по двум полным нормам.
    AERO_HOST_DEVICE static bool slow(double current,double before) { return current>.5*before; }
    // Вес текущей диагональной или предобусловленной поправки.
    AERO_HOST_DEVICE static double weight() { return .65; }
    // Вес фактически принятого изменения предыдущей итерации.
    AERO_HOST_DEVICE static double history_weight() { return -.15; }
    // Первая фильтрованная итерация получает нулевое предыдущее изменение.
    AERO_HOST_DEVICE static double change(double delta,double previous) {
        return weight()*delta+history_weight()*previous;
    }
    // Представление предыдущего принятого изменения в единицах сырой поправки:
    // weight()*(delta+shift(previous)) даёт тот же двухчленный фильтр.
    AERO_HOST_DEVICE static double shift(double previous) { return (history_weight()/weight())*previous; }
};
}
