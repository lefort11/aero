#pragma once
#include <algorithm>
#include <cmath>

namespace aero {
namespace multigrid {
// Коэффициенты трёхчленной итерации Чебышёва для точечно предобусловленного
// уравнения SA. Узлу нужны только текущая обычная поправка и последнее изменение.
class ScalarSmoother {
public:
    // Возвращает коэффициент следующей итерации скалярного сглаживания.
    void next(double initial_norm,double current_norm,unsigned ordinary_iterations) {
        if (!iteration_) {
            // Для x <- x − D⁻¹F с наблюдаемым множителем затухания q
            // медленной моде соответствует собственное число около 1−q.
            const double ratio = std::pow(current_norm/initial_norm,1./std::max(1u,ordinary_iterations));
            const double lower = std::max(1./128.,std::min(.5,.5*(1.-ratio)));
            center_ = .5*(2.+lower);
            radius_ = .5*(2.-lower);
            current = 1./center_;
            previous = 0.;
            reference_ = current_norm;
        } else {
            // На второй итерации у T₁ один предшественник, дальше коэффициент
            // при предыдущем изменении определяется обычной рекурсией Tₖ.
            const double weight = radius_*radius_*current*(iteration_ == 1 ? .5 : .25);
            current = 1./(center_-weight);
            previous = current*weight;
        }
        ++iteration_;
    }
    // На несимметричной или существенно нелинейной строке оценка спектра
    // может оказаться неточной. Тогда возвращаем обычную точечную поправку.
    bool accepts(double norm) const { return std::isfinite(norm) && norm <= reference_; }
    // Сбрасывает последовательность коэффициентов сглаживания.
    void restart() { iteration_ = 0; }
    double current = 0.; // Коэффициент при новой поправке −D⁻¹F.
    double previous = 0.; // Коэффициент при предыдущем изменении рабочей вязкости.
private:
    unsigned iteration_ = 0;
    double center_ = 0., radius_ = 0.; // Середина и полуразмах оценки спектра.
    double reference_ = 0.;            // Норма полной невязки до начала полиномиальных итераций.
};
}
}
