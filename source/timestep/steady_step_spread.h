#pragma once
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace aero {

// Ограничивает dt_max/dt_min для стационарной задачи. Норма и число итераций
// относятся ко всему расчёту, поэтому каждый ранг MPI получит один предел.
class SteadyStepSpread {
public:
    // Первое поле выбранной схемы задаёт масштаб невязки. Его подготовка
    // не является принятым шагом. При продолжении берём масштаб из recovery.
    void start(double norm) {
        if (!std::isfinite(norm) || norm < 0.) return;
        if (reference_ < 0. || (reference_ == 0. && norm > 0.)) reference_ = norm;
        if (last_ < 0.) last_ = norm;
    }

    // После перехода LO -> EBR наблюдаем полную невязку уже нового оператора.
    // Сбрасываем и незавершённый блок, и базу для сравнения затрат BiCGSTAB.
    void reset() { *this = SteadyStepSpread(); }

    // Возвращает опорное значение статистики разброса локальных шагов.
    double reference() const { return reference_; }
    // Возвращает последнюю оценку разброса локальных шагов.
    double last() const { return last_; }
    // Возвращает число накопленных измерений разброса.
    unsigned samples() const { return samples_; }

    // Ручной предел из настроек служит дополнительной верхней границей.
    double limit(double user_ratio = 0.) const {
        return user_ratio >= 1. ? std::min(range_, user_ratio) : range_;
    }

    // Вызывать один раз после принятия шага. Передаётся сумма итераций BiCGSTAB
    // всех поправок Ньютона. Предиктор без СЛАУ тоже стоит одну единицу работы.
    void accepted(double norm, unsigned linear_iterations) {
        if (!std::isfinite(norm) || norm < 0.) return;
        start(norm);
        last_ = norm;
        if (norm == 0.) return; // Точная сходимость. Логарифм нуля не определён.

        const double log_relative = log_ratio(norm, reference_);
        if (samples_ < half_block) first_sum_ += log_relative;
        else second_sum_ += log_relative;
        work_ += 1. + linear_iterations;
        if (++samples_ != block_size) return;

        // Разность средних логарифмов даёт логарифм отношения геометрических
        // средних. Множитель 2 пересчитывает снижение на длину целого блока.
        const double gain = 2. * (first_sum_ - second_sum_) / half_block;
        if (gain <= 0.) {
            // При застое или росте невязки приблизим диапазон к единице.
            range_ = std::sqrt(range_);
        } else {
            const double cost_factor = previous_work_ > 0. ? std::min(1., previous_work_ / work_) : 1.;
            // min(1/relative, range*exp(gain*cost_factor)) вычисляем в
            // логарифмах. Оба отдельных множителя могут выйти за предел double.
            const double log_range = std::min(-log_relative, std::log(range_) + gain * cost_factor);
            const double largest = std::numeric_limits<double>::max();
            if (log_range <= 0.) range_ = 1.;
            else if (log_range >= std::log(largest)) range_ = largest;
            else range_ = std::exp(log_range);
        }

        previous_work_ = work_;
        samples_ = 0;
        first_sum_ = second_sum_ = work_ = 0.;
    }

    // Записывает только данные после имени схемы. Точность max_digits10
    // обеспечивает точное восстановление даже между двумя половинами блока.
    void write(std::ostream& out) const {
        std::ostringstream record;
        record.imbue(std::locale::classic());
        record << std::setprecision(std::numeric_limits<double>::max_digits10)
               << "work1 " << reference_ << ' ' << last_ << ' ' << range_ << ' ' << samples_
               << ' ' << first_sum_ << ' ' << second_sum_ << ' ' << work_ << ' ' << previous_work_;
        out << record.str();
    }

    // Читает одну запись из потока recovery с классической числовой локалью.
    // Старый формат содержал одно значение reference. В этом случае первый
    // новый блок начнётся с K=1. Следующий параметр recovery остаётся в потоке.
    void read(std::istream& in) {
        SteadyStepSpread restored;
        std::string first;
        if (!(in >> first)) invalid_record();
        if (first == "work1") {
            if (!(in >> restored.reference_ >> restored.last_ >> restored.range_ >> restored.samples_
                     >> restored.first_sum_ >> restored.second_sum_ >> restored.work_ >> restored.previous_work_)
                || !restored.valid()) invalid_record();
        } else {
            std::istringstream value(first);
            value.imbue(std::locale::classic());
            if (!(value >> restored.reference_) || !value.eof() || !std::isfinite(restored.reference_)
                || restored.reference_ < 0.) invalid_record();
        }
        // При повреждённой или неполной записи рабочее состояние не меняется.
        *this = restored;
    }

private:
    // Размер блока сбора статистики и его половины.
    enum { half_block = 16, block_size = 2 * half_block };
    double reference_ = -1.; // Исходная пространственная норма этой реконструкции.
    double last_ = -1.;      // Норма последнего измеренного поля.
    double range_ = 1.;      // Действующий предел до учёта ручного ограничения.
    unsigned samples_ = 0;   // Число принятых ненулевых измерений внутри блока.
    double first_sum_ = 0.;  // Сумма логарифмов относительной нормы в первых 16 шагах.
    double second_sum_ = 0.; // Та же сумма во вторых 16 шагах.
    double work_ = 0.;       // Число принятых шагов плюс сумма итераций СЛАУ.
    double previous_work_ = 0.; // Стоимость последнего полного блока, ноль при старте.

    // Разлагаем числа на мантиссу и степень двойки: отношение крайних конечных
    // значений double может округлиться до нуля либо переполниться. Для близких
    // чисел берём один логарифм, сохраняя точность при норме около опорной.
    static double log_ratio(double norm, double reference) {
        int n, r;
        const double mantissa = std::frexp(norm, &n) / std::frexp(reference, &r);
        const int shift = n - r;
        return std::abs(shift) <= 1 ? std::log(std::ldexp(mantissa, shift))
                                   : std::log(mantissa) + shift * std::log(2.);
    }

    // Проверяем достижимую величину каждого накопителя. Тогда следующее
    // наблюдение не переполнится даже после чтения повреждённого recovery.
    bool valid() const {
        if (samples_ >= block_size || !std::isfinite(reference_) || !std::isfinite(last_) ||
            !std::isfinite(range_) || range_ < 1.) return false;
        if ((reference_ < 0. && reference_ != -1.) || (last_ < 0. && last_ != -1.)) return false;
        if ((reference_ < 0. || last_ < 0.) && (range_ != 1. || samples_ != 0 || previous_work_ != 0.)) return false;
        if (reference_ < 0. && last_ != -1.) return false;
        if (reference_ == 0. && (last_ > 0. || samples_ != 0 || previous_work_ != 0. || range_ != 1.)) return false;
        const unsigned first_count = std::min<unsigned>(half_block, samples_);
        const unsigned second_count = samples_ - first_count;
        const double max_log = std::ceil(log_ratio(std::numeric_limits<double>::max(),
                                                   std::numeric_limits<double>::denorm_min()));
        const double max_work = 1. + std::numeric_limits<unsigned>::max();
        return std::abs(first_sum_) <= first_count * max_log && std::abs(second_sum_) <= second_count * max_log
            && std::floor(work_) == work_ && work_ >= samples_ && work_ <= samples_ * max_work
            && (previous_work_ == 0. || (std::floor(previous_work_) == previous_work_ &&
                    previous_work_ >= block_size && previous_work_ <= block_size * max_work));
    }

    // Создаёт признак отсутствующего или недопустимого измерения.
    static void invalid_record() { throw std::runtime_error("Invalid checkpoint for steady local time steps"); }
};

} // namespace aero
