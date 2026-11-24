#pragma once
#include <algorithm>
#include <stdexcept>

namespace aero {
namespace multigrid {
// Между пробами обычный решатель сглаживает изменения газа и обновляет SA.
// Сохраняем только два счётчика: все сеточные данные строятся заново из поля.
class CycleSchedule {
public:
    // Проверяет, наступил ли срок очередной грубосеточной попытки.
    bool due() const { return remaining_ == 0; }
    // Учитывает принятый обычный шаг в расписании грубых коррекций.
    void ordinary_step() { --remaining_; } // Вызывается только между пробами.
    // Обновляет расписание после проверки эффективности грубосеточной коррекции.
    void tested(bool accepted) {
        // Три мелкосеточных шага после удачной поправки. При повторных
        // отказах даём полю измениться сильнее перед следующей попыткой.
        interval_ = accepted ? std::max(4u,interval_/2) : std::min(32u,2*std::max(4u,interval_));
        remaining_ = interval_-1;
    }
    // Возвращает текущий интервал между грубосеточными попытками.
    unsigned interval() const { return interval_; }
    // Возвращает число обычных шагов до следующей попытки.
    unsigned remaining() const { return remaining_; }
    // Восстанавливает состояние расписания из сохранённых счётчиков.
    void restore(unsigned interval,unsigned remaining) {
        if (interval == 0 || interval > 32 || remaining >= interval)
            throw std::runtime_error("Invalid checkpoint multigrid cycle");
        interval_ = interval;
        remaining_ = remaining;
    }
private:
    unsigned interval_ = 1;  // Расстояние между пробами в принятых псевдошагах.
    unsigned remaining_ = 0; // Сколько обычных шагов осталось до следующей пробы.
};
}
}
