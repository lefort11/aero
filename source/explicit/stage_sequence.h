#ifndef AERO_EXPLICIT_STAGE_SEQUENCE_H
#define AERO_EXPLICIT_STAGE_SEQUENCE_H
#include "explicit.h"
#include "explicit_stage.h"
#include <string>

namespace aero {
namespace explicit_detail {
// Преобразует имя временной схемы в её внутреннее обозначение.
ExplicitScheme scheme_from_name(const std::string& name);
// Выбирает таблицу стадий указанной явной схемы.
Stages stages_for(ExplicitScheme scheme);

// Здесь хранится положение стадии внутри шага. Это определяет, когда сохранить U_n и когда готов отчёт.
struct StepStage {
    Stage formula;  // Время и веса текущей строки таблицы.
    unsigned index; // Номер стадии с нуля.
    unsigned count; // Число стадий выбранной схемы.
    // Проверяет, является ли текущая стадия первой в шаге.
    bool first() const { return index == 0; }
    // Проверяет, является ли текущая стадия последней в шаге.
    bool last() const { return index + 1 == count; }
    // Определяет необходимость сохранения начального слоя для следующих стадий.
    bool save_start_for_later() const { return first() && count > 1; }
};

enum class FirstResidual { NeedsAssembly, Prepared };

// Один порядок действий для обычного расчёта и MPI. assemble читает поле завершённой предыдущей стадии.
template <class Assemble, class Update, class Exchange>
void advance_stages(Stages stages, double time, double dt, FirstResidual first_residual,
                    const Assemble& assemble, const Update& update_stage, const Exchange& exchange) {
    for (unsigned i = 0; i < stages.count; ++i) {
        const StepStage stage = {stages.data[i], i, stages.count};
        // Первый поток мог быть вычислен для CFL до выбора dt. Последующие невязки всегда нужны заново.
        if (!stage.first() || first_residual == FirstResidual::NeedsAssembly) {
            assemble(time + stage.formula.time_fraction * dt);
        }
        update_stage(stage);
        // Соседям нужно новое состояние до вычисления следующей невязки и до вывода принятого шага.
        exchange();
    }
}
}
}
#endif
