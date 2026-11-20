#ifndef AERO_RUN_CONTROL_H
#define AERO_RUN_CONTROL_H
#include "../io/settings.h"
#include <iosfwd>

namespace aero {
class Processes;
// Сначала разбираем весь файл в отдельную копию. Ошибка в последней строке отменяет весь набор команд.
struct RunControl {
    Stabilization limits;
    double cfl, upwind, hybrid;
    bool steady, local_time_step;
    double local_time_step_max_ratio;
    bool change_steady_steps = false; // Применяем обе настройки вместе после принятия шага.
    bool reset = false, stop = false, change_cfl = false;
    RunControl(const RunSettings& run, double current)
        : limits(run.stabilization), cfl(current), upwind(run.upw_coeff), hybrid(run.mp6_hybrid),
          steady(run.steady()), local_time_step(run.local_time_step), local_time_step_max_ratio(run.local_time_step_max_ratio) {}
    bool read(const std::string& directory, const Processes& world);
    void report_accepted(std::ostream& out) const;
private:
    std::vector<std::string> accepted_; // Подтверждения выводятся только после применения всего набора.
    void parse(const std::string& commands, std::ostream* warnings);
};
}
#endif
