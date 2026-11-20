#ifndef AERO_RUN_CONTROL_H
#define AERO_RUN_CONTROL_H
#include "../io/settings.h"

namespace aero {
class Processes;
// Сначала разбираем весь файл в отдельную копию. Ошибка в последней строке отменяет весь набор команд.
struct RunControl {
    Stabilization limits;
    double cfl, upwind, hybrid;
    bool reset = false, stop = false, change_cfl = false;
    RunControl(const RunSettings& run, double current)
        : limits(run.stabilization), cfl(current), upwind(run.upw_coeff), hybrid(run.mp6_hybrid) {}
    bool read(const std::string& directory, const Processes& world);
private:
    void parse(const std::string& commands);
};
}
#endif
