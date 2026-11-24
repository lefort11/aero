#ifndef AERO_RUN_CONTROL_H
#define AERO_RUN_CONTROL_H
#include "../io/settings.h"
#include <iosfwd>

namespace aero {
class Processes;
// Процесс 0 читает очередной набор команд и освобождает файл для следующего.
class ControlFile {
public:
    // Создаёт control.txt без удаления заранее сохранённых команд.
    void open(const std::string& directory);
    // Считывает команды и очищает тот же файл для однократного исполнения.
    std::string read_and_clear() const;
private:
    // Путь к файлу команд в каталоге вывода.
    std::string path_;
};
// Сначала разбираем весь файл в отдельную копию. Ошибка в последней строке отменяет весь набор команд.
struct RunControl {
    // Параметры регулятора CFL после применения команд.
    Stabilization limits;
    // Текущие CFL, вес противопоточности и вес гибридной схемы.
    double cfl, upwind, hybrid;
    // Разрешает команды стационарного режима.
    bool steady;
    // Режим выбора узлового шага.
    LocalStepMode local_time_step;
    // Верхняя граница разброса узловых шагов.
    double local_time_step_max_ratio;
    bool change_steady_steps = false; // Применяем обе настройки вместе после принятия шага.
    // Запросы сброса меток, остановки и изменения CFL.
    bool reset = false, stop = false, change_cfl = false;
    // Создаёт изменяемую копию текущих параметров для разбора команд управления.
    RunControl(const RunSettings& run, double current)
        : limits(run.stabilization), cfl(current), upwind(run.upw_coeff), hybrid(run.mp6_hybrid),
          steady(run.steady()), local_time_step(run.local_time_step), local_time_step_max_ratio(run.local_time_step_max_ratio) {}
    // Читает команды на корневом ранге и распространяет результат по MPI.
    bool read(const ControlFile& file, const Processes& world);
    // Печатает подтверждение применённых параметров управления.
    void report_accepted(std::ostream& out) const;
private:
    std::vector<std::string> accepted_; // Подтверждения выводятся только после применения всего набора.
    // Разбирает команды, проверяет значения и сохраняет список принятых изменений.
    void parse(const std::string& commands, std::ostream* warnings);
};
}
#endif
