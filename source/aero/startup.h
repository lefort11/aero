#ifndef AERO_STARTUP_H
#define AERO_STARTUP_H
#include "../io/settings.h"
#include "../mesh/mesh.h"

namespace aero {
// Печатает параметры, по которым фактически запускается расчёт.
void show_settings(const RunSettings&);
// Выводит имена участков сетки и число помеченных граней каждого участка.
void show_boundary_labels(const Mesh&);
// При первом запуске создаёт заготовку секций по меткам сетки. Возвращает false, если её нужно заполнить.
bool prepare_boundary_file(const RunSettings&, const std::vector<std::string>& names);
// Создаёт также отсутствующие родительские каталоги пути.
void create_output_directory(std::string path);
// Готовит каталог расчёта и подкаталоги запрошенных зон вывода.
void prepare_output_directories(const RunSettings&);
// Готовит каталоги только для зон заданного размещения: общих или локальных.
void prepare_zone_directories(const std::string& directory, const std::vector<OutputZone>& zones,
                              bool global);
// Выделяет новый пронумерованный каталог продолжения, чтобы отличать результаты разных перезапусков.
unsigned create_resume_directory(const std::string& prefix);
// Возвращает непустые строки настроек для вывода постановки в журнал.
std::string settings_text(const std::string& path);
void stage(const char* name);
void done();
}
#endif
