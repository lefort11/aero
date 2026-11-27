#ifndef AERO_IO_OUTPUT_SETTINGS_H
#define AERO_IO_OUTPUT_SETTINGS_H

#include "fields.h"

namespace aero {
// Параметры файла, набора полей и расписания одной зоны вывода.
struct OutputZone {
    // Имя секции и путь к выходному файлу относительно каталога результатов.
    std::string name, file;
    // Порядок полей в снимке этой зоны.
    std::vector<std::string> variables;
    std::string wall_label; // Одна физическая метка для отдельного поверхностного файла Cp.
    std::string submesh; // Путь к геометрии среза и таблице интерполяции.
    // Интервал в принятых шагах. Для расписания по физическому времени используется every_time.
    std::uint64_t every_steps = 0;
    // Интервал снимков в единицах физического времени.
    double every_time = 0;
    // Режим серии файлов; в имени каждого снимка учитывается номер шага.
    bool multi = false;
    // В MPI общая зона записывается одним файлом в исходной нумерации, локальная — по разделам.
    bool global = false;
    // Дополнительные файлы визуализации публикуются рядом с основным снимком.
    bool tecplot = false; // Запрашивает двоичный файл Tecplot PLT.
    bool vtk = false;     // Запрашивает двоичный файл VTK Legacy.
    // Проверяет, нужна ли геометрия для записи дополнительных файлов.
    bool visualize() const { return tecplot || vtk; }
    // Кодирование данных зоны при записи на диск.
    FieldFormat format = FieldFormat::Binary;
};

// Отсутствие output.txt отключает запись по расписанию и позволяет читать старые постановки.
std::vector<OutputZone> read_output_zones(const std::string& path, std::vector<std::string>* warnings = nullptr);
// Проверяет расписание, допустимые имена полей и пересечения выходных путей.
void validate_output_zones(const std::vector<OutputZone>& zones);
// Формирует имя файла снимка по настройкам зоны и номеру шага.
std::string output_filename(const OutputZone& zone, std::uint64_t step);
// Заменяет последнее расширение, сохраняя каталог и номер шага.
std::string visualization_filename(const std::string& name, const char* suffix);
// Нормализует имя для проверки, могут ли две зоны записать в один файл.
std::string output_path_key(std::string file);
}
#endif
