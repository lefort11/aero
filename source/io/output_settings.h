#ifndef AERO_IO_OUTPUT_SETTINGS_H
#define AERO_IO_OUTPUT_SETTINGS_H

#include "fields.h"

namespace aero {
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
    // Кодирование данных зоны при записи на диск.
    FieldFormat format = FieldFormat::Binary;
};

// Отсутствие output.txt отключает запись по расписанию и позволяет читать старые постановки.
std::vector<OutputZone> read_output_zones(const std::string& path, std::vector<std::string>* warnings = nullptr);
// Проверяет расписание, допустимые имена полей и пересечения выходных путей.
void validate_output_zones(const std::vector<OutputZone>& zones);
std::string output_filename(const OutputZone& zone, std::uint64_t step);
// Нормализует имя для проверки, могут ли две зоны записать в один файл.
std::string output_path_key(std::string file);
}
#endif
