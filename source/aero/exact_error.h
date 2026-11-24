#ifndef AERO_EXACT_ERROR_H
#define AERO_EXACT_ERROR_H
#include "../parallel/row_table.h"
#include "mpi_periodic.h"
#include "../io/expression.h"
#include "../io/settings.h"
#include <fstream>

namespace aero {
struct FieldNodeSelection;
// Сравнивает расчёт с формулами во времени или с узловыми полями одной сетки.
class ExactError {
public:
    // Подготавливает эталон и объёмные веса для вычисления ошибки решения.
    ExactError(const RunSettings&, const Processes&, const MeshPartition&, const PeriodicPartition*,
               const FieldNodeSelection&, const double* volumes, const std::string& directory);
    // Вычисляет глобальные нормы ошибки относительно эталона и записывает их для текущего шага.
    void write(MeshIndex step, double time, const std::vector<double>& state);
private:
    // Координаты представителя узла, строка численного поля и корень объёмного веса.
    struct Node {
        Coordinate point; // Координаты представителя периодического узла.
        MeshIndex slot;   // Строка в массиве численного решения.
        double root_volume; // Вес для евклидовой нормы ошибки.
    };
    // У снимка значения постоянны во времени. Формулы вычисляем в момент текущего отчёта.
    std::array<double, 6> reference(std::size_t i, double time) const;
    // Число компонент скорости в сравниваемых полях.
    unsigned dimension_;
    // Настройки эталона и модели газа для вычисления ошибки.
    const RunSettings& run_;
    // Объединяет локальные нормы ошибки по процессам.
    const Processes& world_;
    // Узлы, по которым этот процесс накапливает ошибку.
    std::vector<Node> nodes_;
    // Порядок полей в отчёте об ошибке.
    std::vector<std::string> names_;
    // Разобранные выражения эталона в порядке полей.
    std::vector<Expression> formulas_;
    RowTable<double,6> fields_; // Эталон в порядке nodes_, читаем один раз при старте.
    std::ofstream file_; // Только процесс 0 пишет общий отчёт.
};
}
#endif
