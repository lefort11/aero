#ifndef AERO_EXACT_ERROR_H
#define AERO_EXACT_ERROR_H
#include "mpi_periodic.h"
#include "../io/expression.h"
#include "../io/settings.h"
#include <fstream>

namespace aero {
struct FieldNodeSelection;
// Сравнивает расчёт с формулами во времени или с узловыми полями одной сетки.
class ExactError {
public:
    ExactError(const RunSettings&, const Processes&, const MeshPartition&, const PeriodicPartition*,
               const FieldNodeSelection&, const double* volumes, const std::string& directory);
    void write(MeshIndex step, double time, const std::vector<double>& state);
private:
    struct Node {
        Coordinate point; // Координаты представителя периодического узла.
        MeshIndex slot;   // Строка в массиве численного решения.
        double root_volume; // Вес для евклидовой нормы ошибки.
    };
    std::array<double, 6> reference(std::size_t i, double time) const;
    unsigned dimension_;
    const RunSettings& run_;
    const Processes& world_;
    std::vector<Node> nodes_;
    std::vector<std::string> names_;
    std::vector<Expression> formulas_;
    std::vector<std::array<double, 6>> fields_; // Эталон в порядке nodes_, читаем один раз при старте.
    std::ofstream file_; // Только процесс 0 пишет общий отчёт.
};
}
#endif
