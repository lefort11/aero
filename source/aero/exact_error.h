#ifndef AERO_EXACT_ERROR_H
#define AERO_EXACT_ERROR_H
#include "mpi_periodic.h"
#include "../io/expression.h"
#include "../io/settings.h"
#include <fstream>

namespace aero {
// Точное решение задано в примитивных переменных выбранной системы уравнений.
class ExactError {
public:
    ExactError(const RunSettings&, const Processes&, const MeshPartition&, const PeriodicPartition*,
               const double* volumes, const std::string& directory);
    void write(MeshIndex step, double time, const std::vector<double>& state);
private:
    struct Node {
        Coordinate point; // Координаты представителя периодического узла.
        MeshIndex slot;   // Строка в массиве численного решения.
        double root_volume; // Вес для евклидовой нормы ошибки.
    };
    unsigned dimension_;
    const RunSettings& run_;
    const Processes& world_;
    std::vector<Node> nodes_;
    std::vector<std::string> names_;
    std::vector<Expression> formulas_;
    std::ofstream file_; // Только процесс 0 пишет общий отчёт.
};
}
#endif
