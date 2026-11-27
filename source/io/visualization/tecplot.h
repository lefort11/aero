#ifndef AERO_TECPLOT_H
#define AERO_TECPLOT_H
#include "../../mesh/mesh.h"
#include "../fields.h"
#include <iosfwd>
#include <stdexcept>
#include <string>

namespace aero {
class BinaryColumns;
namespace tecplot {
// Ошибка подготовки или записи данных Tecplot.
class ExportError : public std::runtime_error {
public:
    // Создаёт исключение с переданным диагностическим сообщением.
    explicit ExportError(const std::string& message) : std::runtime_error(message) {}
};
// Число потоков экспорта Tecplot; ноль использует стандартную настройку OpenMP.
struct Options {
    // Ноль выбирает стандартное число потоков OpenMP (OMP_NUM_THREADS). В последовательной сборке работает один поток.
    int threads;
    // Выбирает стандартное число потоков экспорта.
    Options() : threads(0) {}
};
// Сообщает, собрана ли утилита с поддержкой OpenMP.
bool openmp_available() noexcept;
// Пишет двоичный Tecplot 360 PLT v112 младшим байтом вперёд, координаты в единицах сетки.
// До записи сверяет SHA-256, размеры и конечность значений. Описание приведено в README.txt.
void write(std::ostream& output, const Mesh& mesh, const FieldSnapshot& fields,
           const Options& options = Options());
// Готовый результат заменяет прежний файл. При ошибке незаконченный временный файл удаляется.
void write(const std::string& path, const Mesh& mesh, const FieldSnapshot& fields,
           const Options& options = Options());
// Проверяет входные данные и экспортирует сетку или снимок в Tecplot.
void write(const std::string& path, const Mesh& mesh, const Options& options = Options());
// Записывает уже связанный с геометрией снимок решателя. Пустой указатель на сетку означает облако Cp.
void write_snapshot(std::ostream& out, const Mesh* mesh, const FieldSnapshot& snapshot, BinaryColumns* columns = nullptr);
}
}
#endif
