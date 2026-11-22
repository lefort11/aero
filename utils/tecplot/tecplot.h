#ifndef AERO_TECPLOT_H
#define AERO_TECPLOT_H
#include "../../source/mesh/mesh.h"
#include "../../source/io/fields.h"
#include <iosfwd>
#include <stdexcept>
#include <string>

namespace aero {
namespace tecplot {
class ExportError : public std::runtime_error {
public:
    explicit ExportError(const std::string& message) : std::runtime_error(message) {}
};
struct Options {
    // Ноль выбирает стандартное число потоков OpenMP (OMP_NUM_THREADS). В последовательной сборке работает один поток.
    int threads;
    Options() : threads(0) {}
};
bool openmp_available() noexcept;
// Пишет двоичный Tecplot 360 PLT v112 младшим байтом вперёд, координаты в единицах сетки.
// До записи сверяет SHA-256, размеры и конечность значений. Описание приведено в README.txt.
void write(std::ostream& output, const Mesh& mesh, const FieldSnapshot& fields,
           const Options& options = Options());
// Для выхода требуется новый путь. При неудачной записи частичный файл удаляется.
void write(const std::string& path, const Mesh& mesh, const FieldSnapshot& fields,
           const Options& options = Options());
void write(const std::string& path, const Mesh& mesh, const Options& options = Options());
int run_cli(int argc, const char* const* argv, std::ostream& out, std::ostream& err);
}
}
#endif
