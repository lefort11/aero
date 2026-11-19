#ifndef AERO_INITIAL_H
#define AERO_INITIAL_H
#include "../mesh/mesh_index.h"
#include <iosfwd>
#include <string>
#include <vector>

namespace aero {
class Mesh;
struct PeriodicView;
struct FieldNodeSelection;
// Сравнивает консервативные компоненты копии с представителем шва в пределах абсолютного и относительного допусков.
void check_periodic_initial(const double* reference, const double* copy, unsigned width, MeshIndex owner,
                            MeshIndex node);

// Возвращает для каждого расчётного узла подряд плотность, компоненты импульса и полную энергию.
// Относительные пути из настроек отсчитываются от directory. Периодическое отображение
// позволяет склеить копии узлов и проверить согласованность их начальных состояний.
std::vector<double> read_initial(const Mesh& mesh, double gamma, std::istream& settings,
                                 const std::string& directory = ".", const PeriodicView* periodic = nullptr,
                                 const FieldNodeSelection* selection = nullptr);

// Разрешает пути к полям относительно каталога файла начальных условий.
std::vector<double> read_initial_file(const Mesh& mesh, double gamma, const std::string& settings_path,
                                      const PeriodicView* periodic = nullptr);
}
#endif
