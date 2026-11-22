#ifndef AERO_INITIAL_H
#define AERO_INITIAL_H
#include "../parallel/row_table.h"
#include "../mesh/mesh_index.h"
#include "../eulersolver/linear_euler.h"
#include <array>
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

// Возвращает состояние каждого узла. Для Euler это плотность, импульс и энергия,
// для Linear — пульсации плотности, скорости и давления.
// Относительные пути из настроек отсчитываются от directory. Периодическое отображение
// позволяет склеить копии узлов и проверить согласованность их начальных состояний.
// Для режима recovery возвращает пустой вектор и передаёт путь солверу через recovery_path.
std::vector<double> read_initial(const Mesh& mesh, double gamma, std::istream& settings,
                                 const std::string& directory = ".", const PeriodicView* periodic = nullptr,
                                 const FieldNodeSelection* selection = nullptr,
                                 const LinearBackground& background = {}, std::vector<std::string>* warnings = nullptr,
                                 double mach = 0., std::string* recovery_path = nullptr);

// Возвращает эталонные rho, ux, uy, uz, p, E в порядке переданных локальных номеров узлов.
RowTable<double,6> read_exact_fields(const Mesh&, double gamma, std::istream&,
        const std::string& directory, const FieldNodeSelection*, const LinearBackground&,
        const std::vector<MeshIndex>& nodes, std::vector<std::string>* warnings = nullptr);

// Разрешает пути к полям относительно каталога файла начальных условий.
std::vector<double> read_initial_file(const Mesh& mesh, double gamma, const std::string& settings_path,
                                      const PeriodicView* periodic = nullptr, const LinearBackground& background = {}, double mach = 0.);
}
#endif
