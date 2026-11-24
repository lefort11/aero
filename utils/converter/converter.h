#ifndef AERO_CONVERTER_H
#define AERO_CONVERTER_H
#include "../../source/mesh/mesh.h"
#include <istream>
namespace aero {
namespace converter {
// Выбор базы и зоны CGNS и имя немаркированной внешней границы.
struct Options {
    // Нумерация баз и зон CGNS начинается с единицы. Ноль разрешён, когда выбор однозначен.
    int base, zone;
    // Имя внешней грани без исходной метки. Пустая строка требует явной метки.
    std::string unmarked_name;
    // Задаёт автоматический выбор базы и зоны и стандартное имя немаркированной границы.
    Options() : base(0), zone(0), unmarked_name("unmarked") {}
};
// Импортирует текстовые и двоичные секции Gmsh из открытого потока.
Mesh read_gmsh(std::istream& input, const Options& options = Options());
// Читает сетку Gmsh и переводит её топологию во внутренний формат.
Mesh read_gmsh(const std::string& path, const Options& options = Options());
// Преобразует выбранную базу и зону CGNS в общую ориентированную сетку.
Mesh read_cgns(const std::string& path, const Options& options = Options());
// Принимает каталог сетки another или путь к mesh.txt. Поддерживаются обычные 2D/3D-сетки с текстовыми массивами .msh.
Mesh read_another(const std::string& path, const Options& options = Options());
// Сообщает, собран ли конвертер с библиотекой CGNS.
bool cgns_available() noexcept;
}
}
#endif
