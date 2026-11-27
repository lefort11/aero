#pragma once
#include "../fields.h"

namespace aero {
class BinaryColumns;
namespace vtk {
// Записывает сетку и поля в VTK Legacy 4.2. Вызывающий уже проверил привязку снимка.
// При mesh == nullptr координаты берутся из стеночного снимка Cp.
void write(std::ostream& out, const Mesh* mesh, const FieldSnapshot* snapshot, BinaryColumns* columns = nullptr);
}
}
