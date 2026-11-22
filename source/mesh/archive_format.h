#ifndef AERO_MESH_ARCHIVE_FORMAT_H
#define AERO_MESH_ARCHIVE_FORMAT_H

#include "mesh_index.h"

namespace aero {
namespace archive_format {
// Каталог AERODEC1 создаёт meshdec и читает солвер. Все смещения в файле измеряются в байтах.
constexpr char signature[] = "AERODEC1";
constexpr MeshIndex current_version = 5;
constexpr MeshIndex word_bytes = 8;
constexpr MeshIndex header_bytes = 16 * word_bytes;    // После заголовка идут имена граничных участков.
// В архивах v1–v4 одно дополнительное слово предшествовало числу узлов.
constexpr MeshIndex header_size(MeshIndex version) { return header_bytes + (version < 5 ? word_bytes : 0); }
constexpr MeshIndex leaf_entry_bytes = 6 * word_bytes; // Смещение, две длины и три счётчика элементов.
constexpr MeshIndex tree_entry_bytes =
    14 * word_bytes; // Восемь целых и шесть координат ограничивающего параллелепипеда.
}
}
#endif
