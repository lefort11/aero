#pragma once
#include "../mesh/mesh_index.h"
#include "../mesh/compact_table.h"

namespace aero {
namespace residual_detail {
// По внутренней поверхности поток выходит из begin в end. Физической площадке
// принадлежит только узел begin и номер метки patch.
struct Surface {
    // Номера концов и граничного патча площадки.
    MeshIndex begin, end, patch;
    double area[3]; // Вектор площади в координатах решателя.
    double measure; // Длина этого вектора.
    unsigned char use_nodal_traces; // Маска перехода на первый порядок для сторон ребра.
};
// Четыре double остаются без изменений; номера и маска занимают ещё 16 байт.
struct Surface32 {
    // Компоненты ориентированной площади и её мера в двойной точности.
    double area[3],measure;
    // Упакованные номера концов и патча площадки.
    std::uint32_t begin,end,patch;
    // Маска сторон с узловым значением вместо реконструкции.
    unsigned char use_nodal_traces;
    // Тип записи после раскрытия упакованных индексов.
    using View=Surface;
    // Проверяет все индексы перед сужением, включая номер физической метки.
    static bool fits(const Surface& s) { return s.begin<=UINT32_MAX && s.end<=UINT32_MAX && s.patch<=UINT32_MAX; }
    // Копирует геометрию точно, сохраняя порядок компонент площади.
    static Surface32 pack(const Surface& s) {
        return {{s.area[0],s.area[1],s.area[2]},s.measure,static_cast<std::uint32_t>(s.begin),
                static_cast<std::uint32_t>(s.end),static_cast<std::uint32_t>(s.patch),s.use_nodal_traces};
    }
    // Исходный формат остаётся доступен для сеток с большими номерами.
    AERO_HOST_DEVICE static Surface view(const Surface& s) { return s; }
    // Расширяет только номера; арифметика потока получает прежние значения.
    AERO_HOST_DEVICE static Surface view(const Surface32& s) {
        return {s.begin,s.end,s.patch,{s.area[0],s.area[1],s.area[2]},s.measure,s.use_nodal_traces};
    }
};
using SurfaceTable=CompactTable<Surface,Surface32>;
}
}
