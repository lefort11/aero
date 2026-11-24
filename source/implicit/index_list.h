#pragma once
#include "../mesh/mesh_index.h"
#include "../parallel/kernel_math.h"

namespace aero {
namespace implicit {
// Граф строится с 64-разрядными номерами. Видеокарта хранит короткий массив,
// только если наибольшее записываемое значение помещается в 32 бита.
// Сложение и умножение адресов в ядрах по-прежнему выполняются в MeshIndex.
struct IndexList {
    const MeshIndex* wide;
    const std::uint32_t* narrow;

    // Подключает массив индексов нужной разрядности без копирования данных.
    AERO_HOST_DEVICE IndexList(const MeshIndex* values = nullptr) : wide(values), narrow(nullptr) {}
    // Подключает массив индексов нужной разрядности без копирования данных.
    AERO_HOST_DEVICE explicit IndexList(const std::uint32_t* values) : wide(nullptr), narrow(values) {}
    // Проверяет наличие подключённого массива индексов.
    AERO_HOST_DEVICE explicit operator bool() const { return narrow || wide; }
    // Возвращает элемент по индексу; допустимость индекса обеспечивает вызывающий код.
    AERO_HOST_DEVICE MeshIndex operator[](MeshIndex i) const { return narrow ? narrow[i] : wide[i]; }
};
}
}
