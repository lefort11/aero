#pragma once
#include "mesh_index.h"
#include "../parallel/kernel_math.h"
#include <cstdint>
#include <cstddef>

namespace aero {
// Нумерация на CPU остаётся 64-битной. Короткий формат выбирается по данным,
// поэтому одна сетка имеет одинаковую раскладку на разных архитектурах GPU.
template<class Index> struct IndexArray {
    // Полноразмерные номера, если таблица не упакована.
    const Index* wide;
    // Короткие номера, если весь массив помещается в 32 бита.
    const std::uint32_t* narrow;
    // Подключает исходный либо упакованный массив без владения его памятью.
    AERO_HOST_DEVICE IndexArray(const Index* p=nullptr) : wide(p),narrow(nullptr) {}
    // Подключает проверенные 32-битные номера.
    AERO_HOST_DEVICE explicit IndexArray(const std::uint32_t* p) : wide(nullptr),narrow(p) {}
    // Однозначно задаёт пустой массив при инициализации общих описаний.
    AERO_HOST_DEVICE IndexArray(std::nullptr_t) : wide(nullptr),narrow(nullptr) {}
    // Проверяет наличие любого из двух форматов.
    AERO_HOST_DEVICE explicit operator bool() const { return wide || narrow; }
    // Расширение индекса точно; дальнейшая адресная арифметика использует Index.
    AERO_HOST_DEVICE Index operator[](MeshIndex i) const { return narrow ? narrow[i] : wide[i]; }
};

// Запись геометрии может возвращать значение либо представление её массивов.
// Представление Cell не копирует восемь узлов и коэффициентов в локальную память потока.
template<class Wide,class Packed> struct CompactTable {
    // Исходные записи с полноразмерными индексами.
    const Wide* wide;
    // Записи с упакованными индексами.
    const Packed* narrow;
    // Тип результата доступа, общий для обоих форматов.
    using View=typename Packed::View;
    // Подключает один из форматов; неявное преобразование сохраняет CPU-интерфейс.
    AERO_HOST_DEVICE CompactTable(const Wide* p=nullptr) : wide(p),narrow(nullptr) {}
    // Подключает упакованные записи после проверки диапазона номеров.
    AERO_HOST_DEVICE explicit CompactTable(const Packed* p) : wide(nullptr),narrow(p) {}
    // Устраняет неоднозначность пустого адреса между двумя форматами.
    AERO_HOST_DEVICE CompactTable(std::nullptr_t) : wide(nullptr),narrow(nullptr) {}
    // Проверяет наличие таблицы без разыменования её адреса.
    AERO_HOST_DEVICE explicit operator bool() const { return wide || narrow; }
    // Читаются только поля, действительно использованные вызывающим ядром.
    AERO_HOST_DEVICE View operator[](MeshIndex i) const { return narrow ? Packed::view(narrow[i]) : Packed::view(wide[i]); }
};
}
