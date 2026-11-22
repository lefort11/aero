#pragma once
#include "kernel_math.h"

namespace aero {
// Матрица фиксированного размера размещается одним массивом в порядке строк.
// row() даёт начало строки для функций, обрабатывающих сразу все её компоненты.
template<class T, unsigned Rows, unsigned Columns>
struct FixedMatrix {
    T values[Rows * Columns];
    AERO_HOST_DEVICE T* row(unsigned i) { return values + i * Columns; }
    AERO_HOST_DEVICE const T* row(unsigned i) const { return values + i * Columns; }
    AERO_HOST_DEVICE T& operator()(unsigned i, unsigned j) { return row(i)[j]; }
    AERO_HOST_DEVICE const T& operator()(unsigned i, unsigned j) const { return row(i)[j]; }
};

// Последний индекс меняется быстрее остальных. Соседние исполнители GPU
// могут читать соседние значения без таблицы указателей или выделения строк.
template<class T, unsigned Layers, unsigned Rows, unsigned Columns>
struct FixedTensor {
    T values[Layers * Rows * Columns];
    AERO_HOST_DEVICE T& operator()(unsigned k, unsigned i, unsigned j) {
        return values[(k * Rows + i) * Columns + j];
    }
    AERO_HOST_DEVICE const T& operator()(unsigned k, unsigned i, unsigned j) const {
        return values[(k * Rows + i) * Columns + j];
    }
};
}
