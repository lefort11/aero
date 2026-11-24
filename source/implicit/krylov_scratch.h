#pragma once
#include "../mesh/mesh_index.h"
#include <algorithm>
#include <cstddef>

namespace aero {
namespace implicit {
enum { NewtonFieldCount = 3, KrylovFieldCount = 6 };

// Пока BiCGSTAB решает систему, потоки по граням и вязкая невязка уже включены
// в правую часть Ньютона. Их временное место можно занять векторами BiCGSTAB.
// Следующая поправка меняет состояние, после чего солвер заново вычисляет потоки.
struct KrylovScratch {
    double* field[KrylovFieldCount]{}; // Адрес начала каждого целого вектора на GPU.
    unsigned count = 0;               // Сколько векторов поместилось в свободных массивах.
    bool contiguous = false;          // Все заимствованные поля занимают одну непрерывную область.

    // При оценке памяти base пуст. Считаем целые векторы по тем же размерам,
    // что при фактической раскладке, и никогда не разделяем вектор между массивами.
    void borrow(void* base, std::size_t bytes, MeshIndex unknowns) {
        if (!unknowns) return;
        const auto available = (bytes / sizeof(double)) / unknowns;
        const auto used = std::min<std::size_t>(KrylovFieldCount-count, available);
        if (used) contiguous=count==0;
        for (std::size_t i=0; i<used; ++i) {
            field[count++] = base ? static_cast<double*>(base) + i*unknowns : nullptr;
        }
    }
};
}
}
