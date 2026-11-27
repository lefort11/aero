#pragma once
#include "csgs_sweep.h"
#include <vector>

namespace aero {
namespace implicit {
// Представление существующего газового графа для одной неизвестной на узел.
struct ScalarTopology {
    const MeshIndex* rows; // Смещения обычных строк для Якоби.
    IndexList columns; // Полные узловые номера обычного графа.
    CsgsSweep sweep; // Порядок и номера соседей общей раскраски CSGS.
    const std::vector<MeshIndex>* offsets; // Цвета GPU либо независимые участки CPU.
    MeshIndex slots; // Число скалярных коэффициентов вместе с выравниванием.
    double* forward; // Промежуточный вектор прямого газового прохода свободен в блоке SA.
};
}
}
