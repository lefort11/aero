#pragma once
#include "jacobian.h"
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace aero {
namespace viscosity {
// Номера столбцов для строки дают пары AES, включая диагонали граней и элементов.
inline void append_columns(const Operator& aes, MeshIndex node, std::vector<MeshIndex>& columns) {
    for (auto k = aes.offsets[node]; k < aes.offsets[node+1]; ++k) {
        const auto index = aes.incidence[k];
        const auto& cell = aes.cells[index/8];
        for (unsigned j = 0; j < cell.count; ++j)
            if (cell.node[j] != node) columns.push_back(cell.node[j]);
    }
}

struct JacobianPattern {
    std::vector<MeshIndex> offsets;
    std::vector<ColumnElement> elements;

    // Преобразуем геометрическую нумерацию в адреса готовой матрицы, общей для
    // конвекции и вязкости. За границей ранга столбцы есть в обмене, строки — у владельца.
    void build(const Operator& aes, const std::vector<MeshIndex>& rows, const std::vector<MeshIndex>& columns) {
        std::vector<MeshIndex> owner_row(aes.node_count,missing_block);
        for (MeshIndex r = 0; r+1 < rows.size(); ++r) owner_row[columns[rows[r]]] = r;
        offsets.assign(aes.node_count+1,0);
        for (MeshIndex e = 0; e < aes.cell_count; ++e) {
            const auto& cell = aes.cells[e];
            for (unsigned j = 0; j < cell.count; ++j) ++offsets[cell.node[j]+1];
        }
        for (MeshIndex j = 0; j < aes.node_count; ++j) offsets[j+1] += offsets[j];
        elements.resize(offsets.back());
        auto next = offsets;
        for (MeshIndex e = 0; e < aes.cell_count; ++e) {
            const auto& cell = aes.cells[e];
            for (unsigned j = 0; j < cell.count; ++j) {
                const auto column = cell.node[j];
                auto& entry = elements[next[column]++];
                entry.cell = e;
                entry.source = j;
                for (unsigned i = 0; i < cell.count; ++i) {
                    const auto row = owner_row[cell.node[i]];
                    if (row == missing_block) { entry.block[i] = missing_block; continue; }
                    const auto first = rows[row];
                    if (column == cell.node[i]) { entry.block[i] = first; continue; }
                    const auto end = columns.begin()+static_cast<std::ptrdiff_t>(rows[row+1]);
                    const auto slot = std::lower_bound(columns.begin()+static_cast<std::ptrdiff_t>(first+1),end,column);
                    if (slot == end || *slot != column) throw std::logic_error("Missing AES Jacobian column");
                    entry.block[i] = static_cast<MeshIndex>(slot-columns.begin());
                }
            }
        }
    }
};
}
}
