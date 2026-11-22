#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace aero {
// Таблица на CPU с известным числом столбцов и одним выделением для всех строк.
// Используется для узловых полей и пар номеров рёбер, когда число строк задаёт сетка.
template<class T, std::size_t Columns>
class RowTable {
    static_assert(Columns>0,"A row must contain at least one column");
    std::vector<T> values_;
public:
    explicit RowTable(std::size_t rows=0) { resize(rows); }
    void resize(std::size_t rows,T fill=T()) {
        if (rows>values_.max_size()/Columns) throw std::length_error("Row table is too large");
        values_.resize(rows*Columns,fill);
    }
    bool empty() const { return values_.empty(); }
    T& operator()(std::size_t i,std::size_t j) { return values_[i*Columns+j]; }
    const T& operator()(std::size_t i,std::size_t j) const { return values_[i*Columns+j]; }
    // Преобразование состояния и расчёт ошибки обрабатывают строку целиком.
    void set_row(std::size_t i,const std::array<T,Columns>& row) {
        std::copy(row.begin(),row.end(),values_.data()+i*Columns);
    }
    std::array<T,Columns> row(std::size_t i) const {
        std::array<T,Columns> result;
        std::copy_n(values_.data()+i*Columns,Columns,result.begin());
        return result;
    }
};
}
