#pragma once
#include <algorithm>
#include <array>
#include <initializer_list>
#include <utility>
#include <vector>

namespace aero {
// Короткая строка коэффициентов по номерам узлов. Обычная строка целиком
// хранится в самом объекте; для большой окрестности размер не ограничен.
// Как в упорядоченном словаре, обход всегда идёт по возрастанию номера узла.
template<class Index, std::size_t Capacity> class SmallWeightRow {
public:
    using value_type = std::pair<Index,double>;
    SmallWeightRow() = default;
    SmallWeightRow(std::initializer_list<value_type> entries) {
        for (const auto& entry : entries) (*this)[entry.first] = entry.second;
    }
    const value_type* begin() const { return large_.empty() ? small_.data() : large_.data(); }
    const value_type* end() const { return begin() + size(); }
    std::size_t size() const { return large_.empty() ? used_ : large_.size(); }
    bool empty() const { return size() == 0; }
    bool count(Index n) const {
        const auto k = position(n);
        return k < size() && begin()[k].first == n;
    }
    void clear() { used_ = 0; large_.clear(); }
    void erase(Index n) {
        const auto k = position(n);
        if (k == size() || begin()[k].first != n) return;
        if (large_.empty()) {
            // Сдвигаем хвост влево, сохраняя порядок остальных узлов.
            std::move(small_.begin()+k+1,small_.begin()+used_,small_.begin()+k);
            --used_;
        } else {
            large_.erase(large_.begin()+k);
        }
    }
    double& operator[](Index n) {
        const auto k = position(n);
        if (large_.empty()) {
            if (k < used_ && small_[k].first == n) return small_[k].second;
            if (used_ < small_.size()) {
                // Пока строка короткая, вставка не обращается к распределителю памяти.
                std::move_backward(small_.begin()+k,small_.begin()+used_,small_.begin()+used_+1);
                ++used_;
                small_[k] = {n,0.};
                return small_[k].second;
            }
            // Длинная строка один раз переносит уже накопленные веса в обычный вектор.
            large_.assign(small_.begin(),small_.end());
            used_ = 0;
        } else if (k < large_.size() && large_[k].first == n) return large_[k].second;
        return large_.insert(large_.begin()+k,{n,0.})->second;
    }
private:
    std::size_t position(Index n) const {
        return static_cast<std::size_t>(std::lower_bound(begin(),end(),n,
            [](const value_type& entry, Index node) { return entry.first < node; }) - begin());
    }
    std::array<value_type,Capacity> small_; // Встроенные пары «номер узла — коэффициент».
    std::size_t used_ = 0;                  // Число занятых мест во встроенном массиве.
    std::vector<value_type> large_;         // Непуст только после переполнения встроенного массива.
};
}
