#pragma once
#include "../mesh/compact_table.h"
#include "parallel_for.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace aero {
// Выбор формата учитывает все значения, а не только число узлов: смещения
// и кодированные вхождения могут оказаться больше самой сетки.
template<class Storage,class Index>
IndexArray<Index> upload_indices(Storage& storage,std::size_t count,const Index* source) {
    if (!count) return {};
    if (*std::max_element(source,source+count)>std::numeric_limits<std::uint32_t>::max())
        return storage.allocate(count,source);
    std::vector<std::uint32_t> packed(count);
    parallel_for(count,[&](std::size_t i) { packed[i]=static_cast<std::uint32_t>(source[i]); });
    return IndexArray<Index>(storage.allocate(count,packed.data()));
}
// Упаковка обратима: double копируются без преобразования; большие номера
// оставляют всю таблицу в исходном формате. Проверка и запись используют один критерий.
template<class Packed,class Storage,class Wide>
CompactTable<Wide,Packed> upload_records(Storage& storage,std::size_t count,const Wide* source) {
    if (!count) return {};
    if (!std::all_of(source,source+count,[](const Wide& value) { return Packed::fits(value); }))
        return storage.allocate(count,source);
    std::vector<Packed> packed(count);
    parallel_for(count,[&](std::size_t i) { packed[i]=Packed::pack(source[i]); });
    return CompactTable<Wide,Packed>(storage.allocate(count,packed.data()));
}
}
