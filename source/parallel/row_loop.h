#pragma once
#include "../mesh/mesh_index.h"
#pragma push_macro("VOID")
#undef VOID
#include <Kokkos_Core.hpp>
#pragma pop_macro("VOID")
#include <type_traits>

namespace aero {
namespace row_loop {
// Диапазон использует индекс сетки. LaunchBounds задаёт прежнюю загрузку блоков CUDA.
template<class Space, unsigned min_blocks = 0>
using Range = Kokkos::RangePolicy<Space, Kokkos::IndexType<MeshIndex>
#ifdef KOKKOS_ENABLE_CUDA
    , Kokkos::LaunchBounds<128, min_blocks>
#endif
    >;

// Устройство получает прежнюю политику HintLightWeight и переданный поток Kokkos.
template<unsigned min_blocks, class Space, class Work>
inline void visit(const Space& exec, const char* name, MeshIndex count, const Work& work, std::false_type) {
    if (count)
        Kokkos::parallel_for(name, Kokkos::Experimental::require(Range<Space,min_blocks>(exec,0,count),
                             Kokkos::Experimental::WorkItemProperty::HintLightWeight), work);
}

// На хосте малый пакет дешевле обработать последовательно. Тело строки общее с GPU.
template<unsigned min_blocks, class Space, class Work>
inline void visit(const Space& exec, const char* name, MeshIndex count, const Work& work, std::true_type) {
    if (count < 256) {
        for (MeshIndex i = 0; i < count; ++i) work(i);
    } else {
        Kokkos::parallel_for(name, Range<Space>(exec,0,count), work);
    }
}

// Выбирает способ запуска при компиляции. Численный оператор хранит адреса активной памяти.
template<unsigned min_blocks = 0, class Space, class Work>
inline void for_indices(const Space& exec, const char* name, MeshIndex count, const Work& work) {
    visit<min_blocks>(exec, name, count, work, std::is_same<Space,Kokkos::DefaultHostExecutionSpace>{});
}
}
}
