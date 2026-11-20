#ifndef AERO_KOKKOS_HOST_H
#define AERO_KOKKOS_HOST_H
// Windows SDK определяет макрос VOID, а в Kokkos 3.2 так называется внутренний тип.
// Сохраняем исходное определение для остального кода, включившего этот заголовок.
#pragma push_macro("VOID")
#undef VOID
#include <Kokkos_Core.hpp>
#pragma pop_macro("VOID")
#include <cstddef>

namespace aero {
namespace host_parallel {
// Класс владеет средой Kokkos, когда библиотека Aero сама запустила её.
// Встроенный солвер может использовать среду, уже созданную вызывающей программой.
class Runtime {
public:
    Runtime() : owner_(!Kokkos::is_initialized()) {
        if (owner_) {
            Kokkos::InitArguments settings;
            settings.disable_warnings = true;
#ifdef _OPENMP
            // Настройки задачи уже применили omp_threads. Берём размер этой команды.
            settings.num_threads = omp_get_max_threads();
#endif
            Kokkos::initialize(settings);
        }
    }
    ~Runtime() {
        if (owner_ && Kokkos::is_initialized())
            Kokkos::finalize();
    }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
private:
    bool owner_; // Только создавшая среду сторона завершает её.
};

// Первый параллельный обход запускает среду. Все следующие обходы используют ту же команду.
inline void ensure_runtime() {
    static Runtime runtime;
}

using Execution = Kokkos::DefaultHostExecutionSpace;
using Range = Kokkos::RangePolicy<Execution, Kokkos::IndexType<std::size_t>>;

// Запускает по одной операции на индекс. В обычной памяти CPU доступны векторы сетки и поля.
template <class Work>
void for_indices(const char* name, std::size_t count, const Work& work) {
    ensure_runtime();
    Kokkos::parallel_for(name, Range(0, count), work);
}
}
}
#endif
