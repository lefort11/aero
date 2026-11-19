#ifndef AERO_PARALLEL_FOR_H
#define AERO_PARALLEL_FOR_H
#include <cstddef>
#include <exception>

namespace aero {
// Строки независимы. Перехватываем исключения внутри OpenMP и после цикла выдаём ошибку
// с наименьшим индексом, как при последовательном обходе.
template <class F> void parallel_for(std::size_t count, const F& row, std::size_t threshold = 1024) {
#ifdef _OPENMP
    if (count >= threshold) {
        // Исключение не может пересечь границу области OpenMP. Сохраняем его и номер строки.
        std::exception_ptr error;
        std::size_t first = count;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
            try {
                row(static_cast<std::size_t>(i));
            } catch (...) {
// В редком случае ошибки несколько потоков могут одновременно предложить свой индекс.
#pragma omp critical(aero_parallel_row_failure)
                {
                    const auto index = static_cast<std::size_t>(i);
                    if (index < first) {
                        first = index;
                        error = std::current_exception();
                    }
                }
            }
        }
        if (error)
            std::rethrow_exception(error);
        return;
    }
#else
    (void)threshold;
#endif
    for (std::size_t i = 0; i < count; ++i)
        row(i);
}
}
#endif
