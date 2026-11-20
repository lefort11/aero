#ifndef AERO_PARALLEL_FOR_H
#define AERO_PARALLEL_FOR_H
#include <cstddef>
#include <exception>
#include <mutex>
#include "kokkos_host.h"

namespace aero {
// Строки независимы. Перехватываем исключения внутри OpenMP и после цикла выдаём ошибку
// с наименьшим индексом, как при последовательном обходе.
template <class F> void parallel_for(std::size_t count, const F& row, std::size_t threshold = 1024) {
    if (count >= threshold) {
        std::exception_ptr error;
        std::size_t first = count;
        std::mutex error_lock;
        const auto visit = [&](std::size_t i) {
            try {
                row(i);
            } catch (...) {
                // Ошибки редки. Блокировка защищает только выбор первой ошибочной строки.
                std::lock_guard<std::mutex> guard(error_lock);
                if (i < first) {
                    first = i;
                    error = std::current_exception();
                }
            }
        };
        host_parallel::for_indices("Aero: independent rows", count, visit);
        if (error)
            std::rethrow_exception(error);
        return;
    }
    for (std::size_t i = 0; i < count; ++i)
        row(i);
}
}
#endif
