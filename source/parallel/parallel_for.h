#ifndef AERO_PARALLEL_FOR_H
#define AERO_PARALLEL_FOR_H
#include <cstddef>
#include <exception>

namespace aero {
// Independent host rows. Keep exceptions inside the OpenMP region, then
// propagate the lowest failing index just as a serial traversal would.
template<class F> void parallel_for(std::size_t count,const F& row,std::size_t threshold=1024) {
#ifdef _OPENMP
    if (count>=threshold) {
        std::exception_ptr error;
        std::size_t first=count;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i=0;i<static_cast<std::ptrdiff_t>(count);++i) {
            try { row(static_cast<std::size_t>(i)); }
            catch (...) {
#pragma omp critical(aero_parallel_row_failure)
                {
                    const auto index=static_cast<std::size_t>(i);
                    if (index<first) { first=index; error=std::current_exception(); }
                }
            }
        }
        if (error) std::rethrow_exception(error);
        return;
    }
#else
    (void)threshold;
#endif
    for (std::size_t i=0;i<count;++i) row(i);
}
}
#endif
