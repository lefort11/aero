#pragma once
#include "bicgstab_iteration.h"
#include "row_execution.h"

namespace aero { namespace implicit {
// Одинаковые редукции и поэлементные операции для скалярных и газовых векторов.
template<class System> struct KrylovOperations {
    // Получает адреса полей и обмены фактической системы.
    const System& self() const { return static_cast<const System&>(*this); }
    // Выполняет выбранную поэлементную операцию BiCGSTAB на CPU или GPU.
    void update(const KrylovUpdate& operation) const {
        apply_rows(self().device, self().count, operation);
    }
    // Вычисляет локальное скалярное произведение или максимальную норму на выбранном исполнителе.
    double local_reduction(const double* x, const double* y) const {
        if (!self().count) return 0.; // Нейтральный вклад пустого ранга и в сумму, и в норму.
#ifdef AERO_WITH_KOKKOS_GPU
        if (self().device >= 0) {
            double scalar;
            device_work::check(device_work::aero_device_krylov_reduce(self().device, self().count, x, y, &scalar));
            return scalar;
        }
#endif
        host_parallel::ensure_runtime();
        return krylov_reduce(host_parallel::Execution(), self().count, x, y);
    }
    // Согласует локальные редукции и ошибки между процессами MPI.
    double reduce(const double* x, const double* y) const {
        const auto& part = self().partition();
        if (!part.reduce) return local_reduction(x, y);
        double value = 0.;
        std::exception_ptr failure;
        try {
            value = local_reduction(x, y);
        } catch (...) {
            failure = std::current_exception();
            value = HUGE_VAL;
        }
        // Для исправного поля достаточно одной редукции. Бесконечный вклад при
        // ошибке на любом ранге переводит все процессы в общую обработку сбоя.
        value = part.reduce(value, y == nullptr);
        if (!std::isfinite(value)) part.local([&] { if (failure) std::rethrow_exception(failure); });
        return value;
    }
    // Обновляет векторы BiCGSTAB и одновременно вычисляет глобальную норму невязки.
    double update_norm(const KrylovUpdate& operation) const {
        // Для CPU тип исполнителя фиксирован, поэтому вызов исходного ядра встраивается.
        return measure_rows(self().partition(), self().device, self().count, operation,
                            [](const host_parallel::Execution& exec, MeshIndex length, const KrylovUpdate& row) {
                                return krylov_update_norm(exec, length, row);
                            });
    }
    // Вычисляет коэффициент стабилизации BiCGSTAB из глобальных скалярных произведений.
    double omega_projection() const {
        const auto& part=self().partition();
        double pair[2]{};
        part.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
            if (self().device>=0) { device_work::check(device_work::aero_device_krylov_project(self().device,self().count,self().work.t,self().work.r,pair)); return; }
#endif
            host_parallel::ensure_runtime();
            krylov_project(host_parallel::Execution(),self().count,self().work.t,self().work.r,pair);
        });
        if (part.reduce) for (auto& value:pair) value=part.reduce(value,false);
        return pair[0]>0. && std::isfinite(pair[0]) ? pair[1]/pair[0] : 0.;
    }
    // Вычисляет глобальное скалярное произведение рабочих векторов.
    double dot(const double* x, const double* y) const { return reduce(x, y); }
    // Возвращает максимальный модуль компоненты вектора по всем процессам.
    double norm(const double* x) const { return reduce(x, nullptr); }

};
} }
