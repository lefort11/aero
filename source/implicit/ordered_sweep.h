#pragma once
#include "csgs_sweep.h"
#include "partition.h"
#include "../parallel/parallel_for.h"
#include "row_execution.h"

namespace aero { namespace implicit {
// Прямой и обратный проход использует один порядок и одну MPI-синхронизацию
// для матрицы газа и для скаляра. Возвращает параметры последнего прохода.
inline CsgsSweep ordered_sweep(CsgsSweep sweep,const std::vector<MeshIndex>& offsets,int device,
        const Partition& partition,double* forward,const double* rhs,double* output) {
    sweep.rhs = rhs;
    sweep.middle_rhs = output; // В прямом проходе выход ещё свободен для b−L*y.

    const MeshIndex groups = offsets.size() - 1;
    for (unsigned pass = 0; pass < 2; ++pass) {
        sweep.backward = pass == 1;
        sweep.output = pass ? output : forward;
        // Все ранги, включая пустые, выполняют один обмен между проходами.
        // При работе без MPI исходная нумерация совпадает с номерами строк.
        if (pass)
            sweep.forward_with_halo =
                partition.extend_vector ? partition.extend_vector(forward) : forward;
        partition.local([&] {
#ifdef AERO_WITH_KOKKOS_GPU
            if (device >= 0) {
                for (MeshIndex j = 0; j < groups; ++j) {
                    const auto color = pass ? groups - 1 - j : j;
                    device_work::check(device_work::aero_device_csgs_color(
                        device, sweep, offsets[color], offsets[color + 1] - offsets[color]));
                }
                return;
            }
#endif
            parallel_for(
                groups,
                [&](MeshIndex group) {
                    const auto begin = offsets[group], end = offsets[group + 1];
                    for (auto k = begin; k < end; ++k)
                        sweep(pass ? end - 1 - (k - begin) : k);
                },
                2);
        });
    }
    return sweep;
}
} }
