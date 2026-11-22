#pragma once
#include <vector>
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#endif

namespace aero {
namespace implicit {
// Ньютон и BiCGSTAB размещают рабочие векторы там же, где выполняют операции.
// На CPU каждым буфером владеет переданный vector. На GPU все выделения
// принадлежат одной карте и освобождаются вместе с решателем.
struct Workspace {
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;
#endif
    double* allocate(int device, std::vector<double>& host, std::size_t count) {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            gpu::DeviceScope scope(device);
            gpu::checked(scope.status);
            arrays.device = device;
            return arrays.allocate<double>(count);
        }
#else
        (void)device;
#endif
        host.resize(count);
        return host.data();
    }
};
}
}
