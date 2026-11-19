#ifndef AERO_EXPLICIT_CUDA_TESTS_CUH
#define AERO_EXPLICIT_CUDA_TESTS_CUH
#include "explicit_test_problem.h"
#include "../cuda/runtime.h"

namespace aero {
namespace explicit_cuda_tests_kernels {
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void evaluate(TestProblem problem, double time, aero::ExplicitField field, double* residual) {
    // Один поток вычисляет невязку одной компоненты по той же эталонной задаче, что и на CPU.
    const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < field.nodes * field.components)
        residual[k] = test_residual(problem, time, field, k);
}
#endif
}
}
#endif
