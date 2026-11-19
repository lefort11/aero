#ifndef AERO_EXPLICIT_TEST_PROBLEM_H
#define AERO_EXPLICIT_TEST_PROBLEM_H
#include "../parallel/kernel_math.h"
#include "explicit.h"
// Три задачи проверяют внешний источник, перенос по кольцу и отказ при недопустимой невязке.
enum class TestProblem { Forced, Ring, Invalid };
AERO_HOST_DEVICE inline double test_residual(TestProblem problem, double t, aero::ExplicitField f, size_t k) {
    if (problem == TestProblem::Invalid)
        return ::nan("");
    const size_t node = k / f.components;
    if (problem == TestProblem::Forced)
        return -f.volumes[node] * (f.values[k] + t);
    // Для первого узла кольца предыдущим считается последний. Номер компоненты остаётся тем же.
    const size_t upstream = node == 0 ? f.nodes - 1 : node - 1;
    return f.values[k] - f.values[upstream * f.components + k % f.components];
}
#ifdef AERO_WITH_CUDA
int current_test_device();
void device_test_residual(TestProblem, double, aero::ExplicitField, double*);
#endif
#endif
