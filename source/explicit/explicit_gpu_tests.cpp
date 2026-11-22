#include "explicit_test_problem.h"
#include "../../tests/device_cases.h"
#include <stdexcept>
namespace {
using namespace aero::device_tests;
}
int current_test_device() {
    int device;
    const auto code = aero::gpu::current(&device);
    if (code != aero::gpu::Status::Success)
        throw std::runtime_error(aero::gpu::message(code));
    return device;
}
// Ставит тестовую невязку в очередь потока интегратора, сохраняя порядок между стадиями.
void device_test_residual(TestProblem problem, double time, aero::ExplicitField field, double* residual) {
    aero::device_work::check(evaluate(field.stream, problem, time, field, residual));
}
