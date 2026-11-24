#define AERO_EULERSOLVER_KERNEL_ONLY
#include "lax_friedrichs.h"
#include "roe.h"
#include "characteristic.h"
#include "../../tests/device_cases.h"
#include "../parallel/device_scope.h"
#include <stdio.h>

namespace {
// Тест показывает запуск ядра вызывающей стороной для набора независимых пар состояний.
using namespace aero::device_tests;
// Сохраняет диагностический текст ошибки GPU-проверки.
int report(aero::gpu::Status code, char* error, unsigned capacity) {
    if (code != aero::gpu::Status::Success && error && capacity)
        snprintf(error, capacity, "CUDA: %s", aero::gpu::message(code));
    return code == aero::gpu::Status::Success ? 0 : 1;
}
}
// Возвращает число устройств, доступных GPU-проверкам потоков.
extern "C" int euler_test_device_count(int* count, char* error, unsigned capacity) {
    *count = 0;
    aero::gpu::Status code = aero::gpu::count(count);
    if (code == aero::gpu::Status::NoDevice || code == aero::gpu::Status::Unavailable) {

        *count = 0;
        return 0;
    }
    return report(code, error, capacity);
}
// Запускает выбранное потоковое ядро на GPU и возвращает результат проверки.
extern "C" int euler_test_device_flux(int device, unsigned mode, unsigned dimension, const double* left,
                                    const double* right, const double* aux_left, const double* aux_right,
                                    const double* direction, double gamma, aero::EulerFluxResult* result,
                                    int* status, char* error, unsigned capacity) {
    aero::gpu::DeviceScope scope(device);
    if (scope.status != aero::gpu::Status::Success)
        return report(scope.status, error, capacity);
    Case input = {};
    input.mode = mode;
    input.dimension = dimension;
    input.gamma = gamma;
    for (unsigned k = 0; k < dimension + 2; ++k) {
        input.left[k] = left[k];
        input.right[k] = right[k];
        if (aux_left)
            input.aux_left[k] = aux_left[k];
        if (aux_right)
            input.aux_right[k] = aux_right[k];
    }
    for (unsigned k = 0; k < dimension; ++k)
        input.direction[k] = direction[k];
    Output output = {123, *result, 456, -1};
    Case* di = nullptr;
    Output* dout = nullptr;
    aero::gpu::Stream stream = nullptr;
    aero::gpu::Status code = aero::gpu::allocate(reinterpret_cast<void**>(&di), sizeof(Case));
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::allocate(reinterpret_cast<void**>(&dout), sizeof(Output));
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::create_stream(&stream);
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::copy(di, &input, sizeof(Case), aero::gpu::Copy::ToDevice);
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::copy(dout, &output, sizeof(Output), aero::gpu::Copy::ToDevice);
    if (code == aero::gpu::Status::Success) {
        aero::device_work::check(one_interface(stream, di, dout));

    }
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::wait(stream);
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::copy(&output, dout, sizeof(Output), aero::gpu::Copy::ToHost);
    if (stream)
        aero::gpu::destroy_stream(stream);
    aero::gpu::release(di);
    aero::gpu::release(dout);
    if (code != aero::gpu::Status::Success)
        return report(code, error, capacity);
    if (output.before != 123 || output.after != 456) {
        snprintf(error, capacity, "CUDA wrote beyond one flux result");
        return 1;
    }
    *result = output.result;
    *status = output.status;
    return 0;
}
