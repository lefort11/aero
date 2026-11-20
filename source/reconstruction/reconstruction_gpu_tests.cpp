#include "reconstruction.h"
#include "../../qa/device_cases.h"
#include "../parallel/device_scope.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

namespace {
using aero::ReconstructionIndex;
struct State {
    aero::ReconstructionStencilView view;
    int device;
    double* values;
    double* left;
    double* right;
    size_t value_capacity, result_capacity;
};

int report(aero::gpu::Status code, char* error, unsigned capacity) {
    if (code != aero::gpu::Status::Success && error && capacity)
        snprintf(error, capacity, "CUDA: %s", aero::gpu::message(code));
    return code == aero::gpu::Status::Success ? 0 : 1;
}
using namespace aero::device_tests;
aero::gpu::Status launch(State* s, const double* values, ReconstructionIndex components, double* left,
                   double* right, aero::gpu::Stream stream) {
    if (!s->view.edge_count)
        return aero::gpu::Status::Success;
    aero::device_work::check(reconstruct_kernel(stream, s->view, values, components, left, right));
    return aero::gpu::Status::Success;
}
aero::gpu::Status reserve(double** pointer, size_t* capacity, size_t count) {
    if (count <= *capacity)
        return aero::gpu::Status::Success;
    double* next = nullptr;
    aero::gpu::Status code = aero::gpu::allocate(reinterpret_cast<void**>(&next), count * sizeof(double));
    if (code != aero::gpu::Status::Success)
        return code;
    code = aero::gpu::release(*pointer);
    if (code != aero::gpu::Status::Success) {
        aero::gpu::release(next);
        return code;
    }
    *pointer = next;
    *capacity = count;
    return aero::gpu::Status::Success;
}
}

extern "C" void test_reconstruction_device_destroy(void* opaque) {
    State* s = static_cast<State*>(opaque);
    if (!s)
        return;
    aero::gpu::DeviceScope scope(s->device);
    aero::gpu::release(s->values);
    aero::gpu::release(s->left);
    aero::gpu::release(s->right);
    free(s);
}
extern "C" int test_reconstruction_device_create(aero::ReconstructionStencilView view, int device,
                                               void** opaque, char* error, unsigned capacity) {
    *opaque = nullptr;
    aero::gpu::DeviceScope scope(device);
    if (scope.status != aero::gpu::Status::Success)
        return report(scope.status, error, capacity);
    State* s = static_cast<State*>(calloc(1, sizeof(State)));
    if (!s)
        return report(aero::gpu::Status::OutOfMemory, error, capacity);
    s->device = device;
    s->view = view;
    *opaque = s;
    return 0;
}
extern "C" int test_reconstruction_device_device(void* opaque, const double* values,
                                               ReconstructionIndex components, double* left, double* right,
                                               void* stream, char* error, unsigned capacity) {
    State* s = static_cast<State*>(opaque);
    aero::gpu::DeviceScope scope(s->device);
    if (scope.status != aero::gpu::Status::Success)
        return report(scope.status, error, capacity);
    return report(launch(s, values, components, left, right, static_cast<aero::gpu::Stream>(stream)), error,
                  capacity);
}
extern "C" int test_reconstruction_device_host(void* opaque, const double* values,
                                             ReconstructionIndex components, double* left, double* right,
                                             char* error, unsigned capacity) {
    State* s = static_cast<State*>(opaque);
    aero::gpu::DeviceScope scope(s->device);
    if (scope.status != aero::gpu::Status::Success)
        return report(scope.status, error, capacity);
    const size_t n = static_cast<size_t>(s->view.node_count * components);
    const size_t m = static_cast<size_t>(s->view.edge_count * components);
    if (!m)
        return 0;
    aero::gpu::Status code = reserve(&s->values, &s->value_capacity, n);
    // Ёмкости выходных буферов учитываются отдельно, чтобы ошибка выделения одного не портила другой.
    size_t right_capacity = s->result_capacity;
    if (code == aero::gpu::Status::Success)
        code = reserve(&s->left, &s->result_capacity, m);
    if (code == aero::gpu::Status::Success)
        code = reserve(&s->right, &right_capacity, m);
    if (code != aero::gpu::Status::Success) {
        // Следующий вызов должен заново выделить правый буфер после неудачной попытки.
        aero::gpu::release(s->left);
        aero::gpu::release(s->right);
        s->left = nullptr;
        s->right = nullptr;
        s->result_capacity = 0;
        return report(code, error, capacity);
    }
    code = aero::gpu::copy(s->values, values, n * sizeof(double), aero::gpu::Copy::ToDevice);
    if (code == aero::gpu::Status::Success)
        code = launch(s, s->values, components, s->left, s->right, nullptr);
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::copy(left, s->left, m * sizeof(double), aero::gpu::Copy::ToHost);
    if (code == aero::gpu::Status::Success)
        code = aero::gpu::copy(right, s->right, m * sizeof(double), aero::gpu::Copy::ToHost);
    return report(code, error, capacity);
}

extern "C" int test_reconstruction_device_one(aero::ReconstructionStencilView view, ReconstructionIndex edge,
                                            unsigned side, const double* own, ReconstructionIndex components,
                                            const double* values, double* output, void* stream) {
    aero::device_work::check(one_side_kernel(stream, view, edge, side, own, components, values, output));
    return 0;
}
