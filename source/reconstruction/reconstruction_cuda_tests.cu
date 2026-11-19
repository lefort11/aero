#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "reconstruction.h"
#include "reconstruction_cuda_tests.cuh"
#include "../cuda/device_scope.h"
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

int report(cudaError_t code, char* error, unsigned capacity) {
    if (code != cudaSuccess && error && capacity)
        snprintf(error, capacity, "CUDA: %s", cudaGetErrorString(code));
    return code == cudaSuccess ? 0 : 1;
}
using namespace aero::reconstruction_cuda_tests_kernels;
cudaError_t launch(State* s, const double* values, ReconstructionIndex components, double* left,
                   double* right, cudaStream_t stream) {
    if (!s->view.edge_count)
        return cudaSuccess;
    const ReconstructionIndex items = 2 * s->view.edge_count;
    const ReconstructionIndex count = (items - 1) / 256 + 1;
    const unsigned blocks = static_cast<unsigned>(count > 65535 ? 65535 : count);
    AERO_LAUNCH(reconstruct_kernel, blocks, 256, stream, s->view, values, components, left, right);
    return cudaGetLastError();
}
cudaError_t reserve(double** pointer, size_t* capacity, size_t count) {
    if (count <= *capacity)
        return cudaSuccess;
    double* next = nullptr;
    cudaError_t code = cudaMalloc(reinterpret_cast<void**>(&next), count * sizeof(double));
    if (code != cudaSuccess)
        return code;
    code = cudaFree(*pointer);
    if (code != cudaSuccess) {
        cudaFree(next);
        return code;
    }
    *pointer = next;
    *capacity = count;
    return cudaSuccess;
}
}

extern "C" void test_reconstruction_cuda_destroy(void* opaque) {
    State* s = static_cast<State*>(opaque);
    if (!s)
        return;
    aero::cuda_detail::DeviceScope scope(s->device);
    cudaFree(s->values);
    cudaFree(s->left);
    cudaFree(s->right);
    free(s);
}
extern "C" int test_reconstruction_cuda_create(aero::ReconstructionStencilView view, int device,
                                               void** opaque, char* error, unsigned capacity) {
    *opaque = nullptr;
    aero::cuda_detail::DeviceScope scope(device);
    if (scope.status != cudaSuccess)
        return report(scope.status, error, capacity);
    State* s = static_cast<State*>(calloc(1, sizeof(State)));
    if (!s)
        return report(cudaErrorMemoryAllocation, error, capacity);
    s->device = device;
    s->view = view;
    *opaque = s;
    return 0;
}
extern "C" int test_reconstruction_cuda_device(void* opaque, const double* values,
                                               ReconstructionIndex components, double* left, double* right,
                                               void* stream, char* error, unsigned capacity) {
    State* s = static_cast<State*>(opaque);
    aero::cuda_detail::DeviceScope scope(s->device);
    if (scope.status != cudaSuccess)
        return report(scope.status, error, capacity);
    return report(launch(s, values, components, left, right, static_cast<cudaStream_t>(stream)), error,
                  capacity);
}
extern "C" int test_reconstruction_cuda_host(void* opaque, const double* values,
                                             ReconstructionIndex components, double* left, double* right,
                                             char* error, unsigned capacity) {
    State* s = static_cast<State*>(opaque);
    aero::cuda_detail::DeviceScope scope(s->device);
    if (scope.status != cudaSuccess)
        return report(scope.status, error, capacity);
    const size_t n = static_cast<size_t>(s->view.node_count * components);
    const size_t m = static_cast<size_t>(s->view.edge_count * components);
    if (!m)
        return 0;
    cudaError_t code = reserve(&s->values, &s->value_capacity, n);
    // Ёмкости выходных буферов учитываются отдельно, чтобы ошибка выделения одного не портила другой.
    size_t right_capacity = s->result_capacity;
    if (code == cudaSuccess)
        code = reserve(&s->left, &s->result_capacity, m);
    if (code == cudaSuccess)
        code = reserve(&s->right, &right_capacity, m);
    if (code != cudaSuccess) {
        // Следующий вызов должен заново выделить правый буфер после неудачной попытки.
        cudaFree(s->left);
        cudaFree(s->right);
        s->left = nullptr;
        s->right = nullptr;
        s->result_capacity = 0;
        return report(code, error, capacity);
    }
    code = cudaMemcpy(s->values, values, n * sizeof(double), cudaMemcpyHostToDevice);
    if (code == cudaSuccess)
        code = launch(s, s->values, components, s->left, s->right, nullptr);
    if (code == cudaSuccess)
        code = cudaMemcpy(left, s->left, m * sizeof(double), cudaMemcpyDeviceToHost);
    if (code == cudaSuccess)
        code = cudaMemcpy(right, s->right, m * sizeof(double), cudaMemcpyDeviceToHost);
    return report(code, error, capacity);
}

extern "C" int test_reconstruction_cuda_one(aero::ReconstructionStencilView view, ReconstructionIndex edge,
                                            unsigned side, const double* own, ReconstructionIndex components,
                                            const double* values, double* output, void* stream) {
    AERO_LAUNCH(one_side_kernel, 1, 1, static_cast<cudaStream_t>(stream), view, edge, side, own, components,
                values, output);
    return static_cast<int>(cudaGetLastError());
}
