#ifndef AERO_TEST_FIELD_H
#define AERO_TEST_FIELD_H
#include "test_checks.h"
#include <vector>
#ifdef AERO_WITH_KOKKOS_GPU
#include "device_test_buffer.h"
#endif
namespace aero_test {
// Буфер тестового поля на выбранном исполнителе. Копируем только входные данные и результат.
class TestField {
public:
    // Выделяет тестовое поле на CPU и при необходимости его копию на GPU.
    TestField(int device, std::size_t count) : host_(count)
#if defined(AERO_WITH_KOKKOS_GPU)
        , gpu_(device < 0 ? 0 : count)
#endif
    { (void)device; }
    // Возвращает адрес поля на выбранном тестовом исполнителе.
    double* data() {
#if defined(AERO_WITH_KOKKOS_GPU)
        if (gpu_.data)
            return gpu_.data;
#endif
        return host_.data();
    }
    // Копирует исходные значения в CPU- и при необходимости GPU-поле.
    void set(const std::vector<double>& values) {
        require(values.size() == host_.size(), "test buffer size mismatch");
#if defined(AERO_WITH_KOKKOS_GPU)
        if (gpu_.data) {
            gpu_.upload(values.data(), values.size());
            return;
        }
#endif
        host_ = values;
    }
    // Возвращает CPU-копию актуального тестового поля.
    const std::vector<double>& read() {
#if defined(AERO_WITH_KOKKOS_GPU)
        if (gpu_.data)
            gpu_.download(host_.data(), host_.size());
#endif
        return host_;
    }
private:
    std::vector<double> host_; // Значения для сверки с известным решением.
#if defined(AERO_WITH_KOKKOS_GPU)
    aero_test::DeviceBuffer<double> gpu_; // Та же раскладка для проверки устройства.
#endif
};
}
#endif
