#ifndef AERO_DEVICE_ARRAYS_H
#define AERO_DEVICE_ARRAYS_H
#include "device_scope.h"
#include "device_check.h"
#include <vector>
#include <limits>
namespace aero {
namespace gpu {
// Массивы одной геометрии или матрицы живут до разрушения владельца. Перед
// выделением вызывающий выбирает device. При разрушении выбираем его сами.
struct DeviceArrays {
    int device = -1;
    std::vector<void*> allocations; // Запоминаем адрес до копирования для очистки и при ошибке.
    // Инициализирует объект стандартными значениями его полей.
    DeviceArrays() = default;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    DeviceArrays(const DeviceArrays&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    DeviceArrays& operator=(const DeviceArrays&) = delete;
    // Резервирует участок массива на GPU с учётом количества элементов и типа.
    template <class T> T* allocate(std::size_t n, const T* src = nullptr) {
        if (!n) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::length_error("GPU array is too large");
        allocations.push_back(nullptr);
        checked(gpu::allocate(&allocations.back(), n * sizeof(T)));
        auto* p = static_cast<T*>(allocations.back());
        if (src) checked(copy(p, src, n * sizeof(T), Copy::ToDevice));
        return p;
    }
    // Выделяет массив GPU и копирует в него содержимое CPU-вектора.
    template <class T> T* upload(const std::vector<T>& values) {
        return allocate(values.size(),values.data());
    }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~DeviceArrays() {
        if (allocations.empty()) return;
        DeviceScope scope(device);
        if (scope.status == Status::Success)
            for (auto p : allocations) gpu::release(p);
    }
};
}
}
#endif
