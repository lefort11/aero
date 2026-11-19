#ifndef AERO_EXPLICIT_H
#define AERO_EXPLICIT_H

#include "../parallel/kernel_math.h"
#ifndef __CUDACC_RTC__
#include <functional>
#include <memory>
#include <vector>
#endif

namespace aero {
enum class ExplicitScheme { Euler, SSPRK3 };
enum class ExplicitBackend { Auto, Cpu, Cuda };

// Представление действительно во время обратного вызова для невязки или чтения поля.
// На CUDA указатели и поток принадлежат выбранной карте. На CPU device = −1, stream = nullptr.
struct ExplicitField {
    size_t nodes, components;
    const double* values;  // Индекс компоненты равен node * components + component.
    const double* volumes; // Положительный постоянный объём каждого расчётного узла.
    ExplicitBackend backend;
    int device;
    void* stream;
};

// Заполните каждую компоненту R суммой выходящих потоков с учётом площади.
// Исключение отклоняет стадию. На CUDA операции ставятся в очередь field.stream.
#ifndef __CUDACC_RTC__
typedef std::function<void(double time, const ExplicitField& field, double* R)> ExplicitResidual;

class ExplicitIntegrator {
public:
    // Оба входных массива находятся на CPU. Конструктор сохраняет их копии.
    ExplicitIntegrator(std::size_t nodes, std::size_t components, const double* volumes,
                       const double* initial, ExplicitBackend backend = ExplicitBackend::Auto,
                       int device = 0);
    ~ExplicitIntegrator();
    ExplicitIntegrator(const ExplicitIntegrator&) = delete;
    ExplicitIntegrator& operator=(const ExplicitIntegrator&) = delete;

    // Время и dt задаёт вызывающий код. Принятое состояние заменяется по завершении всего шага.
    void step(ExplicitScheme scheme, double time, double dt, const ExplicitResidual& residual);
    // Можно передать первую невязку, уже вычисленную для текущих времени и принятого поля.
    // Её память должна быть доступна выбранному исполнителю до конца вызова.
    void step(ExplicitScheme scheme, double time, double dt, const ExplicitResidual& residual,
              const double* first_residual);
    ExplicitBackend backend() const;
    // Даёт доступ к принятому полю между шагами. Для CUDA на время чтения активируется выбранное устройство.
    void inspect(const std::function<void(const ExplicitField&)>& reader) const;
    std::vector<double> state() const; // Копия принятого поля последнего завершённого шага в памяти CPU.
    static int cuda_device_count();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif
}
#endif
