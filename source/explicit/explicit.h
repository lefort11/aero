#ifndef AERO_EXPLICIT_H
#define AERO_EXPLICIT_H

#include "../parallel/kernel_math.h"
#include <functional>
#include <memory>
#include <vector>

namespace aero {
enum class ExplicitScheme { Euler, SSPRK3 };
enum class ExplicitBackend { Auto, Cpu, Cuda };

// Представление действительно во время обратного вызова для невязки или чтения поля.
// На CUDA указатели и поток принадлежат выбранной карте. На CPU device = −1, stream = nullptr.
struct ExplicitField {
    // Количество узлов и компонент на узел.
    size_t nodes, components;
    const double* values;  // Индекс компоненты равен node * components + component.
    const double* volumes; // Положительный постоянный объём каждого расчётного узла.
    // Исполнитель операций над полем.
    ExplicitBackend backend;
    // Номер GPU для указателей поля.
    int device;
    // Поток выполнения операций устройства.
    void* stream;
};

// Заполните каждую компоненту R суммой выходящих потоков с учётом площади.
// Исключение отклоняет стадию. На CUDA операции ставятся в очередь field.stream.
typedef std::function<void(double time, const ExplicitField& field, double* R)> ExplicitResidual;

// Управляет явными стадиями и размещением поля на выбранном исполнителе.
class ExplicitIntegrator {
public:
    // Оба входных массива находятся на CPU. Конструктор сохраняет их копии.
    ExplicitIntegrator(std::size_t nodes, std::size_t components, const double* volumes,
                       const double* initial, ExplicitBackend backend = ExplicitBackend::Auto,
                       int device = 0);
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~ExplicitIntegrator();
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    ExplicitIntegrator(const ExplicitIntegrator&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    ExplicitIntegrator& operator=(const ExplicitIntegrator&) = delete;

    // Время и dt задаёт вызывающий код. Принятое состояние заменяется по завершении всего шага.
    void step(ExplicitScheme scheme, double time, double dt, const ExplicitResidual& residual);
    // Можно передать первую невязку, уже вычисленную для текущих времени и принятого поля.
    // Её память должна быть доступна выбранному исполнителю до конца вызова.
    void step(ExplicitScheme scheme, double time, double dt, const ExplicitResidual& residual,
              const double* first_residual);
    // Возвращает выбранный исполнитель вычислений.
    ExplicitBackend backend() const;
    // Даёт доступ к принятому полю между шагами. Для CUDA на время чтения активируется выбранное устройство.
    void inspect(const std::function<void(const ExplicitField&)>& reader) const;
    // Возвращает CPU-копию последнего принятого поля.
    std::vector<double> state() const; // Копия принятого поля последнего завершённого шага в памяти CPU.
    // Возвращает число доступных CUDA-устройств.
    static int cuda_device_count();

private:
    struct Impl;
    // Владеет рабочими массивами стадий интегратора.
    std::unique_ptr<Impl> impl_;
};
}
#endif
