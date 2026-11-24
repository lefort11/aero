#ifndef AERO_TIMESTEP_H
#define AERO_TIMESTEP_H
#include "../parallel/kernel_math.h"
#include <memory>

namespace aero {
struct MeshView;
struct CvView;
struct PeriodicView;

// Скорости характеристик на внутренних рёбрах и граничных площадках, в порядке геометрии.
struct WaveSpeeds {
    // Скорости волн на внутренних рёбрах.
    const double* edge;
    // Число внутренних значений скорости.
    size_t edge_count;
    // Скорости волн на физических границах.
    const double* boundary;
    // Число граничных значений скорости.
    size_t boundary_count;
};
// Шаг для всей сетки и узел, на котором сработало наиболее жёсткое ограничение.
struct CflStep {
    // Допустимый шаг по условию CFL.
    double dt;
    // Узел, определивший шаг.
    size_t node;
};

// Подготовка один раз получает геометрию с CPU. device = −1 выбирает CPU, другой номер указывает GPU.
// При вызове compute скорости должны находиться на выбранном исполнителе.
class CflTimestep {
public:
    // Подготавливает геометрию ограничения CFL на CPU или GPU.
    CflTimestep(const MeshView&, const CvView&, int device = -1);
    // Подготавливает геометрию ограничения CFL на CPU или GPU.
    explicit CflTimestep(const PeriodicView&, int device = -1);
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~CflTimestep();
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    CflTimestep(const CflTimestep&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    CflTimestep& operator=(const CflTimestep&) = delete;
    // На неподвижной сетке выбирает один шаг для всей области. stream задаёт поток CUDA, готовящий скорости.
    // При равенстве берётся меньший номер расчётного узла. Одновременным вызовам нужны отдельные контексты.
    CflStep compute(double cfl, WaveSpeeds speeds, void* stream = nullptr) const;

private:
    struct Impl;
    // Владеет таблицами для расчёта ограничения CFL.
    std::unique_ptr<Impl> impl_;
};
}
#endif
