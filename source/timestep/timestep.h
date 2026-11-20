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
    const double* edge;
    size_t edge_count;
    const double* boundary;
    size_t boundary_count;
};
// Шаг для всей сетки и узел, на котором сработало наиболее жёсткое ограничение.
struct CflStep {
    double dt;
    size_t node;
};

// Подготовка один раз получает геометрию с CPU. device = −1 выбирает CPU, другой номер указывает GPU.
// При вызове compute скорости должны находиться на выбранном исполнителе.
class CflTimestep {
public:
    CflTimestep(const MeshView&, const CvView&, int device = -1);
    explicit CflTimestep(const PeriodicView&, int device = -1);
    ~CflTimestep();
    CflTimestep(const CflTimestep&) = delete;
    CflTimestep& operator=(const CflTimestep&) = delete;
    // На неподвижной сетке выбирает один шаг для всей области. stream задаёт поток CUDA, готовящий скорости.
    // При равенстве берётся меньший номер расчётного узла. Одновременным вызовам нужны отдельные контексты.
    CflStep compute(double cfl, WaveSpeeds speeds, void* stream = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
