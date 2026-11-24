#ifndef AERO_IMPLICIT_PARTITION_H
#define AERO_IMPLICIT_PARTITION_H
#include "../timestep/node_step.h"
#include "../mesh/mesh_index.h"
#include "krylov_scratch.h"
#include <functional>
#include <memory>
#include "../parallel/modules.h"
#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/transport.h"
#endif

namespace aero {
namespace residual_detail { struct Geometry; }
namespace implicit {
struct GpuMatrixPlan;
// Связь численного метода с разбиением сетки. Сам решатель не зависит от MPI:
// в обычном расчёте списки пусты, номера узлов и строк совпадают.
struct Partition {
    std::shared_ptr<GpuMatrixPlan> gpu_matrix_plan; // Граф и раскраска из проверки размещения на карте.
    KrylovScratch krylov_scratch; // Пуст для CPU и для раздельных рабочих массивов GPU.
    bool shared_csgs = false; // Для GPU с выбранным CSGS собираем матрицу сразу в порядке проходов.
    const SteadyNodeSteps* steady_steps = nullptr; // Владелец меняет режим между принятыми шагами.
#ifdef AERO_HAS_VISCOSITY
    bool compact_jacobian = false; // Для вязких блоков использовать граф рёбер конвективного оператора.
    viscosity::Operator host_aes{}, active_aes{}; // Топология для CSR и массивы исполнителя для производных.
#endif
    MeshIndex rows = 0, faces = 0;       // Число собственных узлов и вычисляемых площадок.
    const MeshIndex* host_owned = nullptr; // Соответствие строки CSR узлу для построения структуры на CPU.
    const MeshIndex* owned = nullptr;    // То же соответствие на исполнителе данной подобласти.
    const MeshIndex* computed = nullptr; // Площадки, за производные которых отвечает процесс.
    std::function<void(double*)> share_state; // Обновить копии узлов после поправки или отката.
    std::function<void(double*)> share_blocks; // Отправить оба блока производных владельцу второго конца.
    std::function<const double*(const double*)> extend_vector; // Добавить значения вектора соседних узлов.
    std::function<double(double, bool)> reduce; // Общие максимум нормы или сумма произведений.
    std::function<void(const std::function<void()>&)> checked; // Согласовать ошибку до следующего обмена.

    // Выполняет локальную операцию с согласованной обработкой ошибок между рангами.
    template<class Work> void local(const Work& work) const {
        if (checked) checked(work);
        else work();
    }
};
// Топология для построения CSR и адреса этой же сетки на исполнителе ранга.
struct LinearGrid {
    const residual_detail::Geometry& host;
    const residual_detail::Geometry& active;
    int device;
    Partition partition;
};
}
}
#endif
