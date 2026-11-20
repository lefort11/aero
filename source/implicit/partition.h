#ifndef AERO_IMPLICIT_PARTITION_H
#define AERO_IMPLICIT_PARTITION_H
#include "../mesh/mesh_index.h"
#include <functional>

namespace aero {
namespace residual_detail { struct Geometry; }
namespace implicit {
// Число компактных рабочих полей определяет длину буфера и резерв памяти карты.
enum { NewtonFieldCount = 3, KrylovFieldCount = 6 };
// Связь численного метода с разбиением сетки. Сам решатель не зависит от MPI:
// в обычном расчёте списки пусты, номера узлов и строк совпадают.
struct Partition {
    MeshIndex rows = 0, faces = 0;       // Число собственных узлов и вычисляемых площадок.
    const MeshIndex* host_owned = nullptr; // Соответствие строки CSR узлу для построения структуры на CPU.
    const MeshIndex* owned = nullptr;    // То же соответствие на исполнителе данной подобласти.
    const MeshIndex* computed = nullptr; // Площадки, за производные которых отвечает процесс.
    std::function<void(double*)> share_state; // Обновить копии узлов после поправки или отката.
    std::function<void(double*)> share_blocks; // Отправить оба блока производных владельцу второго конца.
    std::function<const double*(const double*)> extend_vector; // Добавить значения вектора соседних узлов.
    std::function<double(double, bool)> reduce; // Общие максимум нормы или сумма произведений.
    std::function<void(const std::function<void()>&)> checked; // Согласовать ошибку до следующего обмена.

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
