#ifndef AERO_IMPLICIT_SPATIAL_JACOBIAN_H
#define AERO_IMPLICIT_SPATIAL_JACOBIAN_H
#include <memory>
#include "../mesh/mesh_index.h"
#include "partition.h"
#include "phase_timing.h"

namespace aero {
class SpatialResidual;
namespace residual_detail { struct Geometry; }
namespace implicit {
class BlockJacobi;
// Проверка размещения компактной СЛАУ строит тот же граф и ту же раскраску,
// которые затем принимает решатель. Размер — байты матричных массивов на GPU,
// включая промежуточный вектор CSGS; вектора Ньютона и BiCGSTAB считаются отдельно.
std::shared_ptr<GpuMatrixPlan> gpu_matrix_plan(const residual_detail::Geometry& grid, MeshIndex owned_rows,
                                               const Partition& partition);
// Оценивает память GPU для запланированной структуры матрицы.
MeshIndex gpu_matrix_bytes(const GpuMatrixPlan& plan);
// Пространственная часть якобиана первого порядка с потоком LF или Роу. Структуру
// блочной разреженной матрицы строим один раз по связям явной невязки.
class SpatialJacobian {
public:
    // Подготавливает структуру пространственной матрицы для заданной геометрии.
    explicit SpatialJacobian(const SpatialResidual& residual, bool for_linear_solver = false);
    // Подготавливает структуру пространственной матрицы для заданной геометрии.
    SpatialJacobian(const residual_detail::Geometry& host, const residual_detail::Geometry& active, int device,
                      const Partition& partition = Partition(), bool for_linear_solver = false);
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~SpatialJacobian();
    // Поля лежат на исполнителе невязки. Начальное поле фиксировано при
    // дифференцировании и может отсутствовать, если его не требует граница.
    void assemble(const double* state, const double* initial = nullptr);
    // Умножает последнюю успешно собранную матрицу на вектор. Массивы
    // размером nodes*(dimension+2) должны занимать разные области памяти.
    void apply(const double* vector, double* product) const;
private:
    // Линейная система управляет внутренними данными якобиана.
    friend class BdfLinearSystem;
    // Возвращает число узлов в текущем представлении.
    MeshIndex node_count() const;
    // Возвращает пространственную размерность обрабатываемой сущности.
    unsigned dimension() const;
    // Возвращает номер устройства; отрицательное значение соответствует CPU.
    int device() const;
    // Возвращает обмены и редукции распределённой матрицы.
    const Partition& partition() const;
    // Собирает пространственные производные и временной вклад матрицы шага.
    void assemble_matrix(const double* state, const double* initial, const double* volumes, double dt,
                         bool simplified = false, bool roe = false, const PhaseTiming& timing = {});
    // Сохраняет пространственную диагональ для обновления временного вклада.
    void reserve_diagonal();
    // Обновляет временной вклад диагонали при повторном использовании пространственной матрицы.
    void update_time_diagonal(const double* volumes, double dt);
    // Подготавливает порядок обхода и упакованные блоки предобусловливателя CSGS.
    void prepare_csgs();
    // Выполняет прямой и обратный проходы CSGS с необходимыми обменами.
    void apply_csgs(const BlockJacobi& diagonal, const double* rhs, double* output,double* image) const;
    // Подготавливает разложение диагональных блоков текущей матрицы.
    void prepare_jacobi(BlockJacobi& jacobi) const;
    // Выделяет коэффициенты матрицы согласно подготовленной раскладке.
    void allocate_blocks(const residual_detail::Geometry& host, bool for_linear_solver);
    struct Storage;
    // Владеет коэффициентами и структурой пространственного якобиана.
    std::unique_ptr<Storage> storage_;
};
using LfSpatialJacobian = SpatialJacobian;
}
}
#endif
