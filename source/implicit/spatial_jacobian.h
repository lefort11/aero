#ifndef AERO_IMPLICIT_SPATIAL_JACOBIAN_H
#define AERO_IMPLICIT_SPATIAL_JACOBIAN_H
#include <memory>
#include "../mesh/mesh_index.h"
#include "partition.h"

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
MeshIndex gpu_matrix_bytes(const GpuMatrixPlan& plan);
// Пространственная часть якобиана первого порядка с потоком LF или Роу. Структуру
// блочной разреженной матрицы строим один раз по связям явной невязки.
class SpatialJacobian {
public:
    explicit SpatialJacobian(const SpatialResidual& residual, bool for_linear_solver = false);
    SpatialJacobian(const residual_detail::Geometry& host, const residual_detail::Geometry& active, int device,
                      const Partition& partition = Partition(), bool for_linear_solver = false);
    ~SpatialJacobian();
    // Поля лежат на исполнителе невязки. Начальное поле фиксировано при
    // дифференцировании и может отсутствовать, если его не требует граница.
    void assemble(const double* state, const double* initial = nullptr);
    // Умножает последнюю успешно собранную матрицу на вектор. Массивы
    // размером nodes*(dimension+2) должны занимать разные области памяти.
    void apply(const double* vector, double* product) const;
private:
    friend class BdfLinearSystem;
    MeshIndex node_count() const;
    unsigned dimension() const;
    int device() const;
    const Partition& partition() const;
    void assemble_matrix(const double* state, const double* initial, const double* volumes, double dt, bool simplified = false, bool roe = false);
    void reserve_diagonal();
    void update_time_diagonal(const double* volumes, double dt);
    void prepare_csgs();
    void apply_csgs(const BlockJacobi& diagonal, const double* rhs, double* output,double* image) const;
    void prepare_jacobi(BlockJacobi& jacobi) const;
    void allocate_blocks(const residual_detail::Geometry& host, bool for_linear_solver);
    struct Storage;
    std::unique_ptr<Storage> storage_;
};
using LfSpatialJacobian = SpatialJacobian;
}
}
#endif
