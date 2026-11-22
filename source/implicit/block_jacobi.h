#ifndef AERO_IMPLICIT_BLOCK_JACOBI_H
#define AERO_IMPLICIT_BLOCK_JACOBI_H
#include "../mesh/mesh_index.h"
#include <memory>

namespace aero {
namespace implicit {
// Блочный Якоби для линейной системы A z = r. В каждом собственном узле
// решаем D_i z_i = r_i, где D_i — полный диагональный блок A в этом узле.
// Порядок переменных тот же, что у потока: плотность, импульсы, полная энергия.
struct JacobiBlocks;
class BlockJacobi {
public:
    // dimension = 1, 2 или 3. device = -1 выбирает CPU, иначе это номер GPU Kokkos.
    // use_matrix_diagonal разлагает диагональ на месте применения. Переданные в prepare
    // массивы в этом режиме должны оставаться живыми и неизменными до следующего prepare.
    BlockJacobi(MeshIndex owned_nodes, unsigned dimension, int device = -1, bool use_matrix_diagonal = false);
    ~BlockJacobi();
    BlockJacobi(const BlockJacobi&) = delete;
    BlockJacobi& operator=(const BlockJacobi&) = delete;

    // Вызывать после сборки или обновления якобиана. Для BDF1 блок уже должен
    // содержать (V_i / dt) I и производную пространственной невязки узла.
    // Блоки собственных узлов идут подряд, каждый записан построчно. Исходную
    // матрицу сохраняем для умножений A на вектор в BiCGSTAB.
    void prepare(const double* diagonal);
    // Для блочной CSR-матрицы первый блок строки является диагональным.
    // row_starts задаёт его индекс и позволяет читать прямо из исходной матрицы.
    void prepare(const double* diagonal, const MeshIndex* row_starts);

    // Применяет подготовленный предобусловливатель к новому вектору BiCGSTAB.
    // На GPU оба указателя относятся к памяти выбранной карты, ядро ставится
    // в стандартный поток. Допускается запись поверх входного вектора.
    const JacobiBlocks& factors() const; // Разложение той же диагонали для проходов CSGS.
    void apply(const double* residual, double* correction) const;

private:
    struct Storage;
    std::unique_ptr<Storage> storage_; // Разложения и перестановки строк на выбранном устройстве.
};
}
}
#endif
