#include "block_jacobi.h"
#include "block_jacobi_detail.h"
#include "../parallel/parallel_for.h"
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_api.h"
#endif

namespace aero {
namespace implicit {
// Диагональные блоки и память предобусловливателя на CPU или GPU.
struct BlockJacobi::Storage {
    JacobiBlocks blocks{};               // На CPU и GPU ядро получает адреса местных массивов.
    int device = -1;                     // Владелец разложения, -1 обозначает оперативную память.
    bool ready = false;                  // Только полностью разложенная диагональ пригодна для BiCGSTAB.
    bool use_matrix_diagonal = false;    // Повторно читаем исходный блок вместо второй копии всей диагонали.
    std::vector<double> host_factors;    // LU в памяти CPU.
    std::vector<unsigned char> host_pivots; // Перестановки при разложении на CPU.
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~Storage() {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0 && (blocks.factors || blocks.pivots))
            device_work::aero_device_jacobi_release(device, blocks);
#endif
    }
};

// Выделяет или подключает диагональные блоки на выбранном исполнителе.
BlockJacobi::BlockJacobi(MeshIndex owned_nodes, unsigned dimension, int device, bool use_matrix_diagonal)
    : storage_(new Storage) {
    if (dimension < 1 || dimension > 3 || device < -1)
        throw std::invalid_argument("Block Jacobi: invalid dimension or device");
    const unsigned width = dimension + 2;
    // Проверяем число байтов до умножения независимо от выбранного исполнителя.
    if (owned_nodes > std::numeric_limits<std::size_t>::max() / sizeof(double) / width / width)
        throw std::length_error("Block Jacobi: too many nodes");
    const auto rows = static_cast<std::size_t>(owned_nodes) * width;
    auto& s = *storage_;
    s.device = device;
    s.use_matrix_diagonal = use_matrix_diagonal;
    s.blocks = {owned_nodes, width, nullptr, nullptr, nullptr, nullptr};
    if (device == -1) {
        if (use_matrix_diagonal) return;
        s.host_factors.resize(rows * width);
        s.host_pivots.resize(rows);
        s.blocks.factors = s.host_factors.data();
        s.blocks.pivots = s.host_pivots.data();
        return;
    }
#ifdef AERO_WITH_KOKKOS_GPU
    if (!use_matrix_diagonal)
        device_work::check(device_work::aero_device_jacobi_allocate(device, s.blocks));
#else
    throw std::invalid_argument("Block Jacobi on GPU requires a Kokkos GPU build");
#endif
}
// Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
BlockJacobi::~BlockJacobi() = default;

namespace {
// Номер блока — локальный номер в компактном массиве собственных узлов.
std::domain_error bad_diagonal(MeshIndex node) {
    return std::domain_error("Block Jacobi: singular or non-finite diagonal block at node " +
                             std::to_string(node));
}
}

// Разлагает диагональные блоки и помечает предобусловливатель готовым только при успехе.
void BlockJacobi::prepare(const double* diagonal) {
    prepare(diagonal, nullptr);
}

// Разлагает диагональные блоки и помечает предобусловливатель готовым только при успехе.
void BlockJacobi::prepare(const double* diagonal, const MeshIndex* row_starts) {
    auto& s = *storage_;
    s.ready = false; // После неудачного пересчёта нельзя использовать смесь старых и новых блоков.
    if (s.blocks.nodes && !diagonal)
        throw std::invalid_argument("Block Jacobi: missing diagonal");
    if (s.use_matrix_diagonal) {
        s.blocks.diagonal = diagonal;
        s.blocks.rows = row_starts;
    }
    if (s.device == -1) {
        parallel_for(s.blocks.nodes, [&](std::size_t node) {
            if (!s.blocks.factor_node(node, diagonal, row_starts))
                throw bad_diagonal(node);
        });
    }
#ifdef AERO_WITH_KOKKOS_GPU
    else if (s.blocks.nodes) {
        MeshIndex first_bad;
        device_work::check(device_work::aero_device_jacobi_factor(s.device, s.blocks, diagonal, row_starts, &first_bad));
        if (first_bad < s.blocks.nodes)
            throw bad_diagonal(first_bad);
    }
#endif
    s.ready = true;
}

// Применяет подготовленный предобусловливатель к новому вектору BiCGSTAB.
// На GPU оба указателя относятся к памяти выбранной карты, ядро ставится
// в стандартный поток. Допускается запись поверх входного вектора.
const JacobiBlocks& BlockJacobi::factors() const { return storage_->blocks; }

// Решает независимые системы разложенных диагональных блоков.
void BlockJacobi::apply(const double* residual, double* correction) const {
    const auto& s = *storage_;
    if (!s.ready)
        throw std::logic_error("Block Jacobi: prepare the diagonal before applying it");
    if (!s.blocks.nodes)
        return;
    if (!residual || !correction)
        throw std::invalid_argument("Block Jacobi: missing input or output vector");
    if (s.device == -1) {
        parallel_for(s.blocks.nodes, [&](std::size_t node) {
            s.blocks.solve_node(node, residual, correction);
        });
    }
#ifdef AERO_WITH_KOKKOS_GPU
    else {
        device_work::check(device_work::aero_device_jacobi_apply(s.device, s.blocks, residual, correction));
    }
#endif
}
}
}
