#include "block_jacobi.h"
#include "csgs_order.h"
#include "csgs_sweep.h"
#include "../../qa/test_field.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#include "../parallel/device_api.h"
#endif

// Три строки с D=2I и двумя связями -I. Жадная раскраска даёт цвета {0,2}, {1}.
// Одну дополнительную связь последней строки выводим на другой процесс.
// В прямом проходе её значение нулевое, в обратном с соседнего ранга приходит 4.
void test_csgs(int device) {
    using namespace aero;
    using namespace implicit;
    const std::vector<MeshIndex> rows{0, 2, 5, 8}, columns{0, 1, 1, 0, 2, 2, 1, 3};
    const CsgsOrder order(4, rows, columns, true, 1);
    aero_test::require(order.sequence == std::vector<MeshIndex>({0, 2, 1}) &&
                           order.offsets == std::vector<MeshIndex>({0, 2, 3}),
                       "CSGS coloring");
    const unsigned width = 3;
    std::vector<double> blocks(columns.size() * width * width);
    for (MeshIndex row = 0; row < 3; ++row)
        for (MeshIndex p = rows[row]; p < rows[row + 1]; ++p)
            for (unsigned k = 0; k < width; ++k)
                blocks[p * width * width + k * width + k] = p == rows[row] ? 2. : -1.;
    aero_test::TestField matrix(device, blocks.size()), rhs(device, 9), answer(device, 9), halo(device, 12);
    matrix.set(blocks);
    rhs.set({1, 1, 1, 2, 2, 2, 3, 3, 3});
    BlockJacobi diagonal(3, 1, device);
    CsgsSweep sweep{};
    sweep.rows = rows.data();
    sweep.columns = columns.data();
    sweep.sequence = order.sequence.data();
    sweep.row_of_node = order.row_of_node.data();
    sweep.position = order.position.data();
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;
    if (device >= 0) {
        arrays.device = device;
        sweep.rows = arrays.allocate(rows.size(), rows.data());
        sweep.columns = arrays.allocate(columns.size(), columns.data());
        sweep.sequence = arrays.allocate(order.sequence.size(), order.sequence.data());
        sweep.row_of_node = arrays.allocate(order.row_of_node.size(), order.row_of_node.data());
        sweep.position = arrays.allocate(order.position.size(), order.position.data());
    }
#endif
    diagonal.prepare(matrix.data(), sweep.rows);
    sweep.diagonal = diagonal.factors();
    sweep.blocks = matrix.data();
    sweep.rhs = rhs.data();
    sweep.output = answer.data();
    const auto solve_color = [&](MeshIndex color) {
        const auto begin = order.offsets[color], end = order.offsets[color + 1];
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) {
            device_work::check(device_work::aero_device_csgs_color(device, sweep, begin, end - begin));
            return;
        }
#endif
        for (auto i = begin; i < end; ++i)
            sweep(i);
    };
    solve_color(0);
    solve_color(1);
    const std::vector<double> forward{.5, .5, .5, 2, 2, 2, 1.5, 1.5, 1.5};
    aero_test::require(answer.read() == forward, "CSGS forward substitution");
    auto extended = forward;
    extended.insert(extended.end(), width, 4.);
    halo.set(extended);
    sweep.backward = true;
    sweep.forward_with_halo = halo.data();
    solve_color(1);
    solve_color(0);
    // Последняя строка даёт (3 + 2 + 4)/2. Ошибка обмена изменила бы этот ответ на 2.5.
    aero_test::require(answer.read() == std::vector<double>({1.5, 1.5, 1.5, 2, 2, 2, 4.5, 4.5, 4.5}),
                       "CSGS backward substitution with halo");
}
