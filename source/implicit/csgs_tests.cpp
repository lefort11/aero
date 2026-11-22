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
    aero_test::TestField matrix(device, blocks.size()), rhs(device, 9), answer(device, 12), halo(device, 12);
    matrix.set(blocks);
    rhs.set({1, 1, 1, 2, 2, 2, 3, 3, 3});
    BlockJacobi diagonal(3, 1, device);
    CsgsLayout layout;
    layout.build(order,rows,columns,device>=0);
    aero_test::TestField packed(device,layout.source.size()*width*width), middle(device,9);
    CsgsSweep sweep{};
    CsgsPacking packing{layout.source.data(),packed.data(),layout.source.size(),width,layout.lanes,{},nullptr};
    sweep.sequence=order.sequence.data();
    sweep.rows=layout.rows.data();
    sweep.neighbours=layout.neighbours.data();
    sweep.lanes=layout.lanes;
    sweep.middle_rhs=middle.data();
    const MeshIndex* starts=rows.data();
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;
    if (device>=0) {
        arrays.device=device;
        starts=arrays.upload(rows);
        sweep.sequence=arrays.upload(order.sequence);
        sweep.rows=arrays.upload(layout.rows);
        sweep.neighbours=arrays.upload(layout.neighbours);
        packing.source=arrays.upload(layout.source);
        device_work::check(device_work::aero_device_pack_csgs(device,packing,matrix.data()));
    } else
#endif
        for (MeshIndex p=0;p<packing.count;++p) packing.copy(p,matrix.data());
    diagonal.prepare(matrix.data(),starts);
    sweep.diagonal=diagonal.factors();
    sweep.blocks=packing.values;
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
    // Фиксируем значение внешнего узла отдельно от трёх собственных строк.
    answer.set({0,0,0, 0,0,0, 0,0,0, 4,4,4});
    solve_color(0);
    solve_color(1);
    const std::vector<double> forward{.5, .5, .5, 2, 2, 2, 1.5, 1.5, 1.5};
    auto extended = forward;
    extended.insert(extended.end(), width, 4.);
    aero_test::require(answer.read() == extended, "CSGS forward substitution");
    halo.set(extended);
    sweep.backward = true;
    sweep.forward_with_halo = halo.data();
    solve_color(1);
    solve_color(0);
    // Последняя строка даёт (3 + 2 + 4)/2. Ошибка обмена изменила бы этот ответ на 2.5.
    aero_test::require(answer.read() == std::vector<double>({1.5, 1.5, 1.5, 2, 2, 2, 4.5, 4.5, 4.5, 4, 4, 4}),
                       "CSGS backward substitution with halo");
    // Проверяем укороченное умножение независимо, через исходные строки CSR.
    const auto z = answer.read();
    std::vector<double> expected(3 * width);
    for (MeshIndex row = 0; row < 3; ++row)
        for (MeshIndex p = rows[row]; p < rows[row+1]; ++p)
            for (unsigned i = 0; i < width; ++i)
                for (unsigned j = 0; j < width; ++j)
                    expected[row*width+i] += blocks[(p*width+i)*width+j] * z[columns[p]*width+j];
    aero_test::TestField image(device, expected.size());
    sweep.external = true;
#ifdef AERO_WITH_KOKKOS_GPU
    if (device >= 0)
        device_work::check(device_work::aero_device_csgs_image(device, sweep, 3, image.data()));
    else
#endif
        for (MeshIndex i = 0; i < 3; ++i) sweep.image<1>(i, image.data());
    aero_test::require(image.read() == expected, "CSGS product versus original CSR");
}
