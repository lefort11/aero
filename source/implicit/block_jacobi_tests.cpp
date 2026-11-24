#include "../../tests/test_field.h"
#include "block_jacobi.h"
#include "../parallel/modules.h"
#include "../../tests/test_checks.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#if defined(AERO_WITH_KOKKOS_GPU)
#include "../../tests/device_test_buffer.h"
#include "../parallel/device_scope.h"
#endif

// Сравнивает производные потока LF с независимым численным эталоном.
void test_lf_edge_jacobian(int device);
// Проверяет сборку и действие пространственной матрицы Якоби.
void test_spatial_jacobian(int device);
// Проверяет сборку и решение линейной системы BDF.
void test_bdf1_linear_system(int device);
// Проверяет неявный шаг и его сходимость.
void test_bdf1_step(int device);
// Проверяет проходы и предобусловливание CSGS.
void test_csgs(int device);
#ifdef AERO_HAS_MULTIGRID
// Проверяет ограничение, продление и принятие грубосеточной коррекции.
void test_steady_multigrid(int device);
#endif

namespace {
using aero::implicit::BlockJacobi;
using aero_test::require;

using Values = aero_test::TestField;

// Строим плотные несимметричные блоки разных масштабов. Циклическая перестановка
// строк заставляет алгоритм искать главный элемент и на поздних шагах исключения.
std::vector<double> make_diagonal(std::size_t nodes, unsigned width) {
    const double scales[] = {1e-180, 1., 1e180};
    std::vector<double> matrix(nodes * width * width);
    for (std::size_t node = 0; node < nodes; ++node)
        for (unsigned row = 0; row < width; ++row)
            for (unsigned col = 0; col < width; ++col) {
                const auto before_swap = (row + 1) % width;
                const double value = before_swap == col ? 8. + static_cast<double>(node % 11)
                                                        : .125 * (1 + 2 * before_swap + col);
                matrix[(node * width + row) * width + col] = scales[node % 3] * value;
            }
    return matrix;
}

// Обычное умножение исходной матрицы на заданную поправку даёт независимую правую часть.
std::vector<double> multiply(const std::vector<double>& matrix, const std::vector<double>& x, unsigned width) {
    std::vector<double> b(x.size(), 0.);
    for (std::size_t row = 0; row < x.size(); ++row)
        for (unsigned col = 0; col < width; ++col)
            b[row] += matrix[row * width + col] * x[(row / width) * width + col];
    return b;
}

// Проверяет выбрасывание исключения при недопустимых параметрах.
template <class Exception, class Operation>
void expect_error(Operation operation, const std::string& message) {
    try {
        operation();
    } catch (const Exception& error) {
        require(std::string(error.what()).find(message) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("Expected exception containing " + message);
}

// Проверяет разложение и решение набора диагональных блоков.
void check_blocks(int device, unsigned dimension, bool use_matrix_diagonal) {
    const unsigned width = dimension + 2;
    const std::size_t nodes = 1031; // Неполная последняя группа на GPU, выше порога параллельного цикла CPU.
    BlockJacobi jacobi(nodes, dimension, device, use_matrix_diagonal);
    auto matrix = make_diagonal(nodes, width);
    Values diagonal(device, matrix.size()), rhs(device, nodes * width), result(device, nodes * width);
    expect_error<std::logic_error>([&] { jacobi.apply(rhs.data(), result.data()); }, "prepare");

    // Следующий шаг Ньютона может изменить диагональ. Перестраиваем тот же экземпляр.
    for (unsigned assembly = 0; assembly < 2; ++assembly) {
        diagonal.set(matrix);
        jacobi.prepare(diagonal.data());
        for (unsigned pass = 0; pass < 2; ++pass) {
            std::vector<double> x(nodes * width);
            for (std::size_t i = 0; i < x.size(); ++i)
                x[i] = static_cast<double>(i % 13) - 5.5 + pass;
            const auto b = multiply(matrix, x, width);
            rhs.set(b);
            // BiCGSTAB может переиспользовать буфер невязки. Сверяем оба варианта записи.
            Values& out = pass ? rhs : result;
            jacobi.apply(rhs.data(), out.data());
            const auto& solved = out.read();
            const auto product = multiply(matrix, solved, width);
            for (std::size_t i = 0; i < x.size(); ++i) {
                require(std::abs(solved[i] - x[i]) < 4e-13, "wrong correction");
                double scale = 0.;
                for (unsigned c = 0; c < width; ++c)
                    scale += std::abs(matrix[i * width + c] * solved[(i / width) * width + c]);
                require(std::abs(product[i] - b[i]) <= 2e-14 * scale, "large linear residual");
            }
        }
        require(diagonal.read() == matrix, "prepare modified the assembled Jacobian");
        for (double& value : matrix)
            value *= 1.25;
    }
    // Среди нескольких испорченных блоков оба устройства должны назвать первый.
    for (const double invalid : {0., std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        auto bad = matrix;
        for (const std::size_t node : {17u, 933u})
            std::fill_n(bad.begin() + node * width * width, width, invalid);
        diagonal.set(bad);
        expect_error<std::domain_error>([&] { jacobi.prepare(diagonal.data()); }, "node 17");
        expect_error<std::logic_error>([&] { jacobi.apply(rhs.data(), result.data()); }, "prepare");
    }
    // После сбоя переходим к случаю, где остался только временной член BDF1.
    // Для D = (V/dt) I поправка равна (dt/V) r во всех компонентах.
    std::fill(matrix.begin(), matrix.end(), 0.);
    for (std::size_t row = 0; row < nodes * width; ++row)
        matrix[row * width + (row % width)] = 8.;
    diagonal.set(matrix);
    jacobi.prepare(diagonal.data());
    rhs.set(std::vector<double>(nodes * width, 2.));
    jacobi.apply(rhs.data(), result.data());
    for (double z : result.read())
        require(z == .25, "incorrect inverse of the BDF1 time term");
}

// Сравнивает результат блочного предобусловливания на CPU и GPU.
void check_device(int device) {
    BlockJacobi empty(0, 3, device); // MPI-процесс без собственных узлов проходит те же вызовы.
    empty.prepare(nullptr);
    empty.apply(nullptr, nullptr);
    for (unsigned dimension = 1; dimension <= 3; ++dimension)
        for (bool use_matrix_diagonal : {false,true})
            check_blocks(device, dimension, use_matrix_diagonal);
    std::cout << "Block Jacobi passed on " << (device < 0 ? "CPU" : "GPU " + std::to_string(device)) << '\n';
    test_csgs(device);
    test_lf_edge_jacobian(device);
    test_spatial_jacobian(device);
    test_bdf1_linear_system(device);
    test_bdf1_step(device);
#ifdef AERO_HAS_MULTIGRID
    test_steady_multigrid(device);
    std::cout << "Steady multigrid passed on " << (device < 0 ? "CPU" : "GPU " + std::to_string(device)) << '\n';
#endif
}
}

// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main(int argc, char** argv) {
    try {
        const bool require_gpu = argc == 2 && std::string(argv[1]) == "--require-cuda";
        require(argc == 1 || require_gpu, "Usage: implicit_tests [--require-cuda]");
        expect_error<std::invalid_argument>([] { BlockJacobi bad(1, 4); }, "dimension");
        expect_error<std::length_error>([] { BlockJacobi bad(std::numeric_limits<aero::MeshIndex>::max(), 3); },
                                        "too many nodes");
        check_device(-1);
        int count = 0;
#if defined(AERO_WITH_KOKKOS_GPU)
        const auto status = aero::gpu::count(&count);
        if (status != aero::gpu::Status::NoDevice && status != aero::gpu::Status::Unavailable)
            aero::gpu::checked(status);
        for (int device = 0; device < count; ++device) {
            aero::gpu::DeviceScope on_card(device);
            aero::gpu::checked(on_card.status);
            check_device(device);
        }
#endif
        require(!require_gpu || count > 0, "GPU tests required but no device was tested");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
