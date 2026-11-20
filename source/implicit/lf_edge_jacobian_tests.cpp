#include "roe_edge_jacobian.h"
#include "lf_edge_jacobian.h"
#include "../eulersolver/roe.h"
#include "../eulersolver/lax_friedrichs.h"
#include "../../qa/device_cases.h"
#include "../../qa/test_checks.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../../qa/device_test_buffer.h"
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {
using aero::device_tests::Case;
using aero::implicit::EdgeFluxJacobian;
using aero_test::require;

// Одинаковые исходные состояния для аналитической матрицы и явного потока.
aero::euler_detail::FaceStates face(const Case& test, double upwind) {
    return {test.dimension, test.left, test.right, test.direction, test.gamma, upwind};
}

// Эталон для конечных разностей — штатный численный поток, включая множитель площади.
aero::EulerFluxResult flux(const Case& test, double area, double upwind, bool roe=false, const Case* frozen=nullptr) {
    aero::EulerFluxResult value;
    using namespace aero;
    using namespace euler_detail;
    InterfaceGas gas;
    double fl[5], fr[5], speed;
    require(interface_fluxes(test.dimension,test.left,test.right,test.direction,test.gamma,fl,fr,gas,speed)
            == EulerStatus::Success,"invalid test states");
    // При численной проверке замороженной матрицы сохраняем её базис от невозмущённой пары.
    // Физические потоки и сам скачок в обоих случаях относятся к возмущённой паре.
    if (frozen) require(interface_speed(test.dimension,frozen->left,frozen->right,test.direction,
                                        test.gamma,gas,speed)==EulerStatus::Success,"bad frozen state");
    const auto states=face(test,upwind);
    const auto status=roe ? roe_detail::RoeFlux::combine(states,gas,fl,fr,speed,&value)
                          : RusanovFlux::combine(states,gas,fl,fr,speed,&value);
    require(status==EulerStatus::Success,"invalid reference flux");
    for (unsigned k = 0; k < value.components; ++k)
        value.flux[k] *= area;
    return value;
}

EdgeFluxJacobian jacobian(const Case& test, double area, double upwind, int device, bool simplified = false, bool roe = false) {
    EdgeFluxJacobian blocks = {};
#ifdef AERO_WITH_KOKKOS_GPU
    if (device >= 0) {
        aero_test::DeviceBuffer<EdgeFluxJacobian> result(1);
        const char* error = aero::device_tests::edge_blocks(test, area, upwind, result.data, simplified, roe);
        if (error)
            throw std::runtime_error(error);
        result.download(&blocks, 1);
        require(blocks.width == test.dimension + 2, "GPU rejected valid LF input");
        return blocks;
    }
#else
    (void)device;
#endif
    const auto states=face(test,upwind);
    require(roe ? aero::implicit::roe_edge_jacobian(states,area,blocks,simplified)
                : aero::implicit::lf_edge_jacobian(states,area,blocks,simplified), "invalid edge Jacobian");
    return blocks;
}

void near(double actual, double expected, double tolerance, const char* check) {
    if (!(std::abs(actual - expected) <= tolerance * (1. + std::abs(expected))))
        throw std::runtime_error(std::string(check) + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
}

// Возмущаем по очереди каждую компоненту каждого узла. Сверяем одновременно
// две строки матрицы, поскольку один и тот же поток входит в них с разным знаком.
void check_columns(const Case& test, double area, double upwind, int device, double tolerance, bool simplified = false, bool roe = false) {
    const EdgeFluxJacobian blocks = jacobian(test, area, upwind, device, simplified, roe);
    const unsigned width = test.dimension + 2;
    double matrix[4][25];
    for (auto& block : matrix)
        std::fill_n(block, 25, 1.); // Проверяем добавление к уже записанным вкладам соседних рёбер.
    blocks.add_to(matrix[0], matrix[1], matrix[2], matrix[3]);
    for (unsigned side = 0; side < 2; ++side)
        for (unsigned col = 0; col < width; ++col) {
            Case plus = test, minus = test;
            double& high = (side ? plus.right : plus.left)[col];
            double& low = (side ? minus.right : minus.left)[col];
            const double h = 1.e-6 * std::max(1., std::abs(high));
            high += h;
            low -= h;
            const auto fp = flux(plus, area, upwind, roe, simplified ? &test : nullptr);
            const auto fm = flux(minus, area, upwind, roe, simplified ? &test : nullptr);
            for (unsigned row = 0; row < width; ++row) {
                double expected = (fp.flux[row] - fm.flux[row]) / (high - low);
                const unsigned k = row * width + col;
                near(matrix[side][k] - 1., expected, tolerance, "left residual");
                near(matrix[2 + side][k] - 1., -expected, tolerance, "right residual");
            }
        }
}

Case example(unsigned dimension) {
    Case test = {};
    test.dimension = dimension;
    test.gamma = 1.4;
    test.direction[0] = 2.;
    test.direction[1] = -3.;
    test.direction[2] = 1.;
    test.left[0] = 1.;
    test.right[0] = 1.25;
    for (unsigned k = 1; k <= dimension; ++k) {
        test.left[k] = .25 * k;
        test.right[k] = -.125 * k;
    }
    test.left[dimension + 1] = 4.;
    test.right[dimension + 1] = 2.5;
    return test;
}

// Перестановка концов вместе с разворотом нормали меняет знак всего потока.
// Проверяем это и в точке равенства скоростей, где выбор первой стороны нарушил бы симметрию.
void check_orientation(Case test, int device) {
    const auto forward = jacobian(test, 2.5, .7, device);
    std::swap(test.left, test.right);
    for (double& n : test.direction)
        n = -n;
    const auto reverse = jacobian(test, 2.5, .7, device);
    for (unsigned k = 0; k < forward.width * forward.width; ++k) {
        near(forward.left[k], -reverse.right[k], 1.e-13, "reverse right block");
        near(forward.right[k], -reverse.left[k], 1.e-13, "reverse left block");
    }
}
}

void test_lf_edge_jacobian(int device) {
    if (device < 0) {
        Case invalid = example(3);
        EdgeFluxJacobian previous = {};
        previous.width = 5;
        previous.left[0] = 123.;
        require(!aero::implicit::lf_edge_jacobian(face(invalid, 1.), -1., previous), "accepted negative area");
        require(!aero::implicit::lf_edge_jacobian(face(invalid, 2.), 1., previous), "accepted invalid upwind coefficient");
        invalid.right[0] = 0.;
        require(!aero::implicit::lf_edge_jacobian(face(invalid, 1.), 1., previous), "accepted zero density");
        require(previous.width == 5 && previous.left[0] == 123., "failed edge overwrote previous Jacobian");
    }
    for (unsigned dim = 1; dim <= 3; ++dim) {
        Case test = example(dim);
        for (double upwind : {0., .4, 1.}) {
            check_columns(test, 2.75, upwind, device, 2.e-8);
            check_columns(test, 2.75, upwind, device, 2.e-8, true);
            check_columns(test, 2.75, upwind, device, 2.e-8, false, true);
            check_columns(test, 2.75, upwind, device, 2.e-8, true, true);
            std::swap(test.left, test.right); // Максимум скорости теперь задаёт противоположная сторона.
            check_columns(test, .35, upwind, device, 2.e-8);
            std::swap(test.left, test.right);
        }
        check_orientation(test, device);
        // При нулевой средней нормальной скорости проверяем симметричную производную |u_n|.
        auto stagnant=test;
        std::fill_n(stagnant.left+1,dim,0.);
        std::fill_n(stagnant.right+1,dim,0.);
        check_columns(stagnant,1.,1.,device,3.e-6,false,true);
        // Удвоение всех консервативных переменных сохраняет скорость и скорость
        // звука. Получаем точное равенство аргументов max при ненулевом скачке U.
        for (unsigned k = 0; k < dim + 2; ++k)
            test.right[k] = 2. * test.left[k];
        check_orientation(test, device);
        check_columns(test, 1., 1., device, 3.e-6);
        // Отдельно проверяем излом модуля при нулевой скорости активной стороны.
        test = example(dim);
        std::fill_n(test.left + 1, dim, 0.);
        check_columns(test, 1., 1., device, 3.e-6);
    }
    std::cout << "LF and Roe edge Jacobians passed on " << (device < 0 ? "CPU" : "GPU " + std::to_string(device)) << '\n';
}
