#include "bc_test_case.h"
#include "../../qa/test_checks.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace {
using namespace aero;
std::vector<BcTestCase> cases;
BoundaryPatches patches; // Одна таблица для тех же проверок на CPU и GPU.
BoundaryConfig parse(const std::string& text, unsigned d) {
    std::istringstream input(text);
    return read_boundary(input, d, patches);
}
using aero_test::require;
// В тесте γ = 2. В 2D и 3D поворачиваем нормаль и задаём касательную скорость.
void state(unsigned d, double rho, double p, double un, double* u) {
    double vel[3] = {un, 0, 0};
    if (d > 1) {
        vel[0] = .6 * un - .8 * .4;
        vel[1] = .8 * un + .6 * .4;
    }
    if (d == 3)
        vel[2] = -.3;
    u[0] = rho;
    u[d + 1] = p;
    for (unsigned k = 0; k < d; ++k) {
        u[1 + k] = rho * vel[k];
        u[d + 1] += .5 * rho * vel[k] * vel[k];
    }
}
BcTestCase fixture(unsigned d, const std::string& block, double un) {
    BcTestCase t = {};
    t.config = parse(block, d);
    t.dimension = d;
    t.gamma = 2;
    t.area[0] = d == 1 ? 5 : 3;
    t.area[1] = d == 1 ? 0 : 4;
    state(d, 1, 2, un, t.current);
    return t;
}
void run(BcTestCase t, const std::string& label) {
    EulerFluxResult got = {};
    got.flux[0] = 123;
    const BoundaryStatus status =
        boundary_flux(t.dimension, t.config, patches.free_rules.data(), t.current,
                      t.initial_available ? t.initial : nullptr, t.area, t.gamma, &got);
    require(status == t.expected_status, label + " status");
    if (status == BoundaryStatus::Success) {
        require(got.components == t.dimension + 2, label + " components");
        for (unsigned k = 0; k < got.components; ++k)
            require(std::fabs(got.flux[k] - t.expected.flux[k]) < 1e-11, label + " flux");
        require(std::fabs(got.max_wave_speed - t.expected.max_wave_speed) < 1e-11, label + " speed");
    } else
        require(got.flux[0] == 123, label + " error wrote flux");
    cases.push_back(t);
}
void free_case(BcTestCase t, double rho, double p, double un, const std::string& label) {
    double u[5] = {}, speed;
    state(t.dimension, rho, p, un, u);
    require(euler_flux(t.dimension, u, t.area, t.gamma, t.expected.flux, &speed) == EulerStatus::Success,
            label);
    double ignore[5], inside_speed;
    require(euler_flux(t.dimension, t.current, t.area, t.gamma, ignore, &inside_speed) ==
                EulerStatus::Success,
            label);
    t.expected.max_wave_speed = std::max(speed, inside_speed);
    for (unsigned k = 0; k < t.dimension + 2; ++k)
        t.expected.flux[k] *= 5;
    run(t, label);
}
void error(BcTestCase t, BoundaryStatus s) {
    t.expected_status = s;
    run(t, "invalid input");
}
// Одну и ту же строку superin можно сочетать с любыми дозвуковыми правилами.
std::string free_block(const std::string& superin, const std::string& inflow, const std::string& outflow) {
    return "bc_type free\nsuperin " + superin + "\ninflow " + inflow + "\noutflow " + outflow;
}
void dimension(unsigned d) {
    for (int sign : {-1, 1}) {
        const double un = sign * .6;
        for (std::string mode : {"base_p", "base_u", "base_rho"}) {
            for (std::string source : {"", " initial", " explicit"}) {
                const bool unchanged = source.empty();
                // Выбираем состояние с K = 2 и тем же выходящим инвариантом.
                const double rho = unchanged ? 1 : 1.1025;
                const double p = 2 * rho * rho;
                const double ub = un + 4 * (1 - std::sqrt(rho));
                const double scalar = mode == "base_p" ? p : (mode == "base_u" ? ub : rho);
                std::ostringstream literal;
                literal.precision(17);
                literal << scalar;
                const std::string rule = mode + (source == " explicit" ? " " + literal.str() : source);
                const std::string block = free_block("initial", sign < 0 ? rule : "base_p initial",
                                                     sign > 0 ? rule : "base_u initial");
                BcTestCase t = fixture(d, block, un);
                if (source == " initial") {
                    t.initial_available = true;
                    // Энтропию и касательные данные для этой проверки берём из текущего состояния.
                    state(d, mode == "base_rho" ? rho : .7, mode == "base_p" ? p : 3,
                          mode == "base_u" ? ub : 1.1, t.initial);
                }
                free_case(t, rho, p, ub, rule);
            }
        }
    }
    // Все четыре режима на одной метке: меняется только скорость внутреннего газа.
    const auto split = free_block("initial", "base_p initial", "base_u 0.4");
    BcTestCase t = fixture(d, split, -.6);
    error(t, BoundaryStatus::MissingInitial);
    t.initial_available = true;
    state(d, 3, 2, -5, t.initial);
    free_case(t, 1, 2, -.6, "subsonic inflow takes only initial pressure");
    free_case(fixture(d, split, .6), 1.05 * 1.05, 2 * std::pow(1.05, 4), .4, "subsonic outflow uses target");
    free_case(fixture(d, split, 3), 1, 2, 3, "supersonic outflow ignores all rules");
    t = fixture(d, split, -4);
    error(t, BoundaryStatus::MissingInitial);
    t.initial_available = true;
    state(d, 2, 3, -5, t.initial);
    free_case(t, 2, 3, -5, "supersonic inflow takes full initial state");
    free_case(fixture(d, free_block("rho=2 p=3", "base_p", "base_u"), -4), 2, 3, -4, "supersonic overrides");
    error(fixture(d, free_block("p=100", "base_p", "base_u"), -4), BoundaryStatus::IncompatibleRegime);
    error(fixture(d, free_block("", "base_p", "base_u 10"), .6), BoundaryStatus::InvalidTarget);
    t = fixture(d, "bc_type wall", -.6);
    for (unsigned k = 0; k < d; ++k)
        t.expected.flux[k + 1] = 2 * t.area[k];
    t.expected.max_wave_speed = 2.6;
    run(t, "wall");
    t.area[0] = t.area[1] = 0;
    error(t, BoundaryStatus::InvalidArgument);
}
}
void test_periodic(bool require_gpu);
int main(int argc, char** argv) {
    try {
        for (unsigned d = 1; d <= 3; ++d)
            dimension(d);
        // В этом газе c=2. Разные цели выявляют ошибку на любой из трёх точных границ режимов.
        const auto at_threshold = free_block("ux=-6", "base_p initial", "base_u initial");
        free_case(fixture(1, at_threshold, -2), 1, 2, -6, "negative sonic equality selects superin");
        free_case(fixture(1, at_threshold, 2), 1, 2, 2, "positive sonic equality uses interior");
        auto stopped = fixture(1, free_block("initial", "base_p initial", "base_u"), 0);
        free_case(stopped, 1, 2, 0, "zero selects subsonic outflow");
        // Ключ initial задаёт основу, именованные компоненты поверх неё имеют приоритет.
        auto override = fixture(1, free_block("initial ux=-6", "base_u", "base_u"), -4);
        override.initial_available = true;
        state(1, 2, 3, -5, override.initial);
        free_case(override, 2, 3, -6, "superin overrides initial velocity");
        parse("outflow base_u\nsuperin\ninflow base_p\nbc_type free", 1);
        const auto valid = free_block("", "base_p", "base_u");
        for (const std::string& s : std::vector<std::string>{
                 "bc_type free\ninflow base_p\noutflow base_u", "bc_type free\nsuperin\noutflow base_u",
                 "bc_type free\nsuperin\ninflow base_p", valid + "\nsuperin", valid + "\ninflow base_u",
                 valid + "\noutflow base_p", "bc_type superout", "bc_type superin", "bc_type wall\nsuperin",
                 "bc_type wall\ninflow base_u", "bc_type free\ninflow superin\noutflow base_u",
                 free_block("", "base_u", "superout"), free_block("uz=1", "base_p", "base_u"),
                 free_block("rho=0", "base_p", "base_u"), free_block("p=2 p=1", "base_p", "base_u"),
                 free_block("", "base_u nan", "base_p"), "bc_type wall\nbc_type wall"}) {
            try {
                parse(s, 2);
            } catch (const std::invalid_argument&) {
                continue;
            }
            throw std::runtime_error("invalid syntax accepted: " + s);
        }
        std::cout << cases.size() << " CPU boundary flux cases and parser tests passed.\n";
        const bool required = argc == 2 && std::string(argv[1]) == "--require-cuda";
        test_periodic(required);
#ifdef AERO_WITH_CUDA
        return aero::check_bc_devices(cases.data(), cases.size(), patches.free_rules.data(),
                                      patches.free_rules.size(), required);
#else
        return required ? 1 : 0;
#endif
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
