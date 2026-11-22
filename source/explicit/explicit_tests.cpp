#include "explicit_test_problem.h"
#include "../../tests/test_checks.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace aero;
using aero_test::require;
void near(double value, double expected) {
    require(std::fabs(value - expected) < 2e-12 * (1 + std::fabs(expected)),
            "incorrect stage, update or conserved integral");
}
void evaluate(TestProblem problem, double time, const ExplicitField& field, double* residual) {
#ifdef AERO_WITH_KOKKOS_GPU
    if (field.backend == ExplicitBackend::Cuda) {
        device_test_residual(problem, time, field, residual);
        return;
    }
#endif
    for (std::size_t k = 0; k < field.nodes * field.components; ++k)
        residual[k] = test_residual(problem, time, field, k);
}
double convergence_error(ExplicitBackend backend, int device, ExplicitScheme scheme, unsigned n) {
    const double volume = .23, initial = 1, dt = 1. / n;
    ExplicitIntegrator solver(1, 1, &volume, &initial, backend, device);
    for (unsigned i = 0; i < n; ++i) {
        unsigned calls = 0;
        double start = i * dt;
        solver.step(scheme, start, dt, [&](double time, const ExplicitField& f, double* r) {
            const double fractions[] = {0, 1, .5};
            require(calls < 3, "too many residual evaluations");
            near(time, start + dt * fractions[calls++]);
            evaluate(TestProblem::Forced, time, f, r);
        });
        require(calls == (scheme == ExplicitScheme::Euler ? 1u : 3u), "incorrect residual count");
    }
    return std::fabs(solver.state()[0] - (2 * std::exp(1.) - 2));
}

// Четыре узла периодической линии с интервалами 0.5, 1.5, 1 и 1.
// Сверяем каждую компоненту с полиномом устойчивости RK и проверяем сохранение суммы с весами объёмов.
void periodic(ExplicitBackend backend, int device, ExplicitScheme scheme) {
    const double v[] = {.75, 1, 1.25, 1}, dt = .1;
    const std::vector<double> q = {1, .1, 3, 2, -.3, 4, .8, .9, 2, 1.5, .4, 5};
    const auto apply = [&](const std::vector<double>& u) {
        std::vector<double> du(u.size());
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned c = 0; c < 3; ++c)
                du[3 * i + c] = (u[3 * ((i + 3) % 4) + c] - u[3 * i + c]) / v[i];
        return du;
    };
    const auto a = apply(q), aa = apply(a), aaa = apply(aa);
    ExplicitIntegrator solver(4, 3, v, q.data(), backend, device);
    solver.step(scheme, 0, dt,
                [](double t, const ExplicitField& f, double* r) { evaluate(TestProblem::Ring, t, f, r); });
    const auto out = solver.state();
    for (unsigned c = 0; c < 3; ++c) {
        double balance = 0;
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned k = 3 * i + c;
            double ref = q[k] + dt * a[k];
            if (scheme == ExplicitScheme::SSPRK3)
                ref += dt * dt * (.5 * aa[k] + dt * aaa[k] / 6);
            near(out[k], ref);
            balance += v[i] * (out[k] - q[k]);
        }
        near(balance, 0);
    }
}
void prepared_step(ExplicitBackend backend, int device, ExplicitScheme scheme) {
    const double volume = 2, initial = 1, start = .5, dt = .125;
    const double first = -volume * (initial + start);
    ExplicitIntegrator baseline(1, 1, &volume, &initial, backend, device);
    ExplicitIntegrator solver(1, 1, &volume, &initial, backend, device);
    // Храним заранее вычисленную величину там же, где работает интегратор.
    ExplicitIntegrator cached(1, 1, &volume, &first, backend, device);
    baseline.step(scheme, start, dt, [](double t, const ExplicitField& f, double* r) {
        evaluate(TestProblem::Forced, t, f, r);
    });
    unsigned calls = 0;
    cached.inspect([&](const ExplicitField& prepared) {
        solver.step(
            scheme, start, dt,
            [&](double t, const ExplicitField& f, double* r) {
                require(calls < 2, "first residual evaluated twice");
                near(t, start + (calls == 0 ? dt : .5 * dt));
                ++calls;
                evaluate(TestProblem::Forced, t, f, r);
            },
            prepared.values);
    });
    require(calls == (scheme == ExplicitScheme::Euler ? 0u : 2u), "incorrect number of later stages");
    require(solver.state() == baseline.state(), "precomputed first residual changed the step");
}
#ifdef AERO_WITH_KOKKOS_GPU
void nested_devices() {
    const int caller = current_test_device(), other = caller == 0 ? 1 : 0;
    const double one = 1;
    {
        ExplicitIntegrator outer(1, 1, &one, &one, ExplicitBackend::Cuda, other);
        ExplicitIntegrator inner(1, 1, &one, &one, ExplicitBackend::Cuda, caller);
        require(current_test_device() == caller, "construction changed the caller's device");
        outer.inspect([&](const ExplicitField&) {
            require(current_test_device() == other, "outer device not active");
            try {
                inner.inspect([&](const ExplicitField&) {
                    require(current_test_device() == caller, "inner device not active");
                    throw std::domain_error("rejected by reader");
                });
            } catch (const std::domain_error&) {
            }
            require(current_test_device() == other, "inner inspection did not restore the outer device");
        });
        require(current_test_device() == caller, "inspection changed the caller's device");
    }
    require(current_test_device() == caller, "destruction changed the caller's device");
}
#endif
// Читаем поле непосредственно в сборщике: так проверяются и промежуточные Y1, Y2, и время каждой невязки.
void one_node_stage_values() {
    const double volume = 2, initial = 10, dt = .5;
    const double times[] = {0, .5, .25}, states[] = {10, 9, 9.25}, residuals[] = {4, 8, 2};
    for (auto scheme : {ExplicitScheme::Euler, ExplicitScheme::SSPRK3}) {
        ExplicitIntegrator integrator(1, 1, &volume, &initial, ExplicitBackend::Cpu);
        unsigned call = 0;
        integrator.step(scheme, 0, dt, [&](double time, const ExplicitField& field, double* residual) {
            require(call < 3, "unexpected extra stage");
            near(time, times[call]);
            near(field.values[0], states[call]);
            residual[0] = residuals[call++];
        });
        require(call == (scheme == ExplicitScheme::Euler ? 1u : 3u), "missing stage");
        near(integrator.state()[0], scheme == ExplicitScheme::Euler ? 9.0 : 55.0 / 6);
    }
}
void exercise(ExplicitBackend backend, int device) {
    for (auto scheme : {ExplicitScheme::Euler, ExplicitScheme::SSPRK3}) {
        const double coarse = convergence_error(backend, device, scheme, 16);
        const double fine = convergence_error(backend, device, scheme, 32);
        const double p = std::log(coarse / fine) / std::log(2.);
        const double order = scheme == ExplicitScheme::Euler ? 1. : 3.;
        require(std::fabs(p - order) < .1, "incorrect temporal order");
        std::cout << (scheme == ExplicitScheme::Euler ? "Euler" : "SSPRK3") << " observed order " << p
                  << "\n";
        periodic(backend, device, scheme);
        prepared_step(backend, device, scheme);
    }
    const double v = 2, initial = 1;
    ExplicitIntegrator solver(1, 1, &v, &initial, backend, device);
    for (bool throw_in_callback : {true, false}) {
        unsigned stage = 0;
        bool failed = false;
        try {
            solver.step(ExplicitScheme::SSPRK3, 0, .1, [&](double t, const ExplicitField& f, double* r) {
                if (++stage == 2 && throw_in_callback)
                    throw std::domain_error("rejected by assembler");
                evaluate(stage == 2 ? TestProblem::Invalid : TestProblem::Forced, t, f, r);
            });
        } catch (const std::domain_error&) {
            failed = true;
        }
        require(failed && stage == 2 && solver.state()[0] == initial, "failed RK stage committed");
    }
    solver.step(ExplicitScheme::Euler, 0, .125,
                [](double t, const ExplicitField& f, double* r) { evaluate(TestProblem::Forced, t, f, r); });
    near(solver.state()[0], 1.125);

    // Размер задачи превышает порог OpenMP и покрывает несколько блоков CUDA с неполным последним блоком.
    const std::size_t nodes = 2049, components = 3;
    std::vector<double> volumes(nodes), u(nodes * components);
    for (std::size_t i = 0; i < nodes; ++i)
        volumes[i] = .5 + .125 * static_cast<double>(i % 5);
    for (std::size_t k = 0; k < u.size(); ++k)
        u[k] = static_cast<double>(k % 127) - 32;
    ExplicitIntegrator large(nodes, components, volumes.data(), u.data(), backend, device);
    large.step(ExplicitScheme::Euler, .5, .25,
               [](double t, const ExplicitField& f, double* r) { evaluate(TestProblem::Forced, t, f, r); });
    const auto result = large.state();
    for (std::size_t k = 0; k < u.size(); ++k)
        near(result[k], u[k] + .25 * (u[k] + .5));
}
}
int main(int argc, char** argv) {
    try {
        one_node_stage_values();
        exercise(aero::ExplicitBackend::Cpu, 0);
        std::cout << "CPU explicit tests passed.\n";
        const int count = aero::ExplicitIntegrator::cuda_device_count();
#ifdef AERO_WITH_KOKKOS_GPU
        if (count >= 2)
            nested_devices();
#endif
        if (argc == 2 && std::string(argv[1]) == "--require-cuda")
            require(count > 0, "CUDA device required");
        for (int i = 0; i < count; ++i) {
            exercise(aero::ExplicitBackend::Cuda, i);
            std::cout << "CUDA explicit tests passed on GPU " << i << ".\n";
        }
        const double one = 1;
        aero::ExplicitIntegrator automatic(1, 1, &one, &one);
        require(automatic.backend() == (count ? aero::ExplicitBackend::Cuda : aero::ExplicitBackend::Cpu),
                "incorrect auto backend");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
