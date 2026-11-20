#include "bdf1_step.h"
#include "../residual/residual_detail.h"
#include "../eulersolver/lax_friedrichs.h"
#include "../../qa/test_field.h"
#include <cmath>
#include <iostream>
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#endif

// Два периодических объёма разного размера. Давление одинаково, скорость нулевая,
// а перепад плотности даёт известный поток массы LF на обоих рёбрах.
void test_bdf1_step(int device) {
    using namespace aero;
    using namespace implicit;
    using namespace residual_detail;
    using aero_test::require;
    const Surface edges[] = {{0,1,0,{1.,0.,0.},1.,true}, {1,0,0,{1.,0.,0.},1.,true}};
    const Index offsets[] = {0,2,4};
    const OrientedFace incidence[] = {OrientedFace::outward(0),OrientedFace::inward(1),
                                      OrientedFace::inward(0),OrientedFace::outward(1)};
    Geometry host{};
    host.dim = 1;
    host.nodes = host.edges = host.faces = 2;
    host.surfaces = edges;
    host.offsets = offsets;
    host.incidence = incidence;
    host.options.gamma = 1.4;
    Geometry active = host;
    EulerFluxResult flux[2]{};
    EulerFluxResult* active_flux = flux;
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays arrays;
    if (device >= 0) {
        arrays.device = device;
        active.surfaces = arrays.allocate(2, edges);
        active.offsets = arrays.allocate(3, offsets);
        active.incidence = arrays.allocate(4, incidence);
        active_flux = arrays.allocate<EulerFluxResult>(2);
    }
#endif
    aero_test::TestField state(device,6), volumes(device,2);
    const std::vector<double> initial{1.,0.,2.5, 1.4,0.,2.5};
    state.set(initial);
    volumes.set({.4,.6});
    BdfStep solver(host, active, device);
    NewtonSettings controls;
    controls.newton_relative = 1.e-10;
    double dt = .15;
    controls.second_order = true; // Первый шаг BDF2 совпадает с неявным Эйлером.
    const auto refresh = [&] {
        const auto& u = state.read();
        for (unsigned i = 0; i < 2; ++i) {
            const auto& e = edges[i];
            require(lax_friedrichs_conservative(1, &u[3*e.begin], &u[3*e.end], e.area, 1.4, &flux[i])
                    == EulerStatus::Success, "bad test state");
        }
#ifdef AERO_WITH_KOKKOS_GPU
        if (device >= 0) gpu::checked(gpu::copy(active_flux, flux, sizeof(flux), gpu::Copy::ToDevice));
#endif
    };
    const auto advance = [&] {
        refresh();
        return solver.advance(state.data(), nullptr, volumes.data(), active_flux, dt, controls, refresh);
    };
    // Одной поправки недостаточно. Проверяем, что отказ возвращает оба узла в исходное состояние.
    controls.max_iterations = 1;
    bool rejected = false;
    try { advance(); } catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("Newton iteration limit") != std::string::npos;
    }
    require(rejected && state.read() == initial, "failed Newton step changed the accepted state");
    controls.max_iterations = 20;
    const auto result = advance();
    const auto u = state.read();
    // Независимо восстанавливаем невязку из принятой плотности и скорости звука.
    // Чистый отток массы из первого объёма равен a*(rho_left-rho_right).
    const double speed = std::sqrt(1.4 / std::min(u[0],u[3]));
    const double net = speed * (u[0] - u[3]);
    const double norm = std::max(std::abs(.4/dt*(u[0]-initial[0])+net),
                                 std::abs(.6/dt*(u[3]-initial[3])-net));
    const double relative = norm / (std::sqrt(1.4)*(initial[3]-initial[0]));
    require(result.iterations > 1 && relative <= controls.newton_relative &&
            std::abs(relative-result.relative_residual) < 5.e-15, "wrong final Newton residual");
    require(std::abs(.4*(u[0]-initial[0]) + .6*(u[3]-initial[3])) < 1.e-14, "BDF1 lost mass");
    // На втором шаге меняем dt. Проверка конечного баланса использует коэффициенты
    // интерполяционного полинома непосредственно по трём моментам времени.
    const double old_dt = dt;
    dt = .09;
    BdfHistory history;
    solver.export_history(history);
    require(history.dt == old_dt && history.previous == initial, "wrong previous layer");
    controls.csgs = true;
    controls.roe = true;
    controls.max_iterations = 1;
    rejected = false;
    try {
        advance();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    BdfHistory after_failure;
    solver.export_history(after_failure);
    require(rejected && state.read() == u && after_failure.previous == history.previous &&
                after_failure.dt == old_dt,
            "failed BDF2 step damaged history");
    controls.max_iterations = 20;
    advance();
    const auto next = state.read();
    const double new_flux = std::sqrt(1.4 / std::min(next[0], next[3])) * (next[0] - next[3]);
    for (unsigned i = 0; i < 2; ++i) {
        const unsigned k = 3 * i;
        const double derivative = (2 * dt + old_dt) / (dt * (dt + old_dt)) * next[k] -
                                  (dt + old_dt) / (dt * old_dt) * u[k] +
                                  dt / (old_dt * (dt + old_dt)) * initial[k];
        require(std::abs((i ? .6 : .4) * derivative + (i ? -new_flux : new_flux)) < 5.e-10,
                "variable step BDF2 balance");
    }
    // При перезапуске считываем U_(n-1) и предыдущий dt. Получаем тот же шаг.
    state.set(u);
    solver.import_history(history);
    advance();
    require(state.read() == next, "BDF2 history round trip changed the solution");
    controls.second_order = false;
    // Однородное состояние должно приниматься без вызова линейного решателя.
    state.set({1., 0., 2.5, 1., 0., 2.5});
    const auto uniform = advance();
    require(uniform.iterations == 0 && uniform.relative_residual == 0., "uniform state needs no correction");
    require(solver.spatial_rate_norm(volumes.data(), active_flux) == 0., "uniform spatial residual");
    state.set(initial);
    refresh();
    const double spatial_rate = solver.spatial_rate_norm(volumes.data(), active_flux);
    require(std::abs(spatial_rate - std::sqrt(1.4) * (initial[3] - initial[0]) / .4) < 1.e-14,
            "steady criterion must divide the physical flux imbalance by node volume");
    // Сравниваем с полной сборкой при смене шага, коэффициента BDF и после
    // отклонённой пробы Пикара. Число сборок проверяем отдельно от точности поля.
    std::vector<double> full_assembly_field;
    for (unsigned interval : {0u, 1u, 3u}) {
        BdfStep cached(host, active, device);
        controls.second_order = true;
        controls.roe = false;
        controls.csgs = true;
        controls.jacobian_update_every = interval;
        state.set(initial);
        bool rejected_trial = false;
        for (unsigned n = 0; n < 5; ++n) {
            dt = n % 2 ? .014 : .023;
            refresh();
            const auto trial = [&] { rejected_trial = true; return false; };
            const auto step = cached.advance(state.data(), nullptr, volumes.data(), active_flux, dt,
                                              controls, refresh, trial);
            const unsigned expected = interval ? (n % interval == 0 ? 1u : 0u) : step.iterations;
            require(step.iterations && step.jacobian_assemblies == expected, "wrong Jacobian update interval");
            require(step.relative_residual <= controls.newton_relative, "cached Jacobian failed residual check");
        }
        require(rejected_trial, "predictor rejection path missing");
        const auto final_field = state.read();
        if (!interval) full_assembly_field = final_field;
        else for (unsigned k = 0; k < final_field.size(); ++k)
            require(std::abs(final_field[k] - full_assembly_field[k]) < 2.e-10, "reused matrix changed the solution");
    }
    controls.jacobian_update_every = 0;
    // Для этой пары объёмов масса постоянна. Плотность x первого узла удовлетворяет
    // dx/dt = sqrt(gamma)*(m-x)/(V0*V1*sqrt(x)), m — средняя плотность.
    // Сравниваем с интегралом этого ОДУ, чтобы ошибка по пространству не скрыла порядок BDF2.
    const double mass = .4 * initial[0] + .6 * initial[3];
    const auto integral = [&](double x) {
        return 2. * (std::sqrt(mass) * std::atanh(std::sqrt(x / mass)) - std::sqrt(x));
    };
    const double end_time = .12;
    const double target = integral(initial[0]) + end_time * std::sqrt(1.4) / (.4 * .6);
    double low = initial[0], high = mass;
    for (unsigned k = 0; k < 54; ++k) {
        const double mid = .5 * (low + high);
        if (integral(mid) < target)
            low = mid;
        else
            high = mid;
    }
    double coarse_error = 0., ratio = 0.;
    controls.second_order = true;
    controls.roe = false;
    controls.csgs = false;
    for (unsigned steps : {12u, 24u, 48u}) {
        BdfStep refined(host, active, device);
        state.set(initial);
        for (unsigned n = 0; n < steps; ++n) {
            dt = end_time / steps * (n % 2 ? 1.25 : .75);
            refresh();
            refined.advance(state.data(), nullptr, volumes.data(), active_flux, dt, controls, refresh);
        }
        const double error = std::abs(state.read()[0] - .5 * (low + high));
        if (coarse_error > 0.) {
            ratio = coarse_error / error;
            require(ratio > 3.4 && ratio < 4.8, "BDF2 temporal convergence on alternating steps");
        }
        coarse_error = error;
    }
    std::cout << "BDF2 variable-step refinement ratio = " << ratio << '\n';
    std::cout << "BDF1/BDF2 Newton passed on " << (device < 0 ? "CPU" : "GPU " + std::to_string(device))
              << ", iterations = " << result.iterations << ", relative residual = " << relative << '\n';
}
