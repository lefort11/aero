#include "bdf1_step.h"
#include "time_prediction.h"
#include "../residual/residual_detail.h"
#include "../eulersolver/lax_friedrichs.h"
#include "../../tests/test_field.h"
#include <cmath>
#include <iostream>
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_arrays.h"
#endif

// Узловой полином должен воспроизводить квадратичное поле на неравных шагах,
// в том числе после перехода через конец кольца компактных строк.
static void check_time_prediction() {
    using namespace aero::implicit;
    double slots[2]{};
    KrylovScratch space;
    space.count = 2;
    space.field[0] = slots;
    space.field[1] = slots+1;
    PredictionHistory history(space);
    *history.append(.5) = .25; // U(-0.5), этот слой будет вытеснен.
    *history.append(1.) = 0.;  // U(0).
    *history.append(2.) = 1.;  // U(1).
    double values[]{-1.,9.};   // Владелец хранит U(3) во втором узле.
    const aero::MeshIndex owned[]{1};
    TimePrediction prediction(owned,1,values);
    history.prepare(.5,2,prediction);
    prediction(0);
    aero_test::require(prediction.count==2 && std::abs(values[1]-12.25)<1.e-14 &&
                       values[0]==-1., "variable-step quadratic prediction and owned row");
    // Та же история может содержать интегральные балансы. Узел умножает
    // прогноз скорости на собственный шаг, а свободный слот получает исходный баланс.
    const double rate = 9., volumes[]{1.,2.};
    double steps[]{1.,.25};
    prediction.rate = &rate;
    prediction.save_rate = history.append(.5); // Перезаписываем самый старый из прочитанных слоёв.
    prediction.volumes = volumes;
    prediction.steps = {steps};
    prediction.dt = 1.5;
    values[1] = 3.;
    prediction(0);
    aero_test::require(std::abs(values[1]-(3.+.25*1.5*12.25/2.))<1.e-14 &&
                       *prediction.save_rate==9. && values[0]==-1.,"local rate prediction and ring alias");
    history.prepare(1000.,2,prediction);
    aero_test::require(!prediction.count,"unstable extrapolation weights");
    history.clear();
    history.prepare(.5,2,prediction);
    aero_test::require(!prediction.count,"stale prediction after Krylov solve");

    // Два узла испытывают разные изменения локального шага. Для второго узла
    // L(t)=t², текущий момент t=3, предыдущие моменты t=2 и t=0.
    double local_field[6]{}, balances[]{9.,0.,0.}, previous[]{4.,0.,0.}, oldest[]{0.,0.,0.};
    double local_intervals[]{1.,2.,0.};
    TimePrediction local(owned,3,local_field,volumes,{steps},2.);
    local.count=2;
    local.fields[0]=previous;
    local.fields[1]=oldest;
    local.intervals=local_intervals;
    local.rate=balances;
    local(0);
    aero_test::require(std::abs(local_field[3]-.5*12.25/2.)<1.e-14 &&
                       local_intervals[0]==.5 && local_intervals[1]==1.,"quadratic rate with nodal time intervals");
}

// Два периодических объёма разного размера. Давление одинаково, скорость нулевая,
// а перепад плотности даёт известный поток массы LF на обоих рёбрах.
void test_bdf1_step(int device) {
    using namespace aero;
    using namespace implicit;
    using namespace residual_detail;
    using aero_test::require;
    check_time_prediction();
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
    // Пересчитывает зависимые потоки или коэффициенты после изменения поля.
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
    // Выполняет один контрольный неявный шаг с выбранными параметрами.
    const auto advance = [&] {
        refresh();
        return solver.advance(state.data(), nullptr, volumes.data(), active_flux, dt, controls, refresh);
    };
    // Проверяем сами пробные сборки: счётчик Ньютона содержит только принятые
    // линейные поправки и потому не показывает, запускался ли предиктор.
    for (unsigned limit : {1u, 3u}) {
        for (bool without : {false, true}) {
            BdfStep checked(host,active,device);
            auto settings = controls;
            settings.without_predictor = without;
            settings.max_iterations = limit;
            settings.prediction_order = 6;
            state.set(initial);
            // Второй шаг использует предыдущий слой BDF2 и проверяет переключатель при наличии истории.
            for (unsigned n=0; n<2; ++n) {
                unsigned trials = 0;
                const auto trial = [&] { ++trials; refresh(); return true; };
                refresh();
                const auto result = checked.advance(state.data(),nullptr,volumes.data(),active_flux,.005,settings,refresh,trial);
                require((trials == 0) == without, "Newton limit must not select the predictor");
                require(result.iterations <= limit && (!without || result.iterations > 0),
                        "initial prediction must not consume Newton corrections");
            }
        }
    }
    state.set(initial);
    // Внешний буфер отката совпадает с U_n только до конца попытки BDF1.
    // При переходе на BDF2 история должна пережить перезапись внешнего буфера.
    {
        std::vector<std::vector<double>> reference;
        for (bool borrow : {false,true}) {
            aero_test::TestField backup(device,initial.size());
            Partition partition;
            if (borrow) partition.step_start=backup.data();
            BdfStep checked(host,active,device,partition);
            auto settings=controls;
            settings.second_order=false;
            state.set(initial);
            refresh();
            bool rejected=false;
            settings.max_iterations=1;
            try {
                checked.advance(state.data(),nullptr,volumes.data(),active_flux,dt,settings,
                                [] { throw LinearSolveFailure("borrowed rollback"); });
            } catch (const LinearSolveFailure&) { rejected=true; }
            require(rejected && state.read()==initial,"borrowed start restores rejected BDF1 step");
            settings.max_iterations=20;
            for (unsigned step=0;step<4;++step) {
                backup.set(std::vector<double>(initial.size(),-123.));
                settings.second_order=step!=0;
                refresh();
                checked.advance(state.data(),nullptr,volumes.data(),active_flux,dt,settings,refresh);
                if (!borrow) reference.push_back(state.read());
                else require(state.read()==reference[step],"borrowed start preserves BDF1/BDF2 transition");
                if (borrow && step==0)
                    require(checked.start_field()==backup.data() && backup.read()==initial,"BDF1 reuses supplied start field");
                if (borrow && step>0)
                    require(checked.start_field()!=backup.data(),"BDF2 owns both history fields");
            }
        }
        state.set(initial);
        refresh();
    }
    // Имитируем отказ линейного решателя после изменения пробного поля.
    // Повтор с половиной шага должен совпасть с отдельным расчётом того же шага.
    {
        BdfStep retried(host,active,device), direct(host,active,device);
        auto trial_controls = controls;
        trial_controls.second_order = false;
        trial_controls.max_iterations = 1;
        trial_controls.without_predictor = true; // Принудительный отказ относится к поправке Ньютона.
        state.set(initial);
        refresh();
        bool rejected = false;
        try {
            retried.advance(state.data(),nullptr,volumes.data(),active_flux,dt,trial_controls,
                            [] { throw LinearSolveFailure("test linear solve failure"); });
        } catch (const LinearSolveFailure&) { rejected = true; }
        require(rejected && state.read()==initial,"rollback restores initial gas at both nodes");
        refresh();
        retried.advance(state.data(),nullptr,volumes.data(),active_flux,.5*dt,trial_controls,refresh);
        const auto repeated = state.read();
        state.set(initial);
        refresh();
        direct.advance(state.data(),nullptr,volumes.data(),active_flux,.5*dt,trial_controls,refresh);
        require(state.read()==repeated,"retry matches direct half step");
        state.set(initial);
        refresh();
    }
    // Одной поправки мало для заданного допуска. Поле и история всё равно должны
    // перейти на следующий слой, а отчёт должен вернуть реально достигнутую невязку.
    controls.max_iterations = 1;
    {
        BdfStep limited(host, active, device);
        refresh();
        const auto step = limited.advance(state.data(), nullptr, volumes.data(), active_flux, dt, controls, refresh);
        BdfHistory history;
        limited.export_history(history);
        require(step.correction_iterations.size() == 1 && step.correction_iterations.front() == step.linear_iterations,
                "BiCGSTAB count for the applied Newton correction");
        require(step.iterations == 1 && std::isfinite(step.relative_residual) &&
                step.relative_residual > controls.newton_relative, "Newton iteration cap");
        require(state.read() != initial && history.dt == dt && history.previous == initial,
                "limited Newton step must accept the field and advance BDF2 history");
    }
    state.set(initial);
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
    // Если пространственный оператор отклоняет поле после поправки, шаг всё ещё
    // должен откатиться вместе с историей, даже на последней разрешённой итерации.
    bool rejected = false;
    refresh();
    try {
        solver.advance(state.data(), nullptr, volumes.data(), active_flux, dt, controls,
                       [] { throw std::domain_error("invalid trial field"); });
    } catch (const std::domain_error& error) {
        rejected = std::string(error.what()) == "invalid trial field";
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
    // Для двух разных псевдошагов каждый объём должен уравновешивать один
    // и тот же межузловой поток со своим V/dt. Повторяем с кешированной матрицей.
    {
        aero_test::TestField local_dt(device, 2);
        Partition partition;
        SteadyNodeSteps steps;
        steps.values = local_dt.data();
        partition.steady_steps = &steps;
        BdfStep local_solver(host, active, device, partition);
        NewtonSettings local_controls = controls;
        local_controls.roe = false; // Правая часть теста построена по потоку LF.
        local_controls.jacobian_update_every = 3;
        state.set(initial);
        for (double second_dt : {.3, 0., .15}) {
            // Меняем режим на той же матрице. При общем шаге в памяти остаются
            // старые узловые dt: проверка баланса обнаружит их ошибочное применение.
            steps.local = second_dt > 0.;
            if (steps.local) local_dt.set({.1, second_dt});
            const double step_argument = steps.local ? .5 : .08;
            const double dt_left = steps.local ? .5*.1 : step_argument;
            const double dt_right = steps.local ? .5*second_dt : step_argument;
            const auto old = state.read();
            refresh();
            const auto steady_step = local_solver.advance(state.data(), nullptr, volumes.data(), active_flux,
                                                     step_argument, local_controls, refresh);
            require(steady_step.jacobian_assemblies == (second_dt == .3 ? 1u : 0u), "unexpected spatial assembly on time step switch");
            const auto updated = state.read();
            const double mass_flux = std::sqrt(1.4/std::min(updated[0],updated[3])) * (updated[0]-updated[3]);
            require(std::abs(.4*(updated[0]-old[0])/dt_left + mass_flux) < 1.e-9 &&
                    std::abs(.6*(updated[3]-old[3])/dt_right - mass_flux) < 1.e-9,
                    "steady balance after switching local and global time steps");
        }
    }
    // Однородное состояние должно приниматься без вызова линейного решателя.
    state.set({1., 0., 2.5, 1., 0., 2.5});
    const auto uniform = advance();
    require(uniform.correction_iterations.empty(), "no linear solve for a uniform field");
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
        // Для интервала 1 включаем режим тем же переключателем, что в implicit.txt.
        // Интервал 3 проверяет совместное использование матрицы на нескольких шагах.
        controls.jacobian_update_every = interval == 1 ? 0 : interval;
        controls.set_simplified_newton(interval != 0);
        state.set(initial);
        bool rejected_trial = false;
        for (unsigned n = 0; n < 5; ++n) {
            dt = n % 2 ? .014 : .023;
            refresh();
            // Применяет и оценивает пробную поправку без окончательного принятия шага.
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
    // На чередующихся шагах проверяем решение с историей против обычного Ньютона.
    // Отклоняем первую пробу третьего шага, пока в кольце ещё лежат принятые слои.
    for (bool second_order : {false,true}) {
        std::vector<double> reference;
        for (unsigned order : {0u,6u}) {
            BdfStep checked(host,active,device);
            auto settings = controls;
            settings.second_order = second_order;
            settings.prediction_order = order;
            state.set(initial);
            for (unsigned n=0; n<7; ++n) {
                refresh();
                bool reject_first = order && n==2;
                const auto trial = [&] {
                    if (reject_first) { reject_first=false; return false; }
                    refresh();
                    return true;
                };
                const auto result_step = checked.advance(state.data(), nullptr, volumes.data(), active_flux,
                                                         n%2 ? .003 : .004, settings, refresh, trial);
                require(!reject_first && result_step.relative_residual<=settings.newton_relative,
                        "prediction must recover from rejection and satisfy BDF tolerance");
            }
            const auto field=state.read();
            if (!order) reference=field;
            else for (unsigned k=0; k<field.size(); ++k)
                require(std::abs(field[k]-reference[k])<2.e-11,"time predictor changed accepted BDF solution");
        }
    }
    // Для этой пары объёмов масса постоянна. Плотность x первого узла удовлетворяет
    // dx/dt = sqrt(gamma)*(m-x)/(V0*V1*sqrt(x)), m — средняя плотность.
    // Сравниваем с интегралом этого ОДУ, чтобы ошибка по пространству не скрыла порядок BDF2.
    const double mass = .4 * initial[0] + .6 * initial[3];
    // Вычисляет объёмный интеграл контрольной величины тестового поля.
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
