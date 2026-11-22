#include "settings.h"
#include "../../qa/test_checks.h"
#include "settings_boundary.h"
#include "fields.h"
#include "scheduled_output.h"
#ifdef AERO_TEST_OUTPUT_DEVICE
#include "../explicit/explicit.h"
#endif
#include "../initial/initial.h"
#include "../bc/periodic.h"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using namespace aero;
using aero_test::require;
template <class F> void rejects(F call, const std::string& fragment) {
    try {
        call();
    } catch (const std::exception& e) {
        require(std::string(e.what()).find(fragment) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("expected error containing " + fragment);
}

struct CaseFiles {
    std::string dir;
    CaseFiles() {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (unsigned n = 0;; ++n) {
            dir = "build/settings-test-" + std::to_string(stamp) + "-" + std::to_string(n);
#ifdef _WIN32
            if (_mkdir(dir.c_str()) == 0)
                break;
#else
            if (mkdir(dir.c_str(), 0700) == 0)
                break;
#endif
            require(errno == EEXIST, "cannot create test directory");
        }
    }
    ~CaseFiles() {
        for (const char* name :
             {"main.txt", "bc.txt", "init.txt", "cuda.txt", "gas.txt", "scheme.txt", "implicit.txt", "linear.txt", "turbulence.txt", "test.fields", "output.txt",
              "latest.fields", "p_000001.fields", "p_000002.fields", "p_000003.fields"})
            std::remove((dir + "/" + name).c_str());
#ifdef _WIN32
        _rmdir(dir.c_str());
#else
        rmdir(dir.c_str());
#endif
    }
    void write(const std::string& name, const std::string& content) const {
        std::ofstream out((dir + "/" + name).c_str(), std::ios::binary);
        out << content;
        out.close();
        require(bool(out), "cannot write " + name);
    }
    RunSettings load() const { return read_run_settings({dir, "./results"}); }
};

void arguments() {
    const char* args[] = {"aero", "--output=out dir", "--input=./params"};
    const auto dirs = parse_run_arguments(3, args);
    require(dirs.input == "./params" && dirs.output == "out dir", "CLI directories");
    rejects([&] { parse_run_arguments(2, args); }, "--input");
    args[2] = "--output=x";
    rejects([&] { parse_run_arguments(3, args); }, "duplicate");
    args[2] = "--input=";
    rejects([&] { parse_run_arguments(3, args); }, "directory");
    args[2] = "--wrong=1";
    rejects([&] { parse_run_arguments(3, args); }, "unknown argument");
    const char* edit[] = {"aero", "--interactive", "--input=in", "--output=out", "--interactive"};
    require(parse_run_arguments(4, edit).interactive, "interactive flag");
    rejects([&] { parse_run_arguments(5, edit); }, "duplicate --interactive");
    const char* automatic[] = {"aero", "--recover", "--input=in", "--output=out"};
    const auto selected = parse_run_arguments(4, automatic);
    require(selected.recover && selected.recovery_slot == -1, "automatic recovery");
    const char* restart[] = {"aero", "--input=in", "--output=out", "--recover", "0"};
    require(parse_run_arguments(5, restart).recovery_slot == 0, "recovery slot 0");
    restart[3] = "--recover=1";
    require(parse_run_arguments(4, restart).recovery_slot == 1, "recovery slot 1");
    restart[4] = "--recover";
    rejects([&] { parse_run_arguments(5, restart); }, "duplicate --recover");
    for (const char* bad : {"--recover=", "--recover=2", "--recover=-1"}) {
        restart[3] = bad;
        rejects([&] { parse_run_arguments(4, restart); }, "expects 0 or 1");
    }
    restart[3] = "--recover";
    restart[4] = "2";
    rejects([&] { parse_run_arguments(5, restart); }, "expects 0 or 1");
}

// Загружаем поставляемые с проектом шаблоны тем же путём, которым пользуется солвер.
void templates_case() {
    // Берём сетку с периодикой x и четырьмя метками, входящую в штатный комплект QA.
    const char* args[] = {"aero", "--input=templates", "--output=./results", "main:mesh=../qa/cases/momentum_2d/grid.mesh"};
    const auto run = read_run_settings(parse_run_arguments(4, args));
    require(run.output_zones.size() == 2 && run.output_zones[0].every_steps == 100 &&
                run.output_zones[1].every_time == 0.05 && run.output_zones[1].multi,
            "template output zones");
    require(!run.use_cuda && run.devices.empty() && run.reconstruction == "ebr3" &&
                run.euler_solver == "roe" && run.time_scheme == "ssprk3" && run.cfl == 0.5 &&
                run.final_time == 2 && run.omp_threads == 8 && run.report_every == 100,
            "template settings");
    const Mesh mesh = Mesh::read(run.mesh_path);
    const auto boundary = read_case_boundaries(run.boundary_path, mesh.data());
    require(boundary.periodic_pairs.size() == 1, "template seam count");
    require(boundary.patches.free_rules.size() == 1, "only the free patch owns flow rules");
    for (std::size_t i = 0; i < mesh.data().patch_names.size(); ++i) {
        const std::string& name = mesh.data().patch_names[i];
        const auto& patch = boundary.patches.by_patch.at(i);
        const auto* rules = patch.rules(boundary.patches.free_rules.data());
        if (name == "ymin")
            require(patch.type() == BoundaryType::Wall && !rules, "template wall");
        else if (name == "ymax")
            require(patch.type() == BoundaryType::Free && rules && rules->superin.use_initial &&
                        rules->inflow.source == BoundarySource::Current &&
                        rules->inflow.mode == BoundaryMode::Pressure &&
                        rules->outflow.source == BoundarySource::Current &&
                        rules->outflow.mode == BoundaryMode::NormalVelocity,
                    "template free");
        else
            require(patch.type() == BoundaryType::Periodic && !rules, "template periodic");
    }
    PeriodicDomain domain(mesh, boundary.periodic_pairs);
    const auto grid = domain.view();
    require(run.gamma == 1.4 && run.reconstruction_variables == "conservative",
            "gas and reconstruction variables");
    const auto initial = read_initial_file(mesh, run.gamma, run.initial_path, &grid);
    require(grid.node_count < mesh.view().node_count && initial.size() == grid.node_count * 4 &&
                initial[0] == 1 && std::abs(initial[1] - 0.3) < 1e-12,
            "template initial state");
}

void output_zones(const CaseFiles& files) {
    const std::string latest =
        "[latest]\nfile=latest.fields\nevery_steps=2\nvariables=E, ux, rho, uy\nformat=text\n";
    const std::string series =
        "[history]\nfile=p.fields\nevery_time=0.1\nmulti=1\nvariables=p\nformat=lzma\n";
    const auto path = files.dir + "/output.txt";
    require(read_output_zones(path).empty(), "absent output.txt");
    for (const auto& invalid : std::vector<std::string>{
             latest + "every_time=.1", latest + "every_steps=3", latest + "multi=2", latest + latest,
             latest + "oops=0", "file=f", "[s]\nvariables=p\nevery_steps=1",
             "[s]\nfile=x\nevery_steps=0\nvariables=p", "[s]\nfile=x\nevery_time=nan\nvariables=p",
             "[s]\nfile=x\nevery_steps=-1\nvariables=p", "[s]\nfile=x\nevery_steps=2\nvariables=p,p",
             "[s]\nfile=x\nevery_steps=2\nvariables=q", "[s]\nfile=../x\nevery_steps=2\nvariables=p",
             "[s]\nfile=x\nevery_steps=1\nvariables=p\nlayout=other",
             "[s]\nfile=x\nevery_steps=1\nvariables=p\nlayout=global\nformat=lzma",
             "[s]\nfile=recovery-1.bin\nevery_steps=1\nvariables=p\nlayout=global",
             series + "[alias]\nfile=p_000003.fields\nevery_steps=2\nvariables=E"}) {
        files.write("output.txt", invalid);
        rejects([&] { read_output_zones(path); }, "output.txt");
    }
    files.write("output.txt", latest + series);
    const auto zones = files.load().output_zones;
    require(output_filename(zones[1], 1000000) == "p_1000000.fields", "wide step number");
    const auto mesh = Mesh::read("qa/cases/momentum_2d/grid.mesh");
    const auto bounds = read_case_boundaries("templates/bc.txt", mesh.data());
    PeriodicDomain domain(mesh, bounds.periodic_pairs);
    const auto grid = domain.view();
    std::vector<MeshIndex> map(grid.node_dof, grid.node_dof + grid.original_node_count);
    ScheduledOutput writer(zones, files.dir, mesh, 1.4, map);
    std::vector<double> state;
    for (MeshIndex i = 0; i < grid.node_count; ++i) {
        const double rho = 1 + static_cast<double>(i);
        state.insert(state.end(), {rho, 2 * rho, -rho, 2.5 * rho + 5});
    }
    require(!writer.due(0, 0) && !writer.due(1, 0.09) && writer.limit_dt(0, 1) == 0.1, "initial deadline");
    rejects([&] { writer.due(1, 0.11); }, "missed");
    writer.write(1, 0.1, state);
    const auto first = FieldSnapshot::read(files.dir + "/p_000001.fields");
    first.check_mesh(mesh);
    require(first.step == 1 && first.time == 0.1 && first.fields.size() == 1 && first.fields[0].name == "p" &&
                std::abs(first.fields[0].values[0] - 2) < 1e-12,
            "first pressure snapshot");
    // Проверяем, что существующая история сохранилась: результат первой зоны уже опубликован.
    files.write("p_000002.fields", "existing");
    rejects([&] { writer.write(2, 0.2, state); }, "publish");
    require(std::remove((files.dir + "/p_000002.fields").c_str()) == 0, "release conflicting file");
    auto changed = state;
    changed[3] += 1;
    writer.write(2, 0.2, changed);
    const auto current = FieldSnapshot::read(files.dir + "/latest.fields");
    current.check_mesh(mesh);
    require(current.step == 2 && current.fields.size() == 4 && current.fields[0].name == "E" &&
                current.fields[1].name == "ux" && current.fields[2].name == "rho" &&
                current.fields[3].name == "uy",
            "simultaneous zones and variable order");
    for (std::size_t i = 0; i < map.size(); ++i)
        require(current.fields[2].values[i] == state[4 * map[i]] &&
                    current.fields[0].values[i] == state[4 * map[i] + 3] &&
                    std::abs(current.fields[1].values[i] - 2) < 1e-14 &&
                    std::abs(current.fields[3].values[i] + 1) < 1e-14,
                "periodic node field at " + std::to_string(i));
    writer.write(3, 0.3, changed);
    require(!writer.due(3, 0.3) && writer.due(4, 0.31), "accepted step schedule");
    writer.write(4, 0.31, changed);
    require(FieldSnapshot::read(files.dir + "/latest.fields").step == 4 &&
                FieldSnapshot::read(files.dir + "/p_000001.fields").step == 1,
            "overwrite and history");
    require(FieldSnapshot::read(files.dir + "/p_000003.fields").time == 0.3, "rounded deadline");
    auto clock = zones.front();
    clock.every_steps = 0;
    clock.every_time = 0.3;
    auto fast = zones.back();
    fast.multi = false;
    fast.file = "p_000001.fields";
    ScheduledOutput clocks({clock, fast}, files.dir, mesh, 1.4, map);
    require(clocks.limit_dt(0, 0.7) == 0.1, "nearest of two time clocks");
    clocks.write(1, 0.1, state);
    clocks.write(2, 0.2, state);
    const double accepted = 0.2 + clocks.limit_dt(0.2, 0.7);
    clocks.write(3, accepted, state);
    require(FieldSnapshot::read(files.dir + "/latest.fields").time == accepted &&
                FieldSnapshot::read(files.dir + "/p_000001.fields").time == accepted &&
                clocks.limit_dt(accepted, 0.01) == 0.01,
            "coincident physical deadlines");
    auto invalid_zone = zones[1];
    invalid_zone.variables = {"uz"};
    rejects([&] { ScheduledOutput bad({invalid_zone}, files.dir, mesh, 1.4); }, "dimension");
    for (unsigned d : {1u, 3u}) {
        const auto other = Mesh::read(d == 1 ? "qa/cases/pulse_1d/grid.mesh" : "qa/cases/wave_3d/grid.mesh");
        auto zone = zones.front();
        zone.variables = {d == 1 ? "ux" : "uz", "p"};
        zone.format = FieldFormat::Binary;
        ScheduledOutput plain({zone}, files.dir, other, 1.4);
        std::vector<double> q;
        for (MeshIndex i = 0; i < other.view().node_count; ++i) {
            q.push_back(2);
            for (unsigned j = 1; j <= d; ++j)
                q.push_back(2.0 * j);
            q.push_back(d == 1 ? 6 : 19);
        }
        plain.write(2, 0.09, q);
        const auto saved = FieldSnapshot::read(files.dir + "/latest.fields");
        saved.check_mesh(other);
        require(saved.fields[0].values.back() == d && std::abs(saved.fields[1].values.back() - 2) < 1e-12,
                "plain mesh output");
#ifdef AERO_TEST_OUTPUT_DEVICE
        const int devices = ExplicitIntegrator::cuda_device_count();
        require(devices > 0, "CUDA output test requires a device");
        std::vector<double> volumes(q.size() / (d + 2), 1.0);
        for (int device = 0; device < devices; ++device) {
            ExplicitIntegrator solver(volumes.size(), d + 2, volumes.data(), q.data(), ExplicitBackend::Cuda,
                                      device);
            ScheduledOutput gpu_writer({zone}, files.dir, other, 1.4);
            if (gpu_writer.due(2, 0.09))
                gpu_writer.write(2, 0.09, solver.state());
            const auto gpu = FieldSnapshot::read(files.dir + "/latest.fields");
            gpu.check_mesh(other);
            require(gpu.fields[0].values == saved.fields[0].values &&
                        gpu.fields[1].values == saved.fields[1].values,
                    "output using the CUDA integrator snapshot");
        }
        std::cout << "Verified " << d << "D disk output on " << devices << " CUDA devices.\n";
#endif
    }
}

// Оба закона используют одни опорные величины. Для Сазерленда проверяем и
// дополнительную константу, включая ноль — предел, совпадающий с законом normal.
#ifdef AERO_HAS_VISCOSITY
void viscosity_settings(const CaseFiles& files) {
    files.write("main.txt", "mesh=grid.mesh\nreconstruction=first_order\nreconstruction_variables=conservative\n"
                            "time_scheme=euler\ncfl=.5\neuler_solver=roe\nfinal_time=0\nbackend=cpu");
    const std::string gas = "gamma=1.4\n";
    const std::string reference = gas + "mach=0.5\nreynolds=1e6\nprandtl=0.72\nt_amb=273.15\n";
    files.write("gas.txt", gas);
    require(files.load().viscosity.law == "off", "default inviscid case");

    RunDirectories dirs(files.dir, "./results");
    dirs.overrides["main"]["viscosity"] = "NORMAL";
    for (const char* key : {"mach", "reynolds", "prandtl", "t_amb"}) {
        auto incomplete = reference;
        const auto start = incomplete.find(key);
        incomplete.erase(start, incomplete.find('\n', start) + 1 - start);
        files.write("gas.txt", incomplete);
        rejects([&] { read_run_settings(dirs); }, std::string("missing ") + key);
    }
    files.write("gas.txt", reference);
    const auto normal = read_run_settings(dirs).viscosity;
    require(normal.law == "normal" && normal.reynolds == 1e6 && normal.prandtl == 0.72 &&
            normal.t_amb == 273.15, "normal viscosity parameters");

    dirs.overrides["main"]["viscosity"] = "sutherland";
    rejects([&] { read_run_settings(dirs); }, "missing sutherland_s");
    files.write("gas.txt", reference + "sutherland_s=110.4");
    require(read_run_settings(dirs).viscosity.sutherland_s == 110.4, "Sutherland temperature");
    dirs.overrides["gas"]["sutherland_s"] = "0";
    require(read_run_settings(dirs).viscosity.sutherland_s == 0., "zero Sutherland temperature");
    for (const auto& invalid : RunDirectories::Parameters{{"reynolds","-1"}, {"prandtl","0"},
                                                        {"t_amb","nan"}, {"sutherland_s","-1"}}) {
        dirs.overrides["gas"] = {invalid};
        rejects([&] { read_run_settings(dirs); }, invalid.first);
    }
    dirs.overrides.clear();
    dirs.overrides["main"]["viscosity"] = "normal";
    require(read_run_settings(dirs).input_warnings.empty(), "shared gas file for both laws");
    dirs.overrides["main"]["viscosity"] = "unknown";
    rejects([&] { read_run_settings(dirs); }, "viscosity");
}

#endif
#ifdef AERO_HAS_TURBULENCE
void turbulence_settings(const CaseFiles& files) {
    files.write("main.txt","mesh=grid.mesh\nreconstruction=first_order\nreconstruction_variables=conservative\n"
                           "backend=cpu\ntime_scheme=euler\ncfl=.5\neuler_solver=lf\nfinal_time=0\nviscosity=normal");
    files.write("gas.txt","gamma=1.4\nmach=.5\nreynolds=1000\nprandtl=.72\nt_amb=300");
    require(!files.load().turbulence.enabled,"turbulence defaults to Off");
    RunDirectories dirs(files.dir, "./results");
    dirs.overrides["main"]["Turbulence"]="On";
    rejects([&]{read_run_settings(dirs);},"turbulence.txt");
    files.write("turbulence.txt","model=RANS\nrans_eq=SA");
    rejects([&]{read_run_settings(dirs);},"nut_nu_amb");
    files.write("turbulence.txt","model=RANS\nrans_eq=SA\nnut_nu_amb=.21\n");
    const auto sa=read_run_settings(dirs).turbulence;
    require(sa.enabled && !sa.high_order && sa.prandtl==.9 && sa.nut_nu_amb==.21,"SA parameters");
    for(const auto& wrong:RunDirectories::Parameters{{"model","invalid"},{"rans_eq","SST"},{"nut_nu_amb","nan"},{"pr_t","0"},{"sa_high_order","yes"}}) {
        dirs.overrides["turbulence"]={wrong};
        rejects([&]{read_run_settings(dirs);},wrong.first);
    }
    dirs.overrides.erase("turbulence");
    files.write("turbulence.txt","model=LES\nles_eq=Sigma\nsubgridscale=SLA\n");
    const auto les=read_run_settings(dirs).turbulence;
    require(!les.uses_sa() && les.les.scale==turbulence::Scale::Sla && les.les.f_kh_lim==.1 &&
            les.les.switch_on_vtm && les.les.c_sigma==1.35,"LES defaults");
    for (const auto& wrong : RunDirectories::Parameters{{"F_KH_lim","1.1"},{"c_sigma","0"},
             {"subgridscale","invalid"},{"switch_on_vtm","1"},{"les_eq","invalid"}}) {
        dirs.overrides["turbulence"]={wrong};
        rejects([&]{read_run_settings(dirs);},wrong.first);
    }
    dirs.overrides.erase("turbulence");
    files.write("turbulence.txt","model=DDES\nrans_eq=SA\nnut_nu_amb=.21\nsubgridscale=vol\n");
    const auto ddes=read_run_settings(dirs).turbulence;
    require(ddes.uses_sa() && ddes.ddes && !ddes.uses_sigma() && ddes.c_des==.65,"DDES defaults");
    dirs.overrides["turbulence"]={{"sa_high_order","on"}};
    require(read_run_settings(dirs).turbulence.high_order,"DDES high-order convection");
    dirs.overrides["turbulence"]={{"c_des","0"}};
    rejects([&]{read_run_settings(dirs);},"c_des");
    dirs.overrides["main"]["Turbulence"]="Off";
    require(!read_run_settings(dirs).turbulence.enabled,"Off ignores turbulence.txt");
}
#endif
void main_and_cuda(const CaseFiles& files) {
    const std::string head = "\xef\xbb\xbf# test\r\nmesh = \"path #1/grid.mesh\" # relative\n";
    files.write("gas.txt", "gamma=1.67");
    const std::string common =
        "time_scheme=euler\ncfl=2e-1\neuler_solver=lf\nfinal_time=0\nreconstruction_variables=flux\n";
    const std::string base = head + common + "reconstruction=weno5\n";
    files.write("main.txt", base + "backend=cpu\n");
    require(!files.load().background.active(), "old cases default to Euler");
    files.write("main.txt", base + "backend=cpu\nEquations=Linear\n");
    rejects([&] { files.load(); }, "linear.txt");
    files.write("linear.txt", "rho0=2\np0=3\nux0=-.7\nuy0=0\n");
    const auto linear = files.load();
    require(linear.background.active() && linear.background_velocities == 2 &&
            linear.background.velocity[0] == -.7 && linear.background.sound == std::sqrt(1.67*1.5),
            "signed mean velocity and sound speed");
    for (const char* bad : {"rho0=0\np0=3\nux0=0", "rho0=1\np0=-.1\nux0=0",
                            "rho0=1\np0=3\nux0=nan", "rho0=1\nux0=0"}) {
        files.write("linear.txt", bad);
        rejects([&] { files.load(); }, "linear.txt");
    }
    const std::string implicit_main = head + common.substr(common.find('\n') + 1)
                                      + "time_scheme=BDF1\nbackend=cpu\nreconstruction=first_order\n";
    files.write("main.txt", implicit_main);
    rejects([&] { files.load(); }, "implicit.txt");
    std::ifstream controls("templates/implicit.txt");
    std::ostringstream content;
    content << controls.rdbuf();
    files.write("implicit.txt", content.str());
    const char* args[] = {"aero", "implicit.txt:newton_tol=2e-7", "implicit:newton_max_iter=9"};
    auto directories = parse_run_arguments(3, args);
    directories.input = files.dir;
    const auto bdf = read_run_settings(directories);
    require(bdf.implicit_settings.linear_max_iterations == 200, "default BiCGSTAB limit");
    files.write("implicit.txt", content.str() + "\nsolver_max_iter=17\n");
    require(read_run_settings(directories).implicit_settings.linear_max_iterations == 17, "custom BiCGSTAB limit");
    require(!bdf.implicit_settings.simplified_jacobian, "full LF derivative is the default");
    require(!bdf.implicit_settings.compact_jacobian, "full viscous stencil is the default");
    directories.overrides["implicit"]["compact_jacobian"] = "on";
    require(read_run_settings(directories).implicit_settings.compact_jacobian, "compact viscous stencil");
    directories.overrides["implicit"]["simplified_jacobian"] = "on";
    require(read_run_settings(directories).implicit_settings.simplified_jacobian, "simplified LF derivative");
    require(bdf.uses_implicit_time() && bdf.implicit_settings.newton_relative == 2e-7 &&
            bdf.implicit_settings.max_iterations == 9 && bdf.implicit_settings.linear_relative == 1e-8,
            "BDF1 settings and overrides");
    directories.overrides["main"]["time_scheme"]="BDF2";
    directories.overrides["implicit"]["jacobian_flux"]="Roe";
    directories.overrides["implicit"]["preconditioner"]="CSGS";
    const auto second=read_run_settings(directories).implicit_settings;
    require(second.second_order && second.roe && second.csgs && second.simplified_jacobian,"BDF2 controls");
    directories.overrides["implicit"]["local_time_step"] = "on";
    rejects([&] { read_run_settings(directories); }, "local_time_step");
    directories.overrides["implicit"]["local_time_step"] = "auto";
    rejects([&] { read_run_settings(directories); }, "local_time_step");
    directories.overrides["implicit"]["local_time_step"] = "on";
    directories.overrides["main"]["time_scheme"] = "steady";
    require(read_run_settings(directories).local_time_step == LocalStepMode::On, "local time stepping in steady");
    require(read_run_settings(directories).local_time_step_max_ratio == 4., "default local step ratio");
    directories.overrides["implicit"]["local_time_step"] = "auto";
    require(read_run_settings(directories).local_time_step == LocalStepMode::Auto, "adaptive steady step");
    auto& ratio = directories.overrides["implicit"]["local_time_step_max_ratio"];
    ratio = "1";
    require(read_run_settings(directories).local_time_step_max_ratio == 1., "uniform local steps");
    ratio = "2.5";
    require(read_run_settings(directories).local_time_step_max_ratio == 2.5, "fractional local step ratio");
    ratio = "off";
    require(read_run_settings(directories).local_time_step_max_ratio == 0., "unrestricted local steps");
    directories.overrides["implicit"].erase("local_time_step_max_ratio");
    directories.overrides["implicit"].erase("local_time_step");
    directories.overrides["implicit"]["jacobian_update_every"] = "3";
    auto steady = read_run_settings(directories);
    require(steady.implicit_settings.simplified_newton(), "legacy positive interval enables simplified Newton");
    auto& newton_mode = directories.overrides["implicit"]["simplified_newton"];
    newton_mode = "off";
    require(read_run_settings(directories).implicit_settings.jacobian_update_every == 0, "explicit full Newton");
    newton_mode = "on";
    require(read_run_settings(directories).implicit_settings.jacobian_update_every == 3, "keep multistep Jacobian reuse");
    directories.overrides["implicit"]["jacobian_update_every"] = "0";
    require(read_run_settings(directories).implicit_settings.jacobian_update_every == 1, "one assembly in simplified Newton");
    directories.overrides["implicit"].erase("simplified_newton");
    require(steady.steady() && steady.uses_implicit_time() && !steady.implicit_settings.second_order &&
            steady.local_time_step == LocalStepMode::Off && !steady.steady_absolute && steady.steady_tol == 1.e-6 && steady.steady_max_steps == 10000 &&
            steady.implicit_settings.jacobian_update_every == 3, "steady default controls");
    directories.overrides["implicit"]["steady_criterion"] = "absolute";
    directories.overrides["implicit"]["steady_tol"] = "10";
    require(read_run_settings(directories).steady_tol == 10., "absolute steady tolerance has physical units");
    directories.overrides["implicit"].erase("steady_criterion");
    directories.overrides["implicit"].erase("steady_tol");
    for (const auto& bad : std::vector<std::pair<std::string, std::string>>{
         {"local_time_step","adaptive"}, {"local_time_step_max_ratio","0.9"}, {"local_time_step_max_ratio","nan"}, {"steady_criterion","unknown"}, {"steady_tol","1"}, {"steady_max_steps","0"},
         {"jacobian_update_every","-2"}, {"simplified_newton","maybe"}, {"simplified_jacobian","maybe"}, {"compact_jacobian","maybe"}, {"solver","gmres"}, {"jacobian_flux","unknown"}, {"preconditioner","unknown"},
         {"solver_max_iter","0"}, {"solver_max_iter","2.5"}, {"solver_max_iter","-1"}, {"solver_tol","0"}, {"newton_tol","1"}, {"newton_max_iter","0"}, {"newton_max_iter","2.5"}}) {
        directories.overrides["implicit"][bad.first] = bad.second;
        rejects([&] { read_run_settings(directories); }, bad.first);
        directories.overrides["implicit"].erase(bad.first);
    }
    files.write("implicit.txt", "invalid file ignored for explicit schemes");
    for (const char* scheme : {"first_order", "ebr3", "ebr5", "weno3", "weno5", "mp5", "MPWENO5"}) {
        files.write("main.txt", head + common + "backend=cpu\nreconstruction=" + scheme);
        const auto run = files.load();
        require(run.reconstruction == scheme && run.euler_solver == "lf" && run.time_scheme == "euler" &&
                    run.cfl == 0.2 && run.final_time == 0 && run.devices.empty() &&
                    run.mesh_path == files.dir + "/path #1/grid.mesh",
                "CPU main.txt");
    }
    require(files.load().mpweno_ad_corr && files.load().mp6_hybrid == 0, "optional scheme defaults");
    files.write("scheme.txt", "mpweno_ad_corr=off\nmp6_hybrid=0.4");
    require(!files.load().mpweno_ad_corr && files.load().mp6_hybrid == .4, "MP options");
    for (const char* bad : {"mpweno_ad_corr=2", "mp6_hybrid=-0.1", "mp6_hybrid=1.1"}) {
        files.write("scheme.txt",bad);
        rejects([&] { files.load(); }, "scheme.txt");
    }
    files.write("scheme.txt", "mpweno_ad_corr=on\nmp6_hybrid=1");
    files.write("main.txt", base + "backend=cuda");
    rejects([&] { files.load(); }, "cuda.txt");
    files.write("cuda.txt", "devices = 2, 0, 1");
    require(files.load().devices == std::vector<int>({2, 0, 1}), "device order");
    for (const char* list : {"1,1", "-1", "1,", "0,,2", "1.0", "99999999999999", "0 1"}) {
        files.write("cuda.txt", std::string("devices = ") + list);
        rejects([&] { files.load(); }, "cuda.txt:1");
    }
    files.write("main.txt", base + "backend=cpu");
    require(files.load().devices.empty(), "CPU ignores invalid cuda.txt");
    files.write("main.txt", base + "backend=cpu\nother=x");
    require(files.load().input_warnings.size() == 1, "unknown parameter warning");
    for (const char* extra : { "backend=cpu\ncfl=0.5", "backend=gpu"}) {
        files.write("main.txt", base + extra);
        rejects([&] { files.load(); }, "main.txt");
    }
    files.write("main.txt", base);
    rejects([&] { files.load(); }, "missing backend");
    const std::string valid = base + "backend=cpu";
    files.write("main.txt",
                valid + "\nomp_threads=3\nreport_every=17\ntrace_ratio_min=.5\ntrace_ratio_max=1.5");
    const auto configured = files.load();
    require(configured.omp_threads == 3 && configured.report_every == 17, "explicit report settings");
    require(configured.trace_ratio_min == .5 && configured.trace_ratio_max == 1.5, "trace ratio bounds");
#ifdef _OPENMP
    require(omp_get_max_threads() == 3, "configured OpenMP runtime");
#endif
    for (const char* entry :
         {"omp_threads=-1", "omp_threads=1.5", "omp_threads=2147483648", "report_every=2x",
          "report_every=-1", "report_every=18446744073709551616", "trace_ratio_min=-.1",
          "trace_ratio_min=1", "trace_ratio_max=1", "trace_ratio_max=nan"}) {
        files.write("main.txt", valid + "\n" + entry);
        rejects([&] { files.load(); }, std::string(entry).substr(0, std::string(entry).find('=')));
    }
    files.write("main.txt", valid);
    const auto defaults = files.load();
    require(defaults.trace_ratio_min == 0 && defaults.trace_ratio_max == 1e16, "default trace ratio bounds");
    require(defaults.gamma == 1.67 && !defaults.omp_threads && !defaults.report_every, "case defaults");
    for (const auto& change :
         std::vector<std::pair<std::string, std::string>>{{"2e-1", "0"},
                                                          {"2e-1", "nan"},
                                                          {"final_time=0", "final_time=-1"},
                                                          {"weno5", "eno5"},
                                                          {"euler\n", "rk4\n"},
                                                          {"=lf", "=hll"},
                                                          {"=flux", "=other"}}) {
        std::string invalid = valid;
        invalid.replace(invalid.find(change.first), change.first.size(), change.second);
        files.write("main.txt", invalid);
        rejects([&] { files.load(); }, "main.txt");
    }
    files.write("main.txt", valid);
    for (const char* value : {"1", "0.9", "nan", "1e500", "2x"}) {
        files.write("gas.txt", std::string("gamma=") + value);
        rejects([&] { files.load(); }, "gas.txt");
    }
    files.write("gas.txt", "gamma=1.4");
}

void boundaries(const CaseFiles& files) {
    MeshData data;
    data.dimension = 2;
    data.patch_names = {"top #2", "left", "right", "bottom #1"};
    const std::string physical =
        "[right]\nbc_type wall\n[left]\nbc_type free\nsuperin initial p=2\ninflow base_p\noutflow base_u\n";
    const std::string seam = "[\"bottom #1\"]\nbc_type periodic\npair \"top #2\"\ntranslation 0 1 0\n";
    const std::string path = files.dir + "/bc.txt";
    files.write("bc.txt", physical + seam);
    const auto bc = read_case_boundaries(path, data);
    const auto* left_rules = bc.patches.by_patch[1].rules(bc.patches.free_rules.data());
    require(left_rules && left_rules->superin.use_initial && left_rules->superin.primitive[1] == 2 &&
                bc.patches.by_patch[2].type() == BoundaryType::Wall &&
                bc.periodic_pairs[0].patch == "bottom #1" && bc.periodic_pairs[0].pair == "top #2" &&
                bc.periodic_pairs[0].translation[1] == 1,
            "patch order and labels");
    for (const std::string& invalid :
         std::vector<std::string>{physical, physical + seam + "[\"top #2\"]\nbc_type wall",
                                  physical + seam + "[wing]\nbc_type wall", physical + physical,
                                  "bc_type wall", "[left]\nbc_type free\ninflow base_p",
                                  "[right]\nbc_type periodic\npair right\ntranslation 0 1 0",
                                  "[right]\nbc_type periodic\npair nowhere\ntranslation 0 1 0"}) {
        files.write("bc.txt", invalid);
        rejects([&] { read_case_boundaries(path, data); }, "bc.txt");
    }
}

// Проверяем начальные поля из снимка вместе с привязкой к сетке.
void initial_snapshot(const CaseFiles& files) {
    files.write("main.txt", "mesh=../../qa/cases/pulse_1d/grid.mesh\nreconstruction=first_order\n"
                            "time_scheme=euler\ncfl=1\nbackend=cpu\neuler_solver=lf\nfinal_time="
                            "1\nreconstruction_variables=characteristic");
    files.write("init.txt", "initial_type file\npath test.fields\nfield U");
    const auto run = files.load();
    const Mesh mesh = Mesh::read(run.mesh_path);
    FieldSnapshot snap(mesh);
    Field u;
    u.name = "U";
    u.location = FieldLocation::Node;
    u.components = {"rho", "rho_ux", "E"};
    const auto n = static_cast<std::size_t>(mesh.view().node_count);
    u.values.assign(n, 1);
    u.values.insert(u.values.end(), n, 0.5);
    u.values.insert(u.values.end(), n, 3);
    snap.fields.push_back(u);
    snap.write(files.dir + "/test.fields", FieldFormat::Lzma);
    const auto state = read_initial_file(mesh, run.gamma, run.initial_path);
    require(state.size() == 3 * n && state[0] == 1 && state[1] == 0.5 && state[2] == 3,
            "snapshot relative to init.txt");
}
}

int main() {
    try {
#ifdef _OPENMP
        const int original_threads = omp_get_max_threads();
#endif
        arguments();
        templates_case();
        CaseFiles files;
#ifdef AERO_HAS_VISCOSITY
        viscosity_settings(files);
#endif
#ifdef AERO_HAS_TURBULENCE
        turbulence_settings(files);
#endif
        main_and_cuda(files);
#ifdef _OPENMP
        require(omp_get_max_threads() == original_threads, "restore OpenMP runtime default");
#endif
        boundaries(files);
        initial_snapshot(files);
        output_zones(files);
        std::cout
            << "Settings tests passed (CLI, templates, boundaries, initial snapshot and scheduled output).\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
