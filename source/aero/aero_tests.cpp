#include "partition_kernels.h"
#include "../../qa/test_checks.h"
#include "../mesh/mesh.h"
#include "../mesh/leaf_balance.h"
#include "../io/fields.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using aero_test::require;
void check_partition_kernels() {
    using namespace aero;
    const residual_detail::Index offsets[] = {0, 1, 2};
    using residual_detail::OrientedFace;
    const OrientedFace incidence[] = {OrientedFace::outward(0), OrientedFace::inward(0)};
    residual_detail::Geometry grid{};
    grid.dim = 1;
    grid.offsets = offsets;
    grid.incidence = incidence;
    const MeshIndex owned[] = {1};
    const double volumes[] = {1, 2};
    EulerFluxResult flux{};
    flux.flux[0] = 2;
    flux.flux[1] = 4;
    flux.flux[2] = 6;
    flux.max_wave_speed = 9;
    double state[] = {10, 11, 12, 20, 21, 22}, original[3] = {};
    require(
        partition_detail::update_node(grid, &flux, volumes, owned, state, original, 0, {0, 0, 1}, .5, true),
        "owned update");
    for (unsigned v = 0; v < 3; ++v) {
        require(state[v] == 10 + v && original[v] == 20 + v, "halo untouched and compact original saved");
        require(state[3 + v] == 20 + v + .5 * (v + 1), "signed flux divided by owned volume");
    }
    require(partition_detail::update_node(grid, &flux, volumes, owned, state, original, 0, {1, .75, .25}, .5,
                                          false),
            "SSPRK blend");
    for (unsigned v = 0; v < 3; ++v)
        require(state[3 + v] == 20 + v + .25 * (v + 1), "compact start-of-step indexing");
    require(
        partition_detail::update_node(grid, &flux, volumes, nullptr, state, nullptr, 0, {0, 0, 1}, .5, false),
        "dense Euler update");
    for (unsigned v = 0; v < 3; ++v)
        require(state[v] == 10 + v - (v + 1), "dense node uses outgoing flux");

    double record[4] = {0, 0, 0, -1};
    partition_detail::RecordField{3, state, nullptr}.copy_record(1, 0, nullptr, record);
    partition_detail::RecordField{3, state, nullptr}.copy_record(0, 0, record, nullptr);
    require(std::equal(state, state + 3, state + 3) && record[3] == -1, "state halo record width");
    EulerFluxResult received{};
    partition_detail::RecordField{3, nullptr, &flux}.copy_record(0, 0, nullptr, record);
    partition_detail::RecordField{3, nullptr, &received}.copy_record(0, 0, record, nullptr);
    require(std::equal(flux.flux, flux.flux + 3, received.flux) && received.max_wave_speed == 9,
            "flux halo includes speed");
}
void check_leaf_balance() {
    using namespace aero;
    const auto uniform = [](MeshIndex i) -> LeafCost { return {10 * i, 100 * i}; };
    const std::vector<CpuNodeShare> groups = {{{0, 2}, 500}, {{1, 3}, 1100}};
    const auto plan = balance_leaves(16, uniform, groups, {1, 1, 3, 1});
    require(plan.ranks == std::vector<int>({0, 2, 1, 3}) &&
                plan.cuts == std::vector<MeshIndex>({0, 1, 5, 10, 16}),
            "shared node caps and interleaved ranks");
    const auto tiny = balance_leaves(1, uniform, {{{0, 1, 2}, 100}}, {1, 4, 2});
    require(tiny.cuts == std::vector<MeshIndex>({0, 0, 1, 1}), "fast rank owns the only leaf");
    const auto empty = balance_leaves(0, uniform, groups, {1, 1, 1, 1});
    require(empty.cuts == std::vector<MeshIndex>(5, 0), "empty archive layout");

    // Процессы одного вычислительного узла делят оперативную память. Ёмкость каждой видеокарты учитывается отдельно.
    const std::vector<GpuRankCapacity> cards = {{10, {}}, {20, {}}, {30, {}}, {100, {}}};
    const auto devices =
        balance_gpu_leaves(16, uniform, {{{0, 2}, 500}, {{1, 3}, 1200}}, {1, 1, 1, 1}, cards);
    require(devices.ranks == std::vector<int>({0, 2, 1, 3}) &&
                devices.cuts == std::vector<MeshIndex>({0, 1, 4, 6, 16}),
            "GPU caps and shared RAM");
    const auto fallback = balance_gpu_leaves(1, uniform, {{{0, 1}, 100}}, {4, 1}, {{0, {{0, 1}}}, {0, {}}});
    require(fallback.cuts == std::vector<MeshIndex>({0, 0, 1}),
            "try the remaining GPU despite a pessimistic estimate");

    // На неоднородной сетке перебираем допустимые пары разрезов и сверяем лучший результат с планировщиком.
    const LeafCost costs[] = {{0, 0}, {8, 20}, {10, 32}, {20, 40}, {42, 67}, {57, 72}, {61, 90}};
    const auto prefix = [&](MeshIndex i) { return costs[i]; };
    const std::vector<long double> speeds = {1, 3, 2};
    for (int cap = 10; cap <= 80; cap += 5) {
        const std::vector<CpuNodeShare> nodes = {{{0}, 30}, {{1}, static_cast<long double>(cap)}, {{2}, 35}};
        long double best = 1e9L;
        for (unsigned a = 0; a <= 6; ++a)
            for (unsigned b = a; b <= 6; ++b) {
                if (costs[a].bytes > 30 || costs[b].bytes - costs[a].bytes > static_cast<unsigned>(cap) ||
                    costs[6].bytes - costs[b].bytes > 35)
                    continue;
                best = std::min(
                    best, std::max({costs[a].work / speeds[0], (costs[b].work - costs[a].work) / speeds[1],
                                    (costs[6].work - costs[b].work) / speeds[2]}));
            }
        try {
            const auto cuts = balance_leaves(6, prefix, nodes, speeds).cuts;
            require(best < 1e9L && cuts.front() == 0 && cuts.back() == 6, "infeasible cut accepted");
            for (unsigned i = 0; i < 3; ++i) {
                const auto a = prefix(cuts[i]), b = prefix(cuts[i + 1]);
                require(b.bytes - a.bytes <= nodes[i].mesh_bytes &&
                            (b.work - a.work) / speeds[i] <= best + 1e-8L,
                        "partition lost feasible memory or CPU balance");
            }
        } catch (const MeshError&) {
            require(best == 1e9L, "feasible cuts rejected");
        }
    }
}
std::string read(const std::string& path) {
    std::ifstream in(path.c_str());
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}
void write(const std::string& path, const std::string& text) {
    std::ofstream out(path.c_str());
    out << text;
    out.close();
    require(bool(out), "Cannot write " + path);
}
void remove_dir(const std::string& path) {
#ifdef _WIN32
    _rmdir(path.c_str());
#else
    rmdir(path.c_str());
#endif
}
std::string run_case(const std::string& program, const std::string& device, bool interactive,
                     bool expect_fallback, bool implicit = false) {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::string dir = "build/aero-test-" + std::to_string(stamp), out = dir + "/out",
                      log = dir + "/run.log";
#ifdef _WIN32
    require(_mkdir(dir.c_str()) == 0, "Cannot create " + dir);
    const std::string command = "call \"" + program + "\"";
#else
    require(mkdir(dir.c_str(), 0700) == 0, "Cannot create " + dir);
    const std::string command = "\"" + program + "\"";
#endif
    const std::string backend = device.empty() ? "cpu" : "cuda";
    write(dir + "/main.txt",
          "mesh=../../examples/tecplot/demo_1d.mesh\nreconstruction=first_order\n"
          "cfl=.35\nomp_threads=2\nreport_every=1\nfinal_time=.008\nbackend=" +
              backend +
              (interactive
                   ? "\ntime_scheme=ssprk3\neuler_solver=roe\nreconstruction_variables=characteristic\n"
                   : "\ntime_scheme=euler\neuler_solver=lf\nreconstruction_variables=flux\n"));
    write(dir + "/gas.txt", "gamma=1.4\n");
    write(dir + "/init.txt", "initial_type formula\nrho=1+0.1*x\nux=0\np=1+0.1*x\n");
    if (!device.empty())
        write(dir + "/cuda.txt", "devices=" + device + "\n");
    write(dir + "/output.txt",
          "[latest]\nfile=state.fields\nevery_steps=1\nvariables=rho,ux,p\n"
          "[clock]\nfile=history/p.fields\nevery_time=.004\nmulti=1\nvariables=p\nformat=lzma\n");
    if (implicit) write(dir + "/implicit.txt", read("templates/implicit.txt"));
    const std::string launch = command + " --input=" + dir + " --output=" + out +
        (implicit ? " main:time_scheme=BDF1" : "");
    const auto mesh = aero::Mesh::read("examples/tecplot/demo_1d.mesh");
    std::string rules;
    for (const auto& name : mesh.data().patch_names)
        rules += "[" + name + "]\nbc_type wall\n";
    const std::string redirect = " > " + log + " 2>&1";
    if (interactive) {
#ifdef _WIN32
        std::FILE* pipe = _popen((launch + " --interactive" + redirect).c_str(), "w");
#else
        std::FILE* pipe = popen((launch + " --interactive" + redirect).c_str(), "w");
#endif
        require(pipe != nullptr, "Cannot launch interactive aero");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (read(log).find("press Enter") == std::string::npos &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto generated = read(dir + "/bc.txt");
        write(dir + "/bc.txt", rules);
        std::fputs("\n", pipe);
#ifdef _WIN32
        const int rc = _pclose(pipe);
#else
        const int rc = pclose(pipe);
#endif
        require(rc == 0, read(log));
        for (const auto& name : mesh.data().patch_names)
            require(generated.find("[" + name + "]") != std::string::npos, "missing generated section");
    } else {
        require(std::system((launch + redirect).c_str()) != 0, "missing bc must pause the batch calculation");
        require(read(log).find("Created ") != std::string::npos &&
                    read(log).find("Starting time") == std::string::npos,
                "missing BC workflow");
        write(dir + "/bc.txt", rules);
        require(std::system((launch + redirect).c_str()) == 0, "aero failed; see " + log);
    }
    const auto trace = read(log);
    if (expect_fallback)
        require(trace.find("CUDA unavailable; running on CPU.") != std::string::npos, "missing CPU fallback");
    std::string expected = device.empty() || expect_fallback ? "CPU" : "CUDA device " + device;
    for (const auto& key : std::vector<std::string>{"Active backend = " + expected, "Boundary label",
                                                    "Preparing flux surfaces... done",
                                                    "STEP 1  |", "Final time = 0.008"})
        require(trace.find(key) != std::string::npos, "missing " + key + " in " + log);
    require(trace.find("Building reconstruction stencils") == std::string::npos,
            "first-order run reported a reconstruction stencil build");
    require(read(dir + "/bc.txt") == rules, "existing boundary settings must be preserved");
    for (const auto& name : mesh.data().patch_names)
        require(trace.find(name + "\t1\n") != std::string::npos, "boundary face count for " + name);
    const auto final = aero::FieldSnapshot::read(out + "/state.fields");
    final.check_mesh(mesh);
    require(final.time == 0.008 && final.step >= 2 && final.fields.size() == 3, "final step and fields");
    // Отчёт содержит достигнутую невязку каждого принятого неявного шага.
    const std::string marker =  "Newton         ";
    if (implicit) {
        for (std::uint64_t step = 1; step <= final.step; ++step) {
            const auto header = trace.find("STEP " + std::to_string(step) + "  |");
            const std::string label = "relative residual = ";
            const auto pos = trace.find(label, header);
            require(pos != std::string::npos && pos < trace.find("\nField", header), "missing Newton residual for an accepted step");
            const double residual = std::stod(trace.substr(pos + label.size()));
            require(residual >= 0. && residual <= 1.e-6, "reported Newton residual exceeds tolerance");
        }
    } else {
        require(trace.find(marker) == std::string::npos, "Newton report on an explicit step");
    }
    bool evolved = false;
    for (double u : final.fields[1].values)
        if (std::abs(u) > 1e-8)
            evolved = true;
    require(evolved, "the computed velocity must evolve");
    unsigned records = 0;
    for (std::uint64_t step = 1; step <= final.step; ++step) {
        const auto digits = std::to_string(step);
        const auto path = out + "/history/p_" + std::string(6 - digits.size(), '0') + digits + ".fields";
        if (!std::ifstream(path.c_str()))
            continue;
        const auto snap = aero::FieldSnapshot::read(path);
        snap.check_mesh(mesh);
        ++records;
        require(snap.step == step && snap.time == records * 0.004 && snap.fields.size() == 1 &&
                    snap.fields[0].name == "p",
                "clock history");
        std::remove(path.c_str());
    }
    require(records == 2, "two scheduled times");
    require(bool(std::ifstream(out + "/recovery-0.bin", std::ios::binary)), "missing final recovery");
    std::remove((out + "/recovery-0.bin").c_str());
    std::remove((out + "/recovery.lock").c_str());
    std::remove((out + "/state.fields").c_str());
    remove_dir(out + "/history");
    remove_dir(out);
    for (const char* name :
         {"main.txt", "gas.txt", "bc.txt", "init.txt", "cuda.txt", "implicit.txt", "output.txt", "run.log"})
        std::remove((dir + "/" + name).c_str());
    remove_dir(dir);
    return expected;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2 || argc == 3,
                "usage: aero_tests <aero executable> [CUDA ordinal | --expect-cpu-fallback]");
        const bool fallback = argc == 3 && std::string(argv[2]) == "--expect-cpu-fallback";
        const std::string device = fallback ? "0" : (argc == 3 ? argv[2] : "");
        check_partition_kernels();
        check_leaf_balance();
        const auto backend = run_case(argv[1], device, false, fallback);
        run_case(argv[1], device, true, fallback);
        run_case(argv[1], device, false, fallback, true);
        std::cout << "Aero end-to-end tests passed (" << backend
                  << ", Euler, SSPRK3, BDF1, interactive BC, reports and fields).\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
