#include "timestep.h"
#include "../../qa/test_checks.h"
#include "../bc/periodic.h"
#include "../explicit/explicit.h"
#include "../eulersolver/lax_friedrichs.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace aero;
#ifdef AERO_WITH_CUDA
CflStep gpu_inputs(int device, const CflTimestep&, double, WaveSpeeds);
CflStep gpu_accepted(const CflTimestep&, const ExplicitField&);
void gpu_residual(const ExplicitField&, double*);
#endif
namespace {
using aero_test::require;
void expect(CflStep step, double dt, std::size_t id) {
    require(std::fabs(step.dt - dt) < 2e-13 * dt && step.node == id, "incorrect CFL dt or limiting node");
}
template <class F> void rejects(F f) {
    try {
        f();
    } catch (const std::logic_error&) {
        return;
    }
    throw std::runtime_error("invalid input accepted");
}
CflStep run(int device, const CflTimestep& plan, double cfl, const std::vector<double>& edges,
            const std::vector<double>& boundary = {}) {
    WaveSpeeds s = {edges.data(), edges.size(), boundary.data(), boundary.size()};
#ifdef AERO_WITH_CUDA
    if (device >= 0)
        return gpu_inputs(device, plan, cfl, s);
#else
    (void)device;
#endif
    return plan.compute(cfl, s);
}
Mesh line(std::vector<double> coordinates) {
    MeshData m;
    m.dimension = 1;
    const MeshIndex n = coordinates.size();
    m.x = coordinates;
    m.y.resize(n);
    m.z.resize(n);
    for (MeshIndex i = 0; i < n; ++i) {
        m.face_nodes.indices.push_back(i);
        m.face_nodes.offsets.push_back(i + 1);
        if (i + 1 < n) {
            m.cell_faces.indices.insert(m.cell_faces.indices.end(), {i, i + 1});
            m.cell_faces.offsets.push_back(m.cell_faces.indices.size());
            m.cell_face_orientation.insert(m.cell_face_orientation.end(), {-1, 1});
        }
    }
    m.patch_names = {"left", "right"};
    m.boundary_faces = {0, n - 1};
    m.boundary_patch = {0, 1};
    return Mesh(std::move(m));
}
void cases(int device) {
    Mesh m = line({0, 1, 3});
    const MedianDual cv(m);
    CflTimestep ordinary(m.view(), cv.view(), device);
    expect(run(device, ordinary, 1, {1, 2}, {3, 1}), .25, 0);
    expect(run(device, ordinary, .4, {0, 0}, {0, 2}), .4, 2);
    rejects([&] { run(device, ordinary, 1, {0, 0}, {0, 0}); });
    rejects([&] { run(device, ordinary, 1, {1, 2}, {1, -1}); });
    rejects([&] { run(device, ordinary, 1, {1, std::numeric_limits<double>::quiet_NaN()}, {1, 1}); });
    rejects([&] { run(device, ordinary, 0, {1, 1}, {1, 1}); });
    rejects([&] { ordinary.compute(1, {nullptr, 0, nullptr, 0}); });

    PeriodicPair p = {"left", "right", {3, 0, 0}};
    PeriodicDomain ring(m, {p});
    CflTimestep loop(ring.view(), device);
    expect(run(device, loop, 1, {1, 2}), 1, 0);
    Mesh uniform = line({0, 1, 2, 3});
    PeriodicDomain periodic(uniform, {p});
    CflTimestep grid(periodic.view(), device);
    expect(run(device, grid, .5, {4, 4, 4}), .125, 0);

    // При периоде в один отрезок оба его конца принадлежат одному узлу. В сумму входят оба вклада.
    Mesh one = line({0, 2});
    p.translation[0] = 2;
    PeriodicDomain single(one, {p});
    CflTimestep self(single.view(), device);
    expect(run(device, self, .5, {4}), .25, 0);

    // Вычисляем скорость волн физического потока Эйлера на принятом состоянии до и после шага.
    // При выполнении на GPU состояние всё время остаётся на устройстве.
    const double v = 2, initial[] = {1, 0, 2.5};
    ExplicitIntegrator solver(1, 3, &v, initial, device < 0 ? ExplicitBackend::Cpu : ExplicitBackend::Cuda,
                              device < 0 ? 0 : device);
    for (int step = 0; step < 2; ++step) {
        CflStep actual = {};
        solver.inspect([&](const ExplicitField& f) {
#ifdef AERO_WITH_CUDA
            if (device >= 0) {
                actual = gpu_accepted(self, f);
                return;
            }
#endif
            double n = 1;
            EulerFluxResult flux = {};
            require(lax_friedrichs_conservative(1, f.values, f.values, &n, 1.4, &flux) ==
                        EulerStatus::Success,
                    "flux evaluation");
            actual = self.compute(.5, {&flux.max_wave_speed, 1, nullptr, 0});
        });
        const double pressure = step == 0 ? 1 : 2;
        expect(actual, 1 / std::sqrt(1.4 * pressure), 0);
        if (!step)
            solver.step(ExplicitScheme::Euler, 0, 1, [&](double, const ExplicitField& f, double* r) {
#ifdef AERO_WITH_CUDA
                if (device >= 0) {
                    gpu_residual(f, r);
                    return;
                }
#else
            (void)f;
#endif
                r[0] = 0;
                r[1] = 0;
                r[2] = -5;
            });
    }

    // Строк больше, чем потоков OpenMP или нитей блока CUDA. Наклонная площадка имеет модуль 5.
    // Равные минимумы встречаются внутри одного блока и в разных блоках.
    const std::size_t count = 5003;
    std::vector<double> vol(count, 10), speed(count, 1);
    std::vector<PeriodicEdge> edges(count);
    for (std::size_t i = 0; i < count; ++i)
        edges[i] = {i, (i + 1) % count, i, {3, 4, 0}};
    vol[3901] = vol[3904] = vol[4097] = 2;
    PeriodicView artificial = {count, count, 0, count, nullptr, vol.data(), edges.data(), nullptr};
    CflTimestep large(artificial, device);
    expect(run(device, large, .5, speed), .2, 3901);
    speed.back() = std::numeric_limits<double>::infinity();
    rejects([&] { run(device, large, .5, speed); });
}
}
int main(int argc, char** argv) {
    try {
        cases(-1);
        std::cout << "CPU CFL tests passed.\n";
        const int devices = ExplicitIntegrator::cuda_device_count();
        if (argc == 2 && std::string(argv[1]) == "--require-cuda")
            require(devices > 0, "CUDA device required");
        for (int dev = 0; dev < devices; ++dev) {
            cases(dev);
            std::cout << "CFL tests passed on GPU " << dev << ".\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
