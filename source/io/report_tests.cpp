#include "../parallel/fixed_matrix.h"
#include "report.h"
#include "../../tests/test_checks.h"
#include "../explicit/explicit.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_memory.h"
#endif

namespace {
using namespace aero;
using aero_test::require;
template <class F> void rejects(const F& f) {
    try {
        f();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected rejection");
}
// На двух объёмах вручную задаём экстремумы, интегралы и норму изменения энергии.
void table_example() {
    const double volumes[] = {0.25, 0.75};
    const std::vector<double> current = {1, 2, 0, 4.5, 2, -2, 4, 10};
    auto previous = current;
    previous[3] = 4.49;
    previous[7] = 10.02;
    std::ostringstream out;
    Report report(100, 1.4, 2, 2, volumes);
    for (unsigned step = 0; step < 100; ++step) {
        report.capture(step, std::vector<double>());
        report.write(step, 0, 0, 0, std::vector<double>(), out);
    }
    require(out.str().empty(), "unexpected report before step 100");
    rejects([&] { report.write(100, 0.875, 0.5, 0.01, current, out); });
    report.capture(100, previous);
    auto invalid = current;
    invalid[0] = -1;
    rejects([&] { report.write(100, 0.875, 0.5, 0.01, invalid, out); });
    rejects([&] { report.write(100, 0.875, 0.5, 0, current, out); });
    require(out.str().empty(), "incomplete report was printed");
    report.write(100, 0.875, 0.5, 0.01, current, out);
    rejects([&] { report.write(100, 0.875, 0.5, 0.01, current, out); });
    std::istringstream in(out.str());
    std::string line;
    std::getline(in, line);
    require(line.find("STEP 100 ") == 0 && line.size() == 72, "step and local date");
    std::getline(in, line);
    require(line == std::string(72, '-'), "report separator");
    std::getline(in, line);
    require(line == "t = 0.875    dt = 1.000000e-02    CFL = 0.5", "step parameters");
    while (std::getline(in, line) && line.find("Field") != 0) {}
    require(line.find("Domain integral") != std::string::npos, "column names");
    const char* names[] = {"rho", "ux", "uy", "p", "E"};
    // Столбцы эталона содержат минимум, максимум и сумму произведений значений на объёмы.
    const FixedMatrix<double,5,3> rows = {{1, 2, 1.75, -1, 2, -0.25, 0, 2, 1.5, 1, 2, 1.75, 4.5, 10, 8.625}};
    for (unsigned k = 0; k < 5; ++k) {
        std::string name;
        double lo, hi, integral;
        require(bool(in >> name >> lo >> hi >> integral), "missing row");
        require(name == names[k] && lo == rows(k, 0) && hi == rows(k, 1) && integral == rows(k, 2),
                "wrong row");
    }
    require(out.str().find("RMS(dE/dt) = 1.802776e+00") !=
                std::string::npos,
            "energy RMS");
    Report disabled(0, 1.4, 2, 2, volumes);
    disabled.capture(1, previous);
    disabled.write(1, 0.01, 0.5, 0.01, current, out);
    require(!disabled.due(1) && out.str().find("STEP", 1) == std::string::npos, "disabled reporter");
}

std::string uniform_3d() {
    const unsigned nodes = 1025;
    std::vector<double> volumes(nodes, 1.0), field;
    for (unsigned i = 0; i < nodes; ++i)
        field.insert(field.end(), {1, 0, 0, -2, 5});
    Report report(1, 1.4, 3, nodes, volumes.data());
    std::ostringstream out;
    report.capture(1, field);
    report.write(1, 0.25, 1, 0.25, field, out);
    require(out.str().find("uz ") != std::string::npos &&
                out.str().find("-2.050000e+03") != std::string::npos &&
                out.str().find("= 0.000000e+00") != std::string::npos,
            "3D volume reduction");
    return out.str().substr(out.str().find('\n')); // Исключаем время компьютера из сравнения.
}

// Проверяет отчёт после полных шагов интегратора для выбранной схемы и устройства.
void integration(ExplicitBackend backend, int device, ExplicitScheme scheme) {
    const double volumes[] = {0.25, 0.75}, initial[] = {1, 0, 4, 1, 0, 10};
    ExplicitIntegrator solver(2, 3, volumes, initial, backend, device);
    Report report(100, 1.4, 1, 2, volumes);
    std::ostringstream out;
    const double dt = 0.01;
    for (unsigned step = 1; step <= 200; ++step) {
        const double residual[] = {0, 0, -volumes[0] * (step == 200 ? 2.0 : 1.0),
                                   0, 0, step == 100 ? volumes[1] * 2 : 0};
        report.capture(step, solver);
        solver.step(scheme, (step - 1) * dt, dt, [&](double, const ExplicitField& field, double* r) {
#ifdef AERO_WITH_KOKKOS_GPU
            if (field.backend == ExplicitBackend::Cuda) {
                require(aero::gpu::copy_async(r, residual, sizeof(residual), aero::gpu::Copy::ToDevice,
                                        static_cast<aero::gpu::Stream>(field.stream)) == aero::gpu::Status::Success,
                        "CUDA test residual");
                return;
            }
#else
            (void)field;
#endif
            std::copy(residual, residual + 6, r);
        });
        report.write(step, step * dt, 0.5, dt, solver, out);
    }
    const std::string text = out.str();
    const auto second = text.find("STEP 200 ");
    const auto first = text.find("STEP 100 ");
    require(first < second && text.find("STEP", first + 1) == second && second != std::string::npos &&
                text.find("STEP", second + 1) == std::string::npos,
            "report schedule");
    require(text.find("uy ") == std::string::npos && text.find("uz ") == std::string::npos, "1D rows");
    require(text.find("= 1.802776e+00") < second && text.find("= 1.000000e+00", second) != std::string::npos,
            "RMS must use consecutive accepted steps");
}
}
int main(int argc, char** argv) {
    try {
        table_example();
#ifdef _OPENMP
        omp_set_num_threads(1);
        const auto serial = uniform_3d();
        omp_set_num_threads(4);
        require(serial == uniform_3d(), "OpenMP changed table");
#else
        uniform_3d();
#endif
        integration(ExplicitBackend::Cpu, 0, ExplicitScheme::Euler);
        integration(ExplicitBackend::Cpu, 0, ExplicitScheme::SSPRK3);
        const int devices = ExplicitIntegrator::cuda_device_count();
        if (argc > 1 && std::string(argv[1]) == "--require-cuda")
            require(devices > 0, "CUDA device required");
        for (int device = 0; device < devices; ++device) {
            integration(ExplicitBackend::Cuda, device, ExplicitScheme::Euler);
            integration(ExplicitBackend::Cuda, device, ExplicitScheme::SSPRK3);
        }
        std::cout << "Report tests passed (CPU and " << devices << " CUDA devices).\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
