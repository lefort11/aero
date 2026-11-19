#include "geometry.h"
#include "solver.h"
#include "../mesh/mdec.h"
#include <iostream>

namespace {
using namespace aero;

// Проверяет сочетания параметров, которые зависят от режима запуска и сборки.
void validate_run_mode(const RunSettings& run, const Processes& world, bool archive) {
    if (run.directories.recover && run.directories.geometry_only) {
        throw std::invalid_argument("--recover requires time integration");
    }
    if (world.size > 1 && !archive) {
        throw std::invalid_argument("Multiple processes require an AERODEC1 (.mdec) mesh");
    }
#ifndef AERO_WITH_CUDA
    if (world.size > 1 && !run.directories.geometry_only && run.use_cuda) {
        throw std::invalid_argument("Distributed CUDA calculation requires a build with MPI and CUDA");
    }
#endif
}

// Возвращает код ошибки запуска. В MPI сообщение печатает только процесс 0.
int report_error(const std::exception& error, int rank) {
    if (rank == 0) {
        std::cerr << "\naero: " << error.what() << '\n';
    }
    return 1;
}

// Читает постановку на каждом процессе и запускает выбранный режим.
int run_application(int argc, char** argv, const Processes& world) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        if (!world.rank) {
            std::cout
                << "Usage: aero --input=<directory> --output=<directory> [--interactive] [--geometry-only] "
                   "[--recover [0|1]]\n";
        }
        return 0;
    }
    RunSettings run{};
    bool archive = false;
    // Каждый процесс узнает об ошибке до первого коллективного шага расчёта.
    world.local([&] {
        run = read_run_settings(parse_run_arguments(argc, argv));
        archive = mdec::Archive::matches(run.mesh_path);
        validate_run_mode(run, world, archive);
    });
    if (run.directories.geometry_only) {
        export_case_geometry(run, world, archive);
        return 0;
    }
    ReconstructionOptions options;
    options.scheme = reconstruction_scheme(run.reconstruction).scheme;
    return calculate_case(run, options, world, archive);
}
}

int main(int argc, char** argv) {
    try {
        aero::Processes world(argc, argv);
        try {
            return run_application(argc, argv, world);
        } catch (const std::exception& error) {
            // Выводим причину до коллективного завершения MPI.
            return report_error(error, world.rank);
        }
    } catch (const std::exception& error) {
        return report_error(error, 0); // Ошибка запуска самого MPI.
    }
}
