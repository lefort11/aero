#include "geometry.h"
#include "console_log.h"
#include "startup.h"
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
#ifndef AERO_WITH_KOKKOS_GPU
    if (world.size > 1 && !run.directories.geometry_only && run.use_cuda) {
        throw std::invalid_argument("Distributed GPU calculation requires a build with MPI and a GPU backend");
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

// Один текст обслуживает запрос справки и ошибки аргументов командной строки.
void print_usage(std::ostream& out) {
    out << "Usage: aero [--input=DIR --output=DIR] [OPTIONS] [FILE:KEY=VALUE ...]\n\n"
           "  --input=DIR      Read main.txt and the other case files from DIR.\n"
           "  --output=DIR     Write fields and recovery files to DIR.\n"
           "                   Supply both paths together. If both are omitted, use the current directory.\n"
           "  --geometry-only  Export control-volume geometry.\n"
           "  --interactive    Pause for editing missing boundary conditions.\n"
           "  --recover [0|1]  Resume from the latest valid checkpoint or the specified slot.\n"
           "  FILE:KEY=VALUE   Override a setting for this run. The .txt suffix is optional.\n"
           "  -h, --help       Show this help.\n\n"
           "Examples:\n"
           "  aero --input=case --output=results main:cfl=0.4 gas:gamma=1.4\n"
           "  aero --input=. --output=.\n";
}

// Читает постановку на каждом процессе и запускает выбранный режим.
int run_application(int argc, char** argv, const Processes& world, ConsoleLog& log) {
    const bool no_arguments = argc == 1;
    const bool help_requested = argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help");
    if (no_arguments || help_requested) {
        if (world.rank == 0)
            print_usage(help_requested ? std::cout : std::cerr);
        return help_requested ? 0 : 1;
    }
    RunDirectories directories;
    try {
        world.local([&] { directories = parse_run_arguments(argc, argv); });
    } catch (const std::exception& error) {
        if (world.rank == 0) {
            std::cerr << "aero: " << error.what() << "\n\n";
            print_usage(std::cerr);
        }
        return 1;
    }
    RunSettings run{};
    bool archive = false;
    world.local([&] {
        if (!world.rank) {
            create_output_directory(directories.output);
            log.open(directories.output + "/aero.log");
        }
    });
    // Ошибка входного файла относится к постановке, поэтому сообщаем её отдельно.
    world.local([&] {
        run = read_run_settings(directories);
        archive = mdec::Archive::matches(run.mesh_path);
        validate_run_mode(run, world, archive);
    });
    if (run.directories.geometry_only) {
        export_case_geometry(run, world, archive);
        return 0;
    }
    ReconstructionOptions options;
    if (run.stencil_geometry == "ijk") options.geometry = StencilGeometry::Ijk;
    if (run.stencil_geometry == "pl") options.geometry = StencilGeometry::Pl;
    if (run.stencil_geometry == "auto") options.geometry = StencilGeometry::Auto;
    options.max_turn_degrees = run.stencil_max_angle;
    options.scheme = reconstruction_scheme(run.reconstruction).scheme;
    options.mpweno_ad_corr = run.mpweno_ad_corr;
    if (reconstruction_scheme(options.scheme).family == ReconstructionFamily::Mp) {
        options.mp5_alpha = mp5_alpha_for_time_scheme(run.cfl, run.uses_implicit_time());
        if (!world.rank) std::cout << "MP alpha = " << options.mp5_alpha << '\n';
    }
    return calculate_case(run, options, world, archive);
}
}

// Инициализирует MPI, запускает решатель и преобразует исключения в код завершения.
int main(int argc, char** argv) {
    try {
        aero::Processes world(argc, argv);
        aero::ConsoleLog log;
        try {
            return run_application(argc, argv, world, log);
        } catch (const std::exception& error) {
            // Выводим причину до коллективного завершения MPI.
            return report_error(error, world.rank);
        }
    } catch (const std::exception& error) {
        return report_error(error, 0); // Ошибка запуска самого MPI.
    }
}
