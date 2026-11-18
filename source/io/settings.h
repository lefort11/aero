#ifndef AERO_IO_SETTINGS_H
#define AERO_IO_SETTINGS_H
#include "output_settings.h"

#include <string>
#include <cstdint>
#include <vector>

namespace aero {
struct RunDirectories {
    std::string input, output;
    bool interactive;
    bool geometry_only=false,recover=false;
    int recovery_slot=-1; // --recover alone selects the newest valid checkpoint
    RunDirectories(std::string in="", std::string out="", bool edit=false)
        : input(in),output(out),interactive(edit) {}
};

// Scheme names are validated independently of the installed solver modules.
struct RunSettings {
    RunDirectories directories;
    std::string mesh_path, initial_path, boundary_path;
    std::string reconstruction, reconstruction_variables, time_scheme, euler_solver;
    double cfl, final_time, gamma;
    double trace_ratio_min, trace_ratio_max;
    bool use_cuda;
    std::vector<int> devices;
    int omp_threads;
    std::vector<double> mpi_rank_weights,mpi_node_memory_mib,mpi_gpu_memory_mib;
    bool mpi_balance=false,mpi_timing=false;
    double mpi_memory_factor=4;
    std::uint64_t flowreport_every,recovery_every_steps=0;
    std::vector<OutputZone> output_zones;
};

RunDirectories parse_run_arguments(int argc, const char* const argv[]);
// Read main.txt, gas.txt, optional output.txt and, for CUDA runs, cuda.txt. The mesh-dependent files
// are consumed by read_initial_file and read_case_boundaries after mesh loading.
RunSettings read_run_settings(const RunDirectories& directories);
}
#endif
