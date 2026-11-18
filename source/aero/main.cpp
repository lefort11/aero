#include "startup.h"
#include "geometry.h"
#include "solver.h"
#include "mpi_balance.h"
#include "../mesh/mdec.h"
#include <iostream>
#include <map>

namespace {
using namespace aero;
ReconstructionOptions reconstruction(const std::string& name) {
    ReconstructionOptions r;
    const std::map<std::string,ReconstructionScheme> schemes={
        {"first_order",ReconstructionScheme::FirstOrder},{"ebr3",ReconstructionScheme::Ebr3},
        {"ebr5",ReconstructionScheme::Ebr5},{"weno3",ReconstructionScheme::Weno3},{"weno5",ReconstructionScheme::Weno5}};
    r.scheme=schemes.at(name);
    return r;
}
}
int main(int argc,char** argv) {
    try {
        aero::Processes world(argc,argv);
        try {
            if (argc==2 && std::string(argv[1])=="--help") {
                if (!world.rank) std::cout<<"Usage: aero --input=<directory> --output=<directory> [--interactive] [--geometry-only] [--recover [0|1]]\n";
                return 0;
            }
            aero::RunSettings run{};
            bool archive=false;
            world.local([&] {
                run=aero::read_run_settings(aero::parse_run_arguments(argc,argv));
                archive=aero::mdec::Archive::matches(run.mesh_path);
                if (run.directories.recover && run.directories.geometry_only)
                    throw std::invalid_argument("--recover requires time integration");

                if (world.size>1 && !archive)
                    throw std::invalid_argument("Multiple processes require an AERODEC1 (.mdec) mesh");
#ifndef AERO_WITH_CUDA
                if (world.size>1 && !run.directories.geometry_only && run.use_cuda)
                    throw std::invalid_argument("Distributed CUDA calculation requires a build with MPI and CUDA");
#endif
            });
            if (!run.directories.geometry_only)
                return aero::calculate_case(run,reconstruction(run.reconstruction),world,archive);
            const auto layout=aero::plan_partition(run,world);
            aero::MeshPartition partition;
            if (archive) partition=aero::load_partition(run.mesh_path,world,nullptr,layout);
            else partition.mesh=aero::Mesh::read(run.mesh_path);
            std::unique_ptr<aero::MedianDual> dual;
            world.local([&] { dual.reset(new aero::MedianDual(partition.mesh)); });
            world.local([&] { if (!world.rank) aero::prepare_output_directories(run); });
            world.local([&] {
                aero::write_geometry(run.directories.output,partition.mesh,dual->view(),archive?&partition:nullptr,world.rank,world.size);
            });
            if (!world.rank) std::cout<<"CPU geometry complete, processes = "<<world.size<<'\n';
            return 0;
        } catch (const std::exception& e) {
            if (!world.rank) std::cerr<<"\naero: "<<e.what()<<'\n';
            return 1;
        }
    }
    catch (const std::exception& e) { std::cerr<<"\naero: "<<e.what()<<'\n'; return 1; }
}
