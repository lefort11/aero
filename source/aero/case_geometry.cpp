#include "../parallel/backend.h"
#include "case_geometry.h"
#include "startup.h"
#include "../explicit/explicit.h"
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_check.h"
#endif
#include <fstream>
#include <iostream>

namespace aero {
namespace {
// Показывает исходные настройки и метки сетки перед проверкой граничных условий.
void show_single_mesh(const RunSettings& run, const Mesh& mesh) {
    if (!run.directories.recover && std::ifstream(run.initial_path)) {
        std::cout << "Initial settings\n" << settings_text(run.initial_path);
    }
    if (std::ifstream(run.boundary_path)) {
        std::cout << "Boundary settings\n" << settings_text(run.boundary_path);
    }
    const auto view = mesh.view();
    std::cout << "Dimension = " << view.dimension << "  nodes = " << view.node_count
              << "  cells = " << view.cell_count << '\n';
    show_boundary_labels(mesh);
}

// Выбирает карту для одного процесса. При отсутствии CUDA расчёт продолжится на CPU.
void select_single_device(RunSettings& run) {
    if (!run.use_cuda) {
        return;
    }
    const int device_count = ExplicitIntegrator::cuda_device_count();
    if (device_count == 0) {
        std::cout << "CUDA unavailable; running on CPU.\n";
        run.use_cuda = false;
        return;
    }
    if (run.devices.size() != 1) {
        throw std::invalid_argument("One calculation uses one GPU. Select one device in cuda.txt.");
    }
    const int device = run.devices.front();
    if (device >= device_count) {
        throw std::invalid_argument(std::string("Unavailable ") + gpu_backend_label + " device " + std::to_string(device));
    }
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::checked(gpu::select(device));
#endif
}
}

// Возвращает false, если пользователю нужно заполнить созданный файл граничных условий.
// В однопроцессном режиме также выбирает устройство до построения контрольных объёмов.
bool prepare_case_geometry(RunSettings& run, const MeshPartition& partition, const Processes& world,
                           bool first_attempt, CaseGeometry& geometry) {
    const auto mesh = partition.mesh.view();
    const bool single = world.size == 1;
    MeshData physical;
    bool ready = true;
    world.local([&] {
        // За физическими границами в архиве идут внутренние границы между процессами.
        const auto& names = partition.mesh.data().patch_names;
        if (run.background.active() && run.background_velocities != mesh.dimension)
            throw std::invalid_argument("linear.txt: specify one background velocity component for each mesh dimension");
        physical.dimension = mesh.dimension;
        physical.patch_names.assign(names.begin(), names.begin() + partition.physical_patches);
        if (world.rank == 0 && first_attempt) {
            show_settings(run);
            if (single) {
                show_single_mesh(run, partition.mesh);
            }
            ready = prepare_boundary_file(run, physical.patch_names);
        }
    });
    // Файл проверяет процесс 0. Решение о продолжении принимают все процессы.
    if (!world.minimum(ready ? 1.0 : 0.0)) {
        return false;
    }
    if (single) {
        if (!mesh.cell_count || !mesh.edge_count) {
            throw MeshError("The calculation requires a nonempty mesh");
        }
        select_single_device(run);
    }
    world.local([&] {
        const BoundaryTemperature temperature{{run.gamma, run.mach, run.reference_phi, run.reference_theta},
                                               run.background.rho, run.background.p,
                                               {run.background.velocity[0],run.background.velocity[1],run.background.velocity[2]}};
        geometry.boundaries = read_case_boundaries(run.boundary_path, physical, run.background.active(),
                                                   &run.input_warnings, temperature);
        if (single && run.directories.interactive) {
            std::cout << "Accepted boundary settings\n" << settings_text(run.boundary_path);
        }
        geometry.dual.reset(new MedianDual(partition.mesh));
    });
    if (geometry.periodic()) {
        geometry.seam =
            partition_periodic(partition, geometry.dual->view(), geometry.boundaries.periodic_pairs, world);
    }
    return true;
}

// В периодическом расчёте оператор уже забрал геометрию поверхностей.
// Нумерация и объёмы остаются доступны для шага по времени и записи полей.
void CaseGeometry::release_stencil_geometry() {
    if (!periodic()) {
        return; // В обычной сетке решатель продолжает использовать dual.
    }
    std::vector<PeriodicPartition::Edge>().swap(seam.edges);
    std::vector<CvBoundary>().swap(seam.boundary);
    dual.reset();
}
}
