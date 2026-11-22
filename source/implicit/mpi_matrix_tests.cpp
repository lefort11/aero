#include "../aero/partition_operator.h"
#include "bdf1_linear_system.h"
#include "../../tests/test_field.h"
#include <cmath>
#include <iostream>

namespace {
using namespace aero;
using implicit::BdfLinearSystem;
using Vector = std::vector<double>;

// Выборка из поля на цельной сетке, сначала в нумерацию раздела, затем в нумерацию его строк.
Vector select_nodes(const Vector& field, const std::vector<MeshIndex>& nodes, unsigned width) {
    Vector selected(nodes.size() * width);
    for (size_t k = 0; k < nodes.size(); ++k)
        std::copy_n(field.data() + nodes[k]*width, width, selected.data() + k*width);
    return selected;
}

void compare(const Processes& world, const char* mesh_path, const char* archive, const std::vector<int>& devices) {
    const bool gpu = !devices.empty();
    const auto part = load_partition(archive, world);
    const auto mesh = Mesh::read(mesh_path);
    MedianDual whole_dual(mesh), local_dual(part.mesh);
    const auto whole = whole_dual.view(), local = local_dual.view();
    const unsigned width = whole.dimension + 2;
    BoundaryPatches patches;
    patches.by_patch.resize(part.physical_patches);
    const auto physics = residual_options("characteristic", "lf", 1.4);
    ReconstructionOptions options;
    options.scheme = ReconstructionScheme::FirstOrder;
    SpatialResidual reference(mesh.view(), whole, prepare_reconstruction(mesh, options), patches, physics);
    BdfLinearSystem serial(reference);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu) select_partition_gpu(world, devices);
#else
    if (gpu) throw std::invalid_argument("GPU matrix test requires a Kokkos GPU build");
#endif
    PartitionOperator op(part, local, nullptr, patches, physics, ResidualFrame{}, world, nullptr, gpu, false);
    op.connect(gpu, true);
#ifdef AERO_WITH_KOKKOS_GPU
    if (gpu) world.local([&] {
        GpuMemoryFit fit;
        fit.budget = std::numeric_limits<MeshIndex>::max();
        implicit::NewtonSettings settings;
        settings.csgs=true;
        op.reserve_gpu(false, fit, &settings);
        aero_test::require(!fit.failed, "GPU memory for matrix test");
        op.layout_gpu(false);
    });
#endif
    const auto grid = op.linear_grid();
    BdfLinearSystem distributed(grid.host, grid.active, grid.device, grid.partition);
    Vector state(whole.node_count * width), direction(state.size()), product(state.size());
    for (MeshIndex i = 0; i < whole.node_count; ++i) {
        state[i*width] = i % 2 ? 1. : 1.4;
        state[i*width + 1] = .2;
        state[i*width + width-1] = 2.8;
    }
    for (MeshIndex k = 0; k < direction.size(); ++k) direction[k] = std::cos(.17 * static_cast<double>(k));
    const auto local_direction = select_nodes(direction, part.nodes, width);
    const auto expected_x = select_nodes(local_direction, op.owned, width);
    aero_test::TestField u(grid.device, local.node_count * width), v(grid.device, local.node_count),
                         x(grid.device, expected_x.size()), b(grid.device, expected_x.size());
    u.set(select_nodes(state, part.nodes, width));
    if (local.node_count) v.set(Vector(local.volumes, local.volumes + local.node_count));
    for (bool roe : {false,true}) for (bool simplified : {false, true}) {
        // Проверяем каждую собственную строку. Ошибка знака на стыке проявится даже тогда,
        // когда финальные поля скрывают её за счёт дополнительных итераций Ньютона.
        const double dt = .013;
        implicit::MatrixSettings settings;
        settings.roe = settings.csgs = roe;
        settings.simplified_jacobian = simplified;
        serial.assemble(state.data(), nullptr, whole.volumes, dt, settings);
        serial.apply(direction.data(), product.data());
        const auto expected_b = select_nodes(select_nodes(product, part.nodes, width), op.owned, width);
        distributed.assemble(u.data(), nullptr, v.data(), dt, settings);
        x.set(expected_x);
        distributed.apply(x.data(), b.data());
        double maximum = 0.;
        const auto& image = b.read();
        for (size_t k = 0; k < image.size(); ++k) maximum = std::max(maximum, std::abs(image[k] - expected_b[k]));
        maximum = world.reduce_double(maximum, true);
        aero_test::require(maximum < 1.e-12, "distributed A*x differs from the whole mesh");
        // Правая часть построена однопроцессной матрицей. Истинную невязку решения
        // проверяем дополнительным умножением после выхода из BiCGSTAB.
        b.set(expected_b);
        implicit::LinearTolerance tol;
        tol.relative = 1.e-11;
        const auto solved = distributed.solve(b.data(), x.data(), tol);
        aero_test::require(solved.status == implicit::LinearStatus::Converged, "distributed BiCGSTAB did not converge");
        distributed.apply(x.data(), b.data());
        const auto& ax = b.read();
        double error = 0.;
        for (size_t k = 0; k < ax.size(); ++k) error = std::max(error, std::abs(ax[k] - expected_b[k]));
        error = world.reduce_double(error, true);
        b.set(expected_b);
        const double scale = distributed.infinity_norm(b.data());
        aero_test::require(error <= tol.relative*scale && std::abs(error-solved.residual_norm) < 1.e-14*scale,
                           "incorrect true residual of the distributed system");
        if (!world.rank) std::cout << "MPI A*x max difference = " << maximum << ", linear relative residual = "
                                  << error/scale << ", Roe/CSGS = " << roe << ", simplified = " << simplified << '\n';
    }
}
}

int main(int argc, char** argv) {
    aero::Processes world(argc, argv);
    try {
        if (argc < 3 || world.size < 2) throw std::invalid_argument("Expected mesh and MPI archive");
        std::vector<int> devices;
        for (int i = 3; i < argc; ++i) devices.push_back(std::stoi(argv[i]));
        compare(world, argv[1], argv[2], devices);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
