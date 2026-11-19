#include "../cuda/check.h"
#include "partition_gpu.h"
#include "partition_report.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>

namespace aero {
namespace {
void cuda_check(cudaError_t status) {
    cuda_detail::checked(status, "MPI CUDA: ");
}
}
// Замеряем рабочее ядро потока на одинаковых гранях с узловыми состояниями на всех GPU.
// Геометрия и данные уже на устройстве, поэтому время обменов в замер не входит.
double PartitionGpu::measure_flux_rate(ResidualOptions options, std::size_t count, MeshIndex budget) {
    PartitionGpu sample;
    std::vector<residual_detail::Surface> faces(count);
    std::vector<double> state(5 * count);
    for (std::size_t i = 0; i < count; ++i) {
        faces[i] = {i, (i + 1) % count, 0, {1, 0, 0}, 1, true};
        state[5 * i] = 1 + .001 * (i % 31);
        state[5 * i + 1] = .125;
        state[5 * i + 4] = 2.5;
    }
    residual_detail::FaceBatch batch{};
    batch.geometry.dim = 3;
    batch.geometry.options = options;
    batch.geometry.nodes = batch.geometry.edges = batch.geometry.faces = batch.count = count;
    const auto layout = [&] {
        batch.geometry.surfaces = sample.storage_.upload(faces);
        batch.state = sample.storage_.upload(state);
        batch.flux = sample.storage_.allocate<EulerFluxResult>(count);
        sample.grid_.bad = sample.storage_.allocate<residual_detail::Index>(1);
    };
    layout();
    GpuMemoryFit fit;
    fit.budget = budget;
    if (!sample.reserve(fit))
        return 0;
    layout();
    check(cudaMemsetAsync(sample.grid_.bad, 0xff, sizeof(*sample.grid_.bad), nullptr));
    const auto launch = [&] { residual_detail::gpu_face_fluxes(batch, sample.grid_.bad, nullptr); };
    launch();
    cuda_check(cudaStreamSynchronize(nullptr));
    std::array<double, 5> seconds;
    constexpr unsigned repeats = 8;
    for (double& elapsed : seconds) {
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < repeats; ++i)
            launch();
        cuda_check(cudaStreamSynchronize(nullptr));
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    sample.check_kernel("calibration");
    std::sort(seconds.begin(), seconds.end());
    const double rate = count * repeats / seconds[seconds.size() / 2];
    if (!(rate > 0 && std::isfinite(rate)))
        throw std::runtime_error("GPU calibration: invalid elapsed time");
    return rate;
}
std::vector<double> PartitionGpu::measure_weights(const Processes& world, ResidualOptions options,
                                                  MeshIndex budget) {
    std::vector<double> rates;
    double rate = 0;
    world.local([&] { rates.resize(world.size, 1); });
    // Сравниваем одинаковые объёмы работы, даже если на одном процессе доступно мало памяти.
    constexpr MeshIndex bytes_per_face =
        sizeof(residual_detail::Surface) + 5 * sizeof(double) + sizeof(EulerFluxResult);
    const MeshIndex capacity = budget > 1024 ? (budget - 1024) / bytes_per_face : 0;
    auto count =
        static_cast<std::size_t>(world.minimum(static_cast<double>(std::min<MeshIndex>(32768, capacity))));
    for (; count; count /= 2) {
        world.local([&] { rate = measure_flux_rate(options, count, budget); });
        if (world.minimum(rate) > 0)
            break;
    }
    if (!count) {
        if (!world.rank)
            std::cout << "MPI GPU calibration: insufficient workspace, using equal weights\n";
        return rates;
    }
    world.allgather(&rate, 1, rates.data());
    const double fastest = *std::max_element(rates.begin(), rates.end());
    for (int rank = 0; rank < world.size; ++rank) {
        const double weight = rates[rank] / fastest;
        if (!world.rank)
            std::cout << "MPI GPU calibration: rank " << rank << " faces_per_s " << rates[rank] << " weight "
                      << weight << " sample_faces " << count << '\n';
        rates[rank] = weight;
    }
    return rates;
}
void select_partition_gpu(const Processes& world, const std::vector<int>& devices) {
    const auto node = partition_node_resources(world);
    std::string physical, description;
    world.local([&] {
        int count = 0;
        cuda_check(cudaGetDeviceCount(&count));
        // Планировщик может выдать каждому процессу отдельную карту с одним и тем же локальным номером 0.
        const int slot = count == 1 && devices.size() == 1 ? 0 : node.local_rank;
        if (static_cast<std::size_t>(slot) >= devices.size())
            throw std::invalid_argument("cuda.txt: need one GPU per node-local MPI rank");
        const int device = devices[slot];
        if (device >= count)
            throw std::invalid_argument("Unavailable CUDA device " + std::to_string(device));
        cuda_check(cudaSetDevice(device));
        char bus[32]{};
        cuda_check(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));
        physical = std::to_string(node.leader) + "/" + bus;
        description = "rank " + std::to_string(world.rank) + " local_rank " +
                      std::to_string(node.local_rank) + " device " + std::to_string(device) + " node/PCI " +
                      physical;
    });
    const auto keys = world.gather_text(physical);
    const auto lines = world.gather_text(description);
    world.local([&] {
        if (world.rank)
            return;
        std::set<std::string> assigned;
        for (const auto& key : keys)
            if (!assigned.insert(key).second)
                throw std::invalid_argument("MPI ranks on one node selected the same GPU: " + key);
        for (const auto& line : lines)
            std::cout << "MPI CUDA placement: " << line << '\n';
    });
}
void PartitionGpu::check(cudaError_t status) {
    cuda_check(status);
}
void PartitionGpu::synchronize() {
    check(cudaStreamSynchronize(nullptr));
}
void query_gpu_memory(GpuMemoryFit& fit, const std::vector<double>& limits, const Processes& world) {
    std::size_t available = 0, total = 0;
    cuda_check(cudaMemGetInfo(&available, &total));
    fit.free_bytes = available;
    fit.budget = available - available / 20;
    if (!limits.empty()) {
        if (limits.size() != 1 && limits.size() != static_cast<std::size_t>(world.size))
            throw std::invalid_argument("mpi_gpu_memory_mib: expected one value or one limit per MPI rank");
        const long double bytes =
            static_cast<long double>(limits[limits.size() == 1 ? 0 : world.rank]) * 1048576;
        if (bytes < fit.budget)
            fit.budget = static_cast<MeshIndex>(bytes);
    }
}
PartitionGpu::Storage::~Storage() {
    if (base)
        cudaFree(base);
}
void PartitionGpu::layout(residual_detail::Geometry g, const PartitionStencil& stencil,
                          const std::vector<MeshIndex>& computed, const std::vector<MeshIndex>& owned,
                          const double* volumes, std::size_t state_nodes, bool compact_buffer,
                          bool whole_mesh) {
    g.surfaces = storage_.allocate(g.faces, g.surfaces);
    g.incidence = storage_.allocate(g.offsets[g.nodes], g.incidence);
    g.offsets = storage_.allocate(g.nodes + 1, g.offsets);
    g.patches = upload_boundary_patches(storage_, g.patches);
    g.stencil.offsets = storage_.upload(stencil.linear.offsets);
    g.stencil.nodes = storage_.upload(stencil.linear.nodes);
    g.stencil.weights = storage_.upload(stencil.linear.weights);
    g.stencil.row_owner = storage_.upload(stencil.row_owner);
    g.stencil.slope_offsets = storage_.upload(stencil.slopes.offsets);
    g.stencil.slope_nodes = storage_.upload(stencil.slopes.nodes);
    g.stencil.slope_weights = storage_.upload(stencil.slopes.weights);
    g.stencil.weno_order = storage_.upload(stencil.weno_order);
    g.stencil.weno_epsilon_scale = storage_.upload(stencil.epsilon_scale);
    grid_.geometry = g;
    grid_.volumes = storage_.allocate(g.nodes, volumes);
    grid_.computed = whole_mesh ? nullptr : storage_.upload(computed);
    grid_.owned = whole_mesh ? nullptr : storage_.upload(owned);
    grid_.computed_count = computed.size();
    grid_.owned_count = owned.size();
    grid_.flux = storage_.allocate<EulerFluxResult>(g.faces);
    const unsigned width = g.dim + 2;
    if (state_nodes > std::numeric_limits<std::size_t>::max() / width)
        throw std::length_error("GPU state too large");
    grid_.initial = storage_.allocate<double>(state_nodes * width);
    grid_.state = storage_.allocate<double>(state_nodes * width);
    if (compact_buffer)
        grid_.original = storage_.allocate<double>(owned.size() * (g.dim + 2));
    minimum_ = storage_.allocate<residual_detail::DeviceCflMinimum>(1);
    grid_.bad = minimum_ ? &minimum_->bad : nullptr;
}
bool PartitionGpu::reserve(GpuMemoryFit& fit) {
    fit.required = storage_.used;
    if (fit.required > fit.budget)
        return false;
    void* memory = nullptr;
    const auto status = cudaMalloc(&memory, storage_.used);
    if (status == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        return false;
    }
    check(status);
    storage_.base = static_cast<unsigned char*>(memory);
    storage_.capacity = storage_.used;
    storage_.used = 0;
    return true;
}
void PartitionGpu::upload_state(const std::vector<double>& state, const std::vector<double>& initial) {
    if (storage_.used != storage_.capacity)
        throw std::logic_error("GPU layout changed after reservation");
    if (!state.empty()) {
        check(cudaMemcpy(grid_.state, state.data(), state.size() * sizeof(double), cudaMemcpyHostToDevice));
        check(cudaMemcpy(grid_.initial, initial.data(), initial.size() * sizeof(double),
                         cudaMemcpyHostToDevice));
    }
    check(cudaMemsetAsync(grid_.bad, 0xff, sizeof(*grid_.bad), nullptr));
}
void PartitionGpu::check_kernel(const char* phase) {
    partition_detail::Index bad;
    check(cudaMemcpy(&bad, grid_.bad, sizeof(bad), cudaMemcpyDeviceToHost));
    if (bad != ~partition_detail::Index(0))
        throw std::domain_error(std::string("MPI CUDA ") + phase + " at local index " + std::to_string(bad));
}
void PartitionGpu::fluxes() {
    residual_detail::gpu_face_fluxes({grid_.geometry, grid_.computed, grid_.computed_count, grid_.state,
                                      grid_.initial, grid_.flux, nullptr, nullptr},
                                     grid_.bad, nullptr);
    check_kernel("flux");
}
double PartitionGpu::cfl(double courant) {
    return residual_detail::gpu_flux_cfl(
        {grid_.geometry, grid_.flux, grid_.volumes, grid_.owned, grid_.owned_count}, courant, minimum_,
        nullptr);
}
void PartitionGpu::update(explicit_detail::Stage stage, double dt, bool save_original) {
    partition_detail::update(grid_, stage, dt, save_original);
    check_kernel("update");
}
void PartitionGpu::download_owned(std::vector<double>& compact) const {
    if (compact.empty())
        return;
    // Отчёт снимается перед первой или после последней стадии. В эти моменты буфер начала шага RK свободен.
    partition_detail::transfer(grid_, grid_.owned, grid_.owned_count, grid_.original, false, true);
    check(
        cudaMemcpy(compact.data(), grid_.original, compact.size() * sizeof(double), cudaMemcpyDeviceToHost));
}
void PartitionGpu::download(std::vector<double>& state) const {
    if (!state.empty())
        check(cudaMemcpy(state.data(), grid_.state, state.size() * sizeof(double), cudaMemcpyDeviceToHost));
}
PartitionGpu::Peer PartitionGpu::prepare(const std::vector<MeshIndex>& out, const std::vector<MeshIndex>& in,
                                         unsigned stride) {
    Peer peer;
    peer.out = storage_.upload(out);
    peer.in = storage_.upload(in);
    peer.buffer = storage_.allocate<double>(std::max(out.size(), in.size()) * stride);
    return peer;
}
void PartitionGpu::exchange(const Peer& peer, DoubleExchange::Buffer records, double* mapped, bool edges,
                            bool pack) {
    if (!records.count)
        return;
    const auto count = records.count / (grid_.geometry.dim + 2 + unsigned(edges));
    const auto bytes = records.count * sizeof(double);
    // Для короткого сообщения ядро упаковки может напрямую обратиться к закреплённой памяти.
    if (mapped && bytes <= 128) {
        partition_detail::transfer(grid_, pack ? peer.out : peer.in, count, mapped, edges, pack);
        return;
    }
    if (pack) {
        partition_detail::transfer(grid_, peer.out, count, peer.buffer, edges, true);
        check(cudaMemcpyAsync(records.data, peer.buffer, bytes, cudaMemcpyDeviceToHost, nullptr));
    } else {
        check(cudaMemcpyAsync(peer.buffer, records.data, bytes, cudaMemcpyHostToDevice, nullptr));
        partition_detail::transfer(grid_, peer.in, count, peer.buffer, edges, false);
    }
}
}
