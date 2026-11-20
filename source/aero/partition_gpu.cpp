#include "../parallel/device_check.h"
#include "partition_gpu.h"
#ifdef AERO_HAS_VISCOSITY
#include "../parallel/device_work.h"
#endif
#include "partition_report.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>

namespace aero {
namespace {
void cuda_check(aero::gpu::Status status) {
    gpu::checked(status, "MPI CUDA: ");
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
        state[5 * i] = 1 + .001 * static_cast<double>(i % 31);
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
    check(aero::gpu::fill_async(sample.grid_.bad, 0xff, sizeof(*sample.grid_.bad), nullptr));
    const auto launch = [&] { residual_detail::gpu_face_fluxes(batch, sample.grid_.bad, nullptr); };
    launch();
    cuda_check(aero::gpu::wait(nullptr));
    std::array<double, 5> seconds;
    constexpr unsigned repeats = 8;
    for (double& elapsed : seconds) {
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < repeats; ++i)
            launch();
        cuda_check(aero::gpu::wait(nullptr));
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    sample.check_kernel("calibration");
    std::sort(seconds.begin(), seconds.end());
    const double rate = static_cast<double>(count * repeats) / seconds[seconds.size() / 2];
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
        cuda_check(aero::gpu::count(&count));
        // Планировщик может выдать каждому процессу отдельную карту с одним и тем же локальным номером 0.
        const int slot = count == 1 && devices.size() == 1 ? 0 : node.local_rank;
        if (static_cast<std::size_t>(slot) >= devices.size())
            throw std::invalid_argument("cuda.txt: need one GPU per node-local MPI rank");
        const int device = devices[slot];
        if (device >= count)
            throw std::invalid_argument("Unavailable CUDA device " + std::to_string(device));
        cuda_check(aero::gpu::select(device));
        char bus[32]{};
        cuda_check(aero::gpu::bus_id(bus, sizeof(bus), device));
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
void PartitionGpu::check(aero::gpu::Status status) {
    cuda_check(status);
}
void PartitionGpu::synchronize() {
    check(aero::gpu::wait(nullptr));
}
void query_gpu_memory(GpuMemoryFit& fit, const std::vector<double>& limits, const Processes& world) {
    std::size_t available = 0, total = 0;
    cuda_check(aero::gpu::memory_info(&available, &total));
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
        aero::gpu::release(base);
}
void PartitionGpu::layout(residual_detail::Geometry g, const PartitionStencil& stencil,
                          const std::vector<MeshIndex>& computed, const std::vector<MeshIndex>& owned,
                          const double* volumes, std::size_t state_nodes, bool compact_buffer,
                          bool whole_mesh) {
    const bool store_face_batch = g.packed.tiles != nullptr;
    if (g.packed.tiles) {
        const auto count = (g.edges + 31) / 32;
        // Один буфер обслуживает все стадии. Большую подобласть проходим частями:
        // отложенные веса не требуют дополнительной памяти пропорционально всей сетке.
        if (g.packed.order == 5 && g.stencil.mp5.weno && g.options.mp6_hybrid == 0) {
            using Tile = residual_detail::MpEdgeTile;
            const auto faces = std::min<MeshIndex>(computed.size(), MeshIndex(Tile::batch_limit));
            g.packed.mp_work = storage_.allocate<Tile>((faces + Tile::width - 1) / Tile::width);
        }
        if (g.packed.order == 5)
            g.packed.tiles = storage_.allocate(count, static_cast<const residual_detail::WenoEdgeTile<5>*>(g.packed.tiles));
        else
            g.packed.tiles = storage_.allocate(count, static_cast<const residual_detail::WenoEdgeTile<3>*>(g.packed.tiles));
        g.packed.remaining = storage_.allocate(g.packed.count, g.packed.remaining);
    }
    g.packed.gas = storage_.allocate<EulerState>(g.packed.state_nodes);
    g.packed.normals = storage_.allocate<residual_detail::FaceNormal>(g.packed.state_nodes ? g.edges : 0);
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
#ifdef AERO_HAS_VISCOSITY
    if (aes_) {
        viscous_grid_ = aes_->upload(storage_,state_nodes);
        g.options.diffusion = viscous_grid_.rows;
        g.options.wall = storage_.upload(aes_->wall);
    }
#endif
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
    if (g.options.retry_nonphysical) {
        accepted_state_ = storage_.allocate<double>(state_nodes * width);
        grid_.failed_count = storage_.allocate<partition_detail::Index>(1);
        grid_.failed_nodes = storage_.allocate<partition_detail::Index>(owned.size());
    }
    if (compact_buffer)
        grid_.original = storage_.allocate<double>(owned.size() * (g.dim + 2));
    grid_.bad = storage_.allocate<partition_detail::Index>(1);
    // Выделяем запись в том же порядке при оценке памяти и при заполнении. После
    // резервирования поля обновляются на месте, поэтому их адреса сохраняются между стадиями.
    if (store_face_batch) {
        const auto batch = face_batch();
        device_faces_ = storage_.allocate(1, &batch);
    }
}
bool PartitionGpu::reserve(GpuMemoryFit& fit, MeshIndex implicit_bytes) {
    fit.required = partition_bytes(storage_.used, 1, implicit_bytes);
    if (fit.required > fit.budget)
        return false;
    void* memory = nullptr;
    const auto status = aero::gpu::allocate(&memory, storage_.used);
    if (status == aero::gpu::Status::OutOfMemory) {

        return false;
    }
    check(status);
    storage_.base = static_cast<unsigned char*>(memory);
    storage_.capacity = storage_.used;
    storage_.used = 0;
    return true;
}
void PartitionGpu::upload_state(const std::vector<double>& state, const std::vector<double>& initial) {
    gas_current_ = false;
    if (storage_.used != storage_.capacity)
        throw std::logic_error("GPU layout changed after reservation");
    if (grid_.geometry.packed.normals)
        residual_detail::gpu_prepare_face_normals(grid_.geometry);
    if (!state.empty()) {
        check(aero::gpu::copy(grid_.state, state.data(), state.size() * sizeof(double), aero::gpu::Copy::ToDevice));
        check(aero::gpu::copy(grid_.initial, initial.data(), initial.size() * sizeof(double),
                         aero::gpu::Copy::ToDevice));
    }
    // Ядро меняет индекс только при ошибке. На обычной стадии процессору достаточно
    // дождаться ядра и прочитать закреплённое слово, без отдельной передачи PCIe.
    if (!error_index_.host()) {
        const auto status = error_index_.allocate(grid_.failed_count ? 2 : 1);
        if (status != aero::gpu::Status::OutOfMemory)
            check(status);
    }
    if (error_index_.device()) {
        *error_index_.host() = ~partition_detail::Index(0);
        grid_.bad = error_index_.device();
        // Второе слово того же закреплённого буфера хранит число отказов.
        // После обязательного ожидания стадии его уже можно читать на CPU.
        if (grid_.failed_count) grid_.failed_count = error_index_.device() + 1;
    } else {
        check(aero::gpu::fill_async(grid_.bad, 0xff, sizeof(*grid_.bad), nullptr));
    }
}
void PartitionGpu::check_kernel(const char* phase) {
    partition_detail::Index bad;
    if (error_index_.device()) {
        synchronize();
        bad = *error_index_.host();
    } else {
        check(aero::gpu::copy(&bad, grid_.bad, sizeof(bad), aero::gpu::Copy::ToHost));
    }
    if (bad != ~partition_detail::Index(0)) {
        // Пробное поле предиктора может быть отклонено. Следующая проверка
        // должна видеть ошибки нового запуска, поэтому сбрасываем отметку после чтения.
        if (error_index_.device()) *error_index_.host() = ~partition_detail::Index(0);
        else check(aero::gpu::fill_async(grid_.bad, 0xff, sizeof(*grid_.bad), nullptr));
        throw std::domain_error(std::string("MPI CUDA ") + phase + " at local index " + std::to_string(bad));
    }
}
void PartitionGpu::update_surfaces(const std::vector<residual_detail::Surface>& surfaces) {
    if (!surfaces.empty()) check(gpu::copy(const_cast<residual_detail::Surface*>(grid_.geometry.surfaces),
        surfaces.data(), surfaces.size()*sizeof(surfaces[0]), gpu::Copy::ToDevice));
}
residual_detail::FaceBatch PartitionGpu::face_batch() const {
    return {grid_.geometry, grid_.computed, grid_.computed_count, grid_.state,
            grid_.initial, grid_.flux, nullptr, nullptr, gas_current_};
}
void PartitionGpu::set_alpha(double alpha) {
    auto& mp = grid_.geometry.stencil.mp5;
    if ((!mp.enabled && !mp.weno) || mp.alpha == alpha) return;
    mp.alpha = alpha;
    update_face_batch();
}
void PartitionGpu::set_flux_options(const ResidualOptions& options) {
    grid_.geometry.options.upw_coeff = options.upw_coeff;
    grid_.geometry.options.mp6_hybrid = options.mp6_hybrid;
    update_face_batch();
}
void PartitionGpu::update_face_batch() {
    // Упакованные ядра MP читают настройки из постоянной записи на карте.
    if (device_faces_) {
        const auto batch = face_batch();
        check(gpu::copy(device_faces_, &batch, sizeof(batch), gpu::Copy::ToDevice));
    }
}
void PartitionGpu::fluxes(bool first_order_start,bool defer_viscous_rows) {
    (void)defer_viscous_rows;
    auto batch = face_batch();
    if (first_order_start) batch.use_first_order();
    residual_detail::gpu_face_fluxes(batch, grid_.bad, nullptr, first_order_start ? nullptr : device_faces_);
#ifdef AERO_HAS_VISCOSITY
    if (aes_) {
        const auto error = device_work::aero_device_viscosity(viscous_grid_,grid_.state,grid_.geometry.packed.gas,
            defer_viscous_rows?viscosity::Evaluation::BeforeSaIterations:viscosity::Evaluation::Full);
        if (error) throw std::runtime_error(error);
    }
#endif
    check_kernel("flux");
}
void PartitionGpu::update(explicit_detail::Stage stage, double dt, bool save_original) {
    if (grid_.failed_count) reset_failed_count();
    partition_detail::update(grid_, stage, dt, save_original);
    check_kernel("invalid stage state");
    gas_current_ = !grid_.owned;
}
void PartitionGpu::copy_backup(std::size_t count, bool restore) {
    // Индексы соседей входят в снимок: после отката можно сразу пересчитать потоки.
    if (count) check(gpu::copy_async(restore ? grid_.state : accepted_state_,
        restore ? accepted_state_ : grid_.state, count*sizeof(double), gpu::Copy::OnDevice, nullptr));
    if (restore) gas_current_ = false;
}
void PartitionGpu::reset_failed_count() {
    if (error_index_.device()) error_index_.host()[1] = 0;
    else check(gpu::fill_async(grid_.failed_count, 0, sizeof(*grid_.failed_count), nullptr));
}
void PartitionGpu::failed_nodes(std::vector<MeshIndex>& failed, bool implicit) {
    if (implicit) {
        reset_failed_count();
        partition_detail::check_newton_nodes(grid_);
        if (error_index_.device()) synchronize();
    }
    partition_detail::Index count;
    if (error_index_.device()) count = error_index_.host()[1];
    else check(gpu::copy(&count, grid_.failed_count, sizeof(count), gpu::Copy::ToHost));
    failed.resize(count);
    if (count) check(gpu::copy(failed.data(), grid_.failed_nodes, count*sizeof(failed[0]), gpu::Copy::ToHost));
}

void PartitionGpu::download_owned(std::vector<double>& compact) const {
    if (compact.empty())
        return;
    // Отчёт снимается перед первой или после последней стадии. В эти моменты буфер начала шага RK свободен.
    partition_detail::transfer({grid_.geometry.components(), grid_.state, nullptr}, grid_.owned,
                               grid_.owned_count, nullptr, grid_.original);
    check(
        aero::gpu::copy(compact.data(), grid_.original, compact.size() * sizeof(double), aero::gpu::Copy::ToHost));
}
double* PartitionGpu::expand_owned(const double* compact) const {
    partition_detail::transfer({grid_.geometry.components(), krylov_input_, nullptr}, grid_.owned,
                               grid_.owned_count, compact, nullptr);
    return krylov_input_;
}
void PartitionGpu::download(std::vector<double>& state) const {
    if (!state.empty())
        check(aero::gpu::copy(state.data(), grid_.state, state.size() * sizeof(double), aero::gpu::Copy::ToHost));
}
PartitionGpu::Peer PartitionGpu::prepare(const std::vector<MeshIndex>& out, const std::vector<MeshIndex>& in,
                                         unsigned stride) {
    Peer peer;
    peer.out = storage_.upload(out);
    peer.in = storage_.upload(in);
    peer.buffer = storage_.allocate<double>(std::max(out.size(), in.size()) * stride);
    return peer;
}
void PartitionGpu::exchange(const Peer& peer, DoubleExchange::Buffer records, double* mapped, partition_detail::RecordField field,
                            bool pack) {
    if (!records.count)
        return;
    const auto count = records.count / field.stride();
    const auto bytes = records.count * sizeof(double);
    // Для короткого сообщения ядро упаковки может напрямую обратиться к закреплённой памяти.
    if (mapped && bytes <= 128) {
        partition_detail::transfer(field, pack ? peer.out : peer.in, count, pack ? nullptr : mapped, pack ? mapped : nullptr);
        return;
    }
    if (pack) {
        partition_detail::transfer(field, peer.out, count, nullptr, peer.buffer);
        check(aero::gpu::copy_async(records.data, peer.buffer, bytes, aero::gpu::Copy::ToHost, nullptr));
    } else {
        check(aero::gpu::copy_async(peer.buffer, records.data, bytes, aero::gpu::Copy::ToDevice, nullptr));
        partition_detail::transfer(field, peer.in, count, peer.buffer, nullptr);
    }
}
}
