#include "../parallel/backend.h"
#include "../parallel/device_check.h"
#include "partition_gpu.h"
#include "../residual/boundary_flux_failure.h"
#include "../reconstruction/device_stencil.h"
#include "../parallel/upload_compact.h"
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
namespace edge_tile = residual_detail::edge_tile;
namespace {
// Преобразует ошибку GPU в исключение с контекстом MPI-подобласти.
void gpu_check(aero::gpu::Status status) {
    gpu::checked(status, "MPI GPU: ");
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
    // Размечает очередной набор массивов в общем буфере GPU.
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
    // Запускает измеряемое вычислительное ядро на выбранном GPU.
    const auto launch = [&] { residual_detail::gpu_face_fluxes(batch, sample.grid_.bad, nullptr); };
    launch();
    gpu_check(aero::gpu::wait(nullptr));
    std::array<double, 5> seconds;
    constexpr unsigned repeats = 8;
    for (double& elapsed : seconds) {
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < repeats; ++i)
            launch();
        gpu_check(aero::gpu::wait(nullptr));
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    sample.check_kernel("calibration");
    std::sort(seconds.begin(), seconds.end());
    const double rate = static_cast<double>(count * repeats) / seconds[seconds.size() / 2];
    if (!(rate > 0 && std::isfinite(rate)))
        throw std::runtime_error("GPU calibration: invalid elapsed time");
    return rate;
}
// Измеряет скорость вычисления потоков для балансировки работы GPU.
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
// Назначает процессу физическую карту с учётом рангов на одном узле.
void select_partition_gpu(const Processes& world, const std::vector<int>& devices) {
    const auto node = partition_node_resources(world);
    std::string physical, description;
    world.local([&] {
        int count = 0;
        gpu_check(aero::gpu::count(&count));
        // Планировщик может выдать каждому процессу отдельную карту с одним и тем же локальным номером 0.
        const int slot = count == 1 && devices.size() == 1 ? 0 : node.local_rank;
        if (static_cast<std::size_t>(slot) >= devices.size())
            throw std::invalid_argument("cuda.txt: need one GPU per node-local MPI rank");
        const int device = devices[slot];
        if (device >= count)
            throw std::invalid_argument(std::string("Unavailable ") + gpu_backend_label + " device " + std::to_string(device));
        gpu_check(aero::gpu::select(device));
        char bus[32]{};
        gpu_check(aero::gpu::bus_id(bus, sizeof(bus), device));
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
            std::cout << "MPI " << gpu_backend_label << " placement: " << line << '\n';
    });
}
// Проверяет статус операции GPU и преобразует ошибку в исключение.
void PartitionGpu::check(aero::gpu::Status status) {
    gpu_check(status);
}
// Ожидает завершения операций GPU перед чтением результатов на CPU.
void PartitionGpu::synchronize() {
    check(aero::gpu::wait(nullptr));
}
// Измеряет доступную память GPU и применяет ограничения пользователя.
void query_gpu_memory(GpuMemoryFit& fit, const std::vector<double>& limits, const Processes& world) {
    std::size_t available = 0, total = 0;
    gpu_check(aero::gpu::memory_info(&available, &total));
    fit.free_bytes = available;
    fit.budget = available - available / 20;
    // После точного подсчёта CSR и рабочих массивов оставляем запас для драйвера.
    constexpr std::size_t compact_headroom = 128u * 1024u * 1024u;
    fit.compact_budget = available - std::min(available / 20, compact_headroom);
    if (!limits.empty()) {
        if (limits.size() != 1 && limits.size() != static_cast<std::size_t>(world.size))
            throw std::invalid_argument("mpi_gpu_memory_mib: expected one value or one limit per MPI rank");
        const long double bytes =
            static_cast<long double>(limits[limits.size() == 1 ? 0 : world.rank]) * 1048576;
        if (bytes < fit.budget)
            fit.budget = static_cast<MeshIndex>(bytes);
        if (bytes < fit.compact_budget)
            fit.compact_budget = static_cast<MeshIndex>(bytes);
    }
}
// Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
PartitionGpu::Storage::~Storage() {
    if (base)
        aero::gpu::release(base);
}
// Размечает массивы геометрии, полей и рабочих данных в едином буфере GPU.
void PartitionGpu::layout(residual_detail::Geometry g,
                          const std::vector<MeshIndex>& computed, const std::vector<MeshIndex>& owned,
                          const double* volumes, std::size_t state_nodes, bool compact_buffer,
                          bool whole_mesh) {
    const bool store_face_batch = g.packed.tiles || g.options.variables == ResidualVariables::CharacteristicFlux;
    const auto cache_start=storage_.used;
#ifdef AERO_HAS_TURBULENCE
    grid_.filtered_gas=nullptr;
#endif
    if (g.packed.tiles) {
        const auto count = edge_tile::count(g.edges);
        // Один буфер обслуживает все стадии. Большую подобласть проходим частями:
        // отложенные веса не требуют дополнительной памяти пропорционально всей сетке.
        if (g.packed.order == 5 && g.options.variables == ResidualVariables::Characteristic &&
            g.stencil.mp5.weno && g.options.mp6_hybrid == 0) {
            using Tile = residual_detail::MpEdgeTile;
            const auto faces = std::min<MeshIndex>(computed.size(), MeshIndex(Tile::batch_limit));
            g.packed.mp_work = storage_.allocate<Tile>(edge_tile::count(faces));
        }
        if (g.packed.order == 5)
            g.packed.tiles = storage_.allocate(count, static_cast<const residual_detail::WenoEdgeTile<5>*>(g.packed.tiles));
        else
            g.packed.tiles = storage_.allocate(count, static_cast<const residual_detail::WenoEdgeTile<3>*>(g.packed.tiles));
        g.packed.remaining = storage_.allocate(g.packed.count, g.packed.remaining);
    }
    g.packed.gas = storage_.allocate<EulerState>(g.packed.state_nodes);
    g.packed.normals = storage_.allocate<residual_detail::FaceNormal>(g.packed.state_nodes ? g.edges : 0);
    cache_bytes_=storage_.used-cache_start;
    // Граничные условия обращаются к исходному полю лишь на физических площадках.
    // Сохраняем отображение на CPU до замены адреса геометрии адресом устройства.
    const auto boundary_count = g.faces - g.edges;
    if (storage_.base) {
        boundary_nodes_.resize(boundary_count);
        parallel_for(boundary_count, [&](MeshIndex i) { boundary_nodes_[i] = g.surfaces[g.edges+i].begin; });
    }
    g.initial_on_boundary = true;
    // Эти копии определяются геометрией шаблона, а не свободной памятью конкретной карты.
    if (!storage_.base)
        compact_slopes_.inspect_edges(g.stencil,g.surfaces);
    g.surfaces = upload_records<residual_detail::Surface32>(storage_,g.faces,g.surfaces.wide);
    g.incidence = upload_records<residual_detail::OrientedFace32>(storage_,g.offsets[g.nodes],g.incidence.wide);
    g.offsets = upload_indices(storage_,g.nodes+1,g.offsets.wide);
    g.patches = upload_boundary_patches(storage_, g.patches);
    if (!storage_.base) compact_slopes_.prepare(g.stencil);
    const auto exceptions=compact_slopes_.bind(g.stencil);
    compact_slopes_.use_edge_data(g.stencil,g.surfaces);
    const auto reconstruction_start=storage_.used;
    g.stencil = upload_reconstruction_stencil(storage_,g.stencil,exceptions,
        g.options.variables==ResidualVariables::CharacteristicFlux);
    reconstruction_bytes_=storage_.used-reconstruction_start;
    if (storage_.base) compact_slopes_=CompactSlopes{};
#ifdef AERO_HAS_VISCOSITY
    if (aes_) {
        viscosity::FlowScratch fields{};
        if (aes_->uses_implicit_coordinates()) {
            const FluxWorkspace layout(g.faces,state_nodes,aes_->view().row_count);
            grid_.flux = reinterpret_cast<EulerFluxResult*>(storage_.allocate<unsigned char>(layout.bytes));
            fields = layout.viscosity_fields(grid_.flux);
        }
        viscous_grid_ = aes_->upload(storage_,state_nodes,aes_->uses_implicit_coordinates() ? &fields : nullptr);
        g.options.diffusion = viscous_grid_.rows;
        g.options.wall = storage_.upload(aes_->wall);
#ifdef AERO_HAS_TURBULENCE
        viscous_grid_.no_slip=g.options.wall;
        // На одном процессе обновляются все узлы. Соседние значения MPI могут
        // измениться при обмене, поэтому там материал готовится после обмена.
        if (whole_mesh && g.packed.state_nodes && g.packed.state_nodes==owned.size())
            grid_.filtered_gas=storage_.allocate(1,&viscous_grid_);
#endif
    }
#endif
    grid_.geometry = g;
    grid_.volumes = storage_.allocate(g.nodes, volumes);
    grid_.computed = whole_mesh ? nullptr : storage_.upload(computed);
    grid_.owned = whole_mesh ? nullptr : storage_.upload(owned);
    grid_.computed_count = computed.size();
    grid_.owned_count = owned.size();
#ifdef AERO_HAS_VISCOSITY
    if (!aes_ || !aes_->uses_implicit_coordinates())
#endif
        grid_.flux = storage_.allocate<EulerFluxResult>(g.faces);
    const unsigned width = g.dim + 2;
    if (state_nodes > std::numeric_limits<std::size_t>::max() / width)
        throw std::length_error("GPU state too large");
    grid_.initial = storage_.allocate<double>(boundary_count * width);
    grid_.state = storage_.allocate<double>(state_nodes * width);
    if (g.options.retry_nonphysical) {
        accepted_state_ = storage_.allocate<double>(state_nodes * width);
        grid_.failed_count = storage_.allocate<partition_detail::Index>(1);
        grid_.failed_nodes = storage_.allocate<partition_detail::Index>(owned.size());
    }
    if (compact_buffer)
        // На целой сетке SSPRK и откат сохраняют одно U_n до конца попытки.
        // При MPI компактные собственные строки и полный буфер отката различаются.
        grid_.original = whole_mesh && g.options.retry_nonphysical ? accepted_state_ :
                         storage_.allocate<double>(owned.size() * (g.dim + 2));
    grid_.bad = storage_.allocate<partition_detail::Index>(1);
    // Выделяем запись в том же порядке при оценке памяти и при заполнении. После
    // резервирования поля обновляются на месте, поэтому их адреса сохраняются между стадиями.
    if (store_face_batch) {
        const auto batch = face_batch();
        device_faces_ = storage_.allocate(1, &batch);
    }
}
// Пробует зарезервировать рассчитанный объём и передаёт фактическую потребность планировщику.
bool PartitionGpu::reserve(GpuMemoryFit& fit, MeshIndex implicit_bytes, MeshIndex coarse_bytes) {
    fit.array_count = storage_.array_count;
    fit.layout_bytes = storage_.used;
    fit.reconstruction_bytes=reconstruction_bytes_;
    fit.cache_bytes=cache_bytes_;
    fit.implicit_bytes=implicit_bytes;
    fit.coarse_bytes=coarse_bytes;
    fit.required = partition_bytes(partition_bytes(storage_.used, 1, implicit_bytes),1,coarse_bytes);
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
// Переносит на GPU начальные значения для граничных условий.
void PartitionGpu::upload_boundary_initial(const std::vector<double>& initial) {
    if (boundary_nodes_.empty()) return;
    const auto width = grid_.geometry.components();
    std::vector<double> values(boundary_nodes_.size() * width);
    // Соседние площадки одной метки могут иметь разные начальные условия.
    // Копируем все компоненты без усреднения, в том же порядке, что и площадки.
    parallel_for(boundary_nodes_.size(), [&](MeshIndex i) {
        std::copy_n(initial.data() + width*boundary_nodes_[i], width, values.data() + width*i);
    });
    check(gpu::copy(grid_.initial, values.data(), values.size()*sizeof(double), gpu::Copy::ToDevice));
}
// Загружает текущее и начальное поля в размеченную память GPU.
void PartitionGpu::upload_state(const std::vector<double>& state, const std::vector<double>& initial) {
    gas_current_ = false;
    if (storage_.used != storage_.capacity)
        throw std::logic_error("GPU layout changed after reservation");
    if (grid_.geometry.packed.normals)
        residual_detail::gpu_prepare_face_normals(grid_.geometry);
#ifdef AERO_HAS_VISCOSITY
    if (viscous_grid_.tetra.cached_cells)
        device_work::check(device_work::aero_device_prepare_tetra_volumes(viscous_grid_));
#endif
    if (!state.empty()) {
        check(aero::gpu::copy(grid_.state, state.data(), state.size() * sizeof(double), aero::gpu::Copy::ToDevice));
        upload_boundary_initial(initial);
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
// Проверяет ошибку ядра и сообщает фазу и отказавший узел.
void PartitionGpu::check_kernel(const char* phase, bool checking_flux) {
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
        if (checking_flux && bad >= grid_.geometry.edges && bad < grid_.geometry.faces)
            throw BoundaryFluxFailure(bad);
        throw std::domain_error(std::string("MPI GPU ") + phase + " at local index " + std::to_string(bad));
    }
}
// Обновляет геометрию площадок в памяти GPU.
void PartitionGpu::update_surfaces(const std::vector<residual_detail::Surface>& surfaces) {
    if (surfaces.empty()) return;
    const auto& target=grid_.geometry.surfaces;
    if (target.narrow) {
        std::vector<residual_detail::Surface32> packed(surfaces.size());
        parallel_for(surfaces.size(),[&](std::size_t i) {
            if (!residual_detail::Surface32::fits(surfaces[i]))
                throw std::logic_error("Updated surface exceeds compact index range");
            packed[i]=residual_detail::Surface32::pack(surfaces[i]);
        });
        check(gpu::copy(const_cast<residual_detail::Surface32*>(target.narrow),packed.data(),
                       packed.size()*sizeof(packed[0]),gpu::Copy::ToDevice));
    } else check(gpu::copy(const_cast<residual_detail::Surface*>(target.wide),
                          surfaces.data(),surfaces.size()*sizeof(surfaces[0]),gpu::Copy::ToDevice));
}
// Собирает представление данных для обработки площадок на GPU.
residual_detail::FaceBatch PartitionGpu::face_batch() const {
    return {grid_.geometry, grid_.computed, grid_.computed_count, grid_.state,
            grid_.initial, grid_.flux, nullptr, nullptr, gas_current_};
}
// Обновляет параметр ограничителя MP и зависимую запись ядра.
void PartitionGpu::set_alpha(double alpha) {
    auto& mp = grid_.geometry.stencil.mp5;
    if ((!mp.enabled && !mp.weno) || mp.alpha == alpha) return;
    mp.alpha = alpha;
    update_face_batch();
}
// Обновляет коэффициенты диссипации и гибридной поправки потоков.
void PartitionGpu::set_flux_options(const ResidualOptions& options) {
    grid_.geometry.options.upw_coeff = options.upw_coeff;
    grid_.geometry.options.mp6_hybrid = options.mp6_hybrid;
    update_face_batch();
}
// Синхронизирует запись пакетного ядра с текущими параметрами.
void PartitionGpu::update_face_batch() {
    // Упакованные ядра MP читают настройки из постоянной записи на карте.
    if (device_faces_) {
        const auto batch = face_batch();
        check(gpu::copy(device_faces_, &batch, sizeof(batch), gpu::Copy::ToDevice));
    }
}
// Вычисляет потоки площадок с учётом режима первого порядка.
void PartitionGpu::fluxes(const residual_detail::FaceBatch& batch, bool first_order_start) {
    residual_detail::gpu_face_fluxes(batch, grid_.bad, nullptr, first_order_start ? nullptr : device_faces_);
}
#ifdef AERO_HAS_VISCOSITY
void PartitionGpu::viscous_fluxes(viscosity::Evaluation phase) {
#ifdef AERO_HAS_TURBULENCE
    if (phase == viscosity::Evaluation::Full && stage_material_current())
        phase = viscosity::Evaluation::CurrentMaterial;
#endif
    device_work::check(device_work::aero_device_viscosity(viscous_grid_,grid_.state,grid_.geometry.packed.gas,phase));
}
#endif
// Выполняет явную стадию собственных узлов и проверяет допустимость результата.
void PartitionGpu::update(explicit_detail::Stage stage, double dt, bool save_original) {
    if (grid_.failed_count) reset_failed_count();
    partition_detail::update(grid_, stage, dt, save_original);
    check_kernel("invalid stage state");
    gas_current_ = !grid_.owned;
}
// Сохраняет поле перед попыткой шага или восстанавливает его после отказа.
void PartitionGpu::copy_backup(std::size_t count, bool restore) {
    // Индексы соседей входят в снимок: после отката можно сразу пересчитать потоки.
    if (count) check(gpu::copy_async(restore ? grid_.state : accepted_state_,
        restore ? accepted_state_ : grid_.state, count*sizeof(double), gpu::Copy::OnDevice, nullptr));
    if (restore) gas_current_ = false;
}
// Обнуляет счётчик отказавших узлов перед проверкой состояния.
void PartitionGpu::reset_failed_count() {
    if (error_index_.device()) error_index_.host()[1] = 0;
    else check(gpu::fill_async(grid_.failed_count, 0, sizeof(*grid_.failed_count), nullptr));
}
// Собирает номера узлов с недопустимым газодинамическим состоянием.
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

// Собирает только собственные узлы раздела для отчёта или многостадийного шага.
void PartitionGpu::download_owned(std::vector<double>& compact) const {
    if (compact.empty())
        return;
    // На одной карте собственные узлы уже записаны подряд. Перестановка требуется
    // только при MPI-разбиении, когда внутри поля встречаются копии соседних узлов.
    const double* source = grid_.state;
    if (grid_.owned) {
        partition_detail::transfer({grid_.geometry.components(), grid_.state, nullptr}, grid_.owned,
                               grid_.owned_count, nullptr, grid_.original);
        source = grid_.original;
    }
    check(gpu::copy(compact.data(), source, compact.size()*sizeof(double), gpu::Copy::ToHost));
}
// Раскладывает компактный вектор собственных узлов в полную нумерацию подобласти.
double* PartitionGpu::expand_owned(const double* compact) const {
    partition_detail::transfer({grid_.geometry.components(), krylov_input_, nullptr}, grid_.owned,
                               grid_.owned_count, compact, nullptr);
    return krylov_input_;
}
// Копирует текущее поле с GPU в CPU-вектор.
void PartitionGpu::download(std::vector<double>& state) const {
    if (!state.empty())
        check(aero::gpu::copy(state.data(), grid_.state, state.size() * sizeof(double), aero::gpu::Copy::ToHost));
}
// Загружает индексы отправки и приёма и готовит буферы одного соседа.
PartitionGpu::Peer PartitionGpu::prepare(const std::vector<MeshIndex>& out, const std::vector<MeshIndex>& in,
                                         unsigned stride) {
    Peer peer;
    peer.out = storage_.upload(out);
    peer.in = storage_.upload(in);
    peer.buffer = storage_.allocate<double>(std::max(out.size(), in.size()) * stride);
    return peer;
}
// Упаковывает или распаковывает записи поля для обмена с соседом.
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
