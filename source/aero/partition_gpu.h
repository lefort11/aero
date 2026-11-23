#ifndef AERO_PARTITION_GPU_H
#define AERO_PARTITION_GPU_H
#include "partition_kernels.h"
#include "../io/diagnostic_fields.h"
#include "../residual/flux_cfl.h"
#include "mpi_balance.h"
#include "mpi_stencil.h"
#include "compact_slopes.h"
#include "../parallel/device_memory.h"
#include "../parallel/pinned_buffer.h"
#include "../parallel/processes.h"
#include <limits>

#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/aes.h"
#endif

namespace aero {
#ifdef AERO_HAS_VISCOSITY
// Потоки, свойства газа и вязкие строки живут одновременно при сборке невязки.
// На время решения СЛАУ все три массива свободны. Непрерывное размещение
// позволяет отдать BiCGSTAB также остатки между целыми векторами.
struct FluxWorkspace {
    MeshIndex flux_bytes, gas_end, bytes;
    FluxWorkspace(MeshIndex faces, MeshIndex gas_nodes, MeshIndex rows)
        : flux_bytes(partition_bytes(0,faces,sizeof(EulerFluxResult))),
          gas_end(partition_bytes(flux_bytes,gas_nodes,sizeof(viscosity::Gas))),
          bytes(partition_bytes(gas_end,rows,sizeof(viscosity::Row))) {}

    viscosity::FlowScratch viscosity_fields(EulerFluxResult* flux) const {
        if (!flux) return {nullptr,nullptr}; // Первый проход только подсчитывает занимаемую память.
        auto* base = reinterpret_cast<unsigned char*>(flux);
        static_assert(sizeof(EulerFluxResult)%alignof(viscosity::Gas)==0 &&
                      sizeof(viscosity::Gas)%alignof(viscosity::Row)==0, "Invalid workspace alignment");
        return {reinterpret_cast<viscosity::Gas*>(base+flux_bytes),
                reinterpret_cast<viscosity::Row*>(base+gas_end)};
    }
};
#endif
void select_partition_gpu(const Processes& world, const std::vector<int>& devices);
void query_gpu_memory(GpuMemoryFit& fit, const std::vector<double>& limits, const Processes& world);

class PartitionGpu {
    // Одинаковый порядок размещения используется при подсчёте размера и при загрузке.
    // Все массивы устройства помещаются в одну выделенную область.
    struct Storage {
        // Базовый адрес одной области GPU. До резервирования указатель пуст и вычисляется только размер.
        unsigned char* base = nullptr;
        std::size_t used = 0, capacity = 0;
        std::size_t array_count = 0; // Число непустых массивов в общей области GPU.
        ~Storage();
        template <class T> T* allocate(std::size_t n, const T* host = nullptr) {
            if (!n)
                return nullptr;
            const auto max = std::numeric_limits<std::size_t>::max();
            if (used > max - 255)
                throw std::length_error("GPU layout too large");
            // Округляем начало каждого массива вверх до границы в 256 байт.
            const auto offset = (used + 255) & ~std::size_t(255);
            if (n > (max - offset) / sizeof(T))
                throw std::length_error("GPU allocation too large");
            used = offset + n * sizeof(T);
            // Первый проход тем же кодом даёт точное требование к памяти без выделения отдельных массивов.
            if (!base) {
                ++array_count;
                return nullptr;
            }
            if (used > capacity)
                throw std::logic_error("GPU layout exceeded reserved storage");
            T* data = reinterpret_cast<T*>(base + offset);
            if (host)
                check(aero::gpu::copy(data, host, n * sizeof(T), aero::gpu::Copy::ToDevice));
            return data;
        }
        template <class T> T* upload(const std::vector<T>& host) {
            return allocate(host.size(), host.data());
        }
    } storage_;
    CompactSlopes compact_slopes_; // Живёт на CPU между подсчётом памяти и однократной загрузкой.
    std::vector<MeshIndex> boundary_nodes_; // По площадке находим исходное значение для граничного условия.
    void upload_boundary_initial(const std::vector<double>& initial);
#ifdef AERO_HAS_VISCOSITY
    const viscosity::Aes* aes_ = nullptr;
    viscosity::Operator viscous_grid_{};
#endif
    partition_detail::Grid grid_{};
    // Та же геометрия и адреса полей нужны Kokkos на каждой стадии. Запись живёт в storage_.
    residual_detail::FaceBatch* device_faces_ = nullptr;
    double* accepted_state_ = nullptr; // Резервная копия слоя остаётся на карте.
    double* krylov_input_ = nullptr; // Полный локальный вектор с копиями соседних узлов для A*x.
    bool gas_current_ = false; // Для цельной сетки обновление узлов готовит газ одновременно с полем.
    PinnedBuffer<partition_detail::Index> error_index_; // Индекс сбоя, доступный CPU после окончания стадии.
    static void check(aero::gpu::Status status);
    residual_detail::FaceBatch face_batch() const; // Адреса полей и текущие настройки потока.
    void update_face_batch(); // Обновляет параметры, которые читают упакованные ядра.
    void reset_failed_count(); // Счётчик доступен на CPU после синхронизации стадии.
    void check_kernel(const char* phase, bool checking_flux = false);
    static double measure_flux_rate(ResidualOptions options, std::size_t faces, MeshIndex budget);

public:
    PartitionGpu() = default;
#ifdef AERO_HAS_VISCOSITY
    const EulerState* current_gas() const { return gas_current_ ? grid_.geometry.packed.gas : nullptr; }
#ifdef AERO_HAS_TURBULENCE
    bool has_stage_material() const { return grid_.filtered_gas!=nullptr; }
    bool stage_material_current() const { return gas_current_ && has_stage_material(); }
#endif
    viscosity::Operator viscosity_view() const { return viscous_grid_; }
    void set_viscosity(const viscosity::Aes* aes) { aes_ = aes; }
#endif
    static std::vector<double> measure_weights(const Processes& world, ResidualOptions options,
                                               MeshIndex budget);
    void layout(residual_detail::Geometry geometry,
                const std::vector<MeshIndex>& computed, const std::vector<MeshIndex>& owned,
                const double* volumes, std::size_t state_nodes, bool compact_buffer, bool whole_mesh);
    // Пробует зарезервировать рассчитанный объём и передаёт фактическую потребность планировщику.
    bool reserve(GpuMemoryFit& fit, MeshIndex implicit_bytes = 0);
    // Копирует на карту принятое поле и исходное состояние, необходимое граничным условиям.
    void upload_state(const std::vector<double>& state, const std::vector<double>& initial);
    // Поля в памяти карты. Потоки вычисляются обычным оператором на этих же адресах.
    const partition_detail::Grid& active_grid() const { return grid_; }
    static void synchronize();
    PartitionGpu(const PartitionGpu&) = delete;
    PartitionGpu& operator=(const PartitionGpu&) = delete;
    void failed_nodes(std::vector<MeshIndex>& failed, bool implicit);
    void copy_backup(std::size_t count, bool restore);
    void set_flux_options(const ResidualOptions& options);
    void set_alpha(double alpha);
    void update_surfaces(const std::vector<residual_detail::Surface>& surfaces);
    void fluxes(bool first_order_start,bool defer_viscous_rows=false);
    void update(explicit_detail::Stage stage, double dt, bool save_original);
    // Собирает только собственные узлы раздела для отчёта или многостадийного шага.
    void download_owned(std::vector<double>& compact) const;
    void download(std::vector<double>& state) const;
    double* layout_node_scalars(MeshIndex count, const double* initial = nullptr) {
        return storage_.allocate<double>(count, initial);
    }
    template<class T> T* layout_guard_array(const std::vector<T>& values) { return storage_.upload(values); }
    diagnostic::Moments* layout_moments(MeshIndex count) { return storage_.allocate<diagnostic::Moments>(count); }
    void layout_linear_input(MeshIndex count) { krylov_input_ = storage_.allocate<double>(count); }
    double* expand_owned(const double* compact) const;

    struct Peer {
        MeshIndex *out = nullptr, *in = nullptr;
        double* buffer = nullptr;
    };
    Peer prepare(const std::vector<MeshIndex>& out, const std::vector<MeshIndex>& in, unsigned stride);
    void exchange(const Peer& peer, DoubleExchange::Buffer records, double* mapped, partition_detail::RecordField field, bool pack);
};
}
#endif
