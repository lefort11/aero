#ifndef AERO_PARTITION_GPU_H
#define AERO_PARTITION_GPU_H
#include "partition_kernels.h"
#include "../io/diagnostic_fields.h"
#include "../residual/flux_cfl.h"
#include "mpi_balance.h"
#include "mpi_stencil.h"
#include "../parallel/device_memory.h"
#include "../parallel/pinned_buffer.h"
#include "../parallel/processes.h"
#include <limits>

#ifdef AERO_HAS_VISCOSITY
#include "../viscosity/aes.h"
#endif

namespace aero {
void select_partition_gpu(const Processes& world, const std::vector<int>& devices);
void query_gpu_memory(GpuMemoryFit& fit, const std::vector<double>& limits, const Processes& world);

class PartitionGpu {
    // Одинаковый порядок размещения используется при подсчёте размера и при загрузке.
    // Все массивы устройства помещаются в одну выделенную область.
    struct Storage {
        // Базовый адрес одной области GPU. До резервирования указатель пуст и вычисляется только размер.
        unsigned char* base = nullptr;
        std::size_t used = 0, capacity = 0;
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
            if (!base)
                return nullptr;
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
    void check_kernel(const char* phase);
    static double measure_flux_rate(ResidualOptions options, std::size_t faces, MeshIndex budget);

public:
    PartitionGpu() = default;
#ifdef AERO_HAS_VISCOSITY
    viscosity::Operator viscosity_view() const { return viscous_grid_; }
    void set_viscosity(const viscosity::Aes* aes) { aes_ = aes; }
#endif
    static std::vector<double> measure_weights(const Processes& world, ResidualOptions options,
                                               MeshIndex budget);
    void layout(residual_detail::Geometry geometry, const PartitionStencil& stencil,
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
    double* layout_node_scalars(MeshIndex count) { return storage_.allocate<double>(count); }
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
