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
    // Размер потоков, конец газового участка и полный размер общей рабочей области.
    MeshIndex flux_bytes, gas_end, bytes;
    // Вычисляет смещения участков общего буфера потоков, газа и вязких строк.
    FluxWorkspace(MeshIndex faces, MeshIndex gas_nodes, MeshIndex rows)
        : flux_bytes(partition_bytes(0,faces,sizeof(EulerFluxResult))),
          gas_end(partition_bytes(flux_bytes,gas_nodes,sizeof(viscosity::Gas))),
          bytes(partition_bytes(gas_end,rows,sizeof(viscosity::Row))) {}

    // Возвращает участки общего буфера для газа и вязких градиентов.
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
// Назначает процессу физическую карту с учётом рангов на одном узле.
void select_partition_gpu(const Processes& world, const std::vector<int>& devices);
// Измеряет доступную память GPU и применяет ограничения пользователя.
void query_gpu_memory(GpuMemoryFit& fit, const std::vector<double>& limits, const Processes& world);

// Управляет полями, рабочей памятью и вычислительными ядрами подобласти на GPU.
class PartitionGpu {
    // Одинаковый порядок размещения используется при подсчёте размера и при загрузке.
    // Все массивы устройства помещаются в одну выделенную область.
    struct Storage {
        // Базовый адрес одной области GPU. До резервирования указатель пуст и вычисляется только размер.
        unsigned char* base = nullptr;
        // Занятая и зарезервированная части хранилища.
        std::size_t used = 0, capacity = 0;
        std::size_t array_count = 0; // Число непустых массивов в общей области GPU.
        // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
        ~Storage();
        // Резервирует выровненный участок общего GPU-буфера с проверкой переполнения.
        template <class T> T* allocate(std::size_t n, const T* host = nullptr) {
            if (!n)
                return nullptr;
            const auto max = std::numeric_limits<std::size_t>::max();
            // Сначала проверяем запас для округления адреса вверх, чтобы выравнивание само не переполнило размер.
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
        // Размещает CPU-вектор в общем GPU-буфере и возвращает его адрес.
        template <class T> T* upload(const std::vector<T>& host) {
            return allocate(host.size(), host.data());
        }
    // Владеет выделениями текущего GPU-раздела.
    } storage_;
    CompactSlopes compact_slopes_; // Живёт на CPU между подсчётом памяти и однократной загрузкой.
    MeshIndex reconstruction_bytes_=0, cache_bytes_=0; // Части той же раскладки для отчёта о памяти.
    std::vector<MeshIndex> boundary_nodes_; // По площадке находим исходное значение для граничного условия.
    // Переносит на GPU начальные значения для граничных условий.
    void upload_boundary_initial(const std::vector<double>& initial);
#ifdef AERO_HAS_VISCOSITY
    // Исходная геометрия вязкого оператора.
    const viscosity::Aes* aes_ = nullptr;
    // Представление вязкого оператора с адресами GPU.
    viscosity::Operator viscous_grid_{};
#endif
    // Адреса сетки и полей, передаваемые ядрам.
    partition_detail::Grid grid_{};
    // Та же геометрия и адреса полей нужны Kokkos на каждой стадии. Запись живёт в storage_.
    residual_detail::FaceBatch* device_faces_ = nullptr;
    double* accepted_state_ = nullptr; // Резервная копия слоя остаётся на карте.
    double* krylov_input_ = nullptr; // Полный локальный вектор с копиями соседних узлов для A*x.
    bool gas_current_ = false; // Для цельной сетки обновление узлов готовит газ одновременно с полем.
    PinnedBuffer<partition_detail::Index> error_index_; // Индекс сбоя, доступный CPU после окончания стадии.
    // Проверяет статус операции GPU и преобразует ошибку в исключение.
    static void check(aero::gpu::Status status);
    // Собирает представление данных для обработки площадок на GPU.
    residual_detail::FaceBatch face_batch() const; // Адреса полей и текущие настройки потока.
    // Синхронизирует запись пакетного ядра с текущими параметрами.
    void update_face_batch(); // Обновляет параметры, которые читают упакованные ядра.
    // Обнуляет счётчик отказавших узлов перед проверкой состояния.
    void reset_failed_count(); // Счётчик доступен на CPU после синхронизации стадии.
    // Проверяет ошибку ядра и сообщает фазу и отказавший узел.
    void check_kernel(const char* phase, bool checking_flux = false);
    // Замеряем рабочее ядро потока на одинаковых гранях с узловыми состояниями на всех GPU.
    // Геометрия и данные уже на устройстве, поэтому время обменов в замер не входит.
    static double measure_flux_rate(ResidualOptions options, std::size_t faces, MeshIndex budget);

public:
    // Инициализирует объект стандартными значениями его полей.
    PartitionGpu() = default;
    // Начальное поле неявного шага может использовать готовую копию для отката.
    double* step_start_buffer() const { return accepted_state_; }
#ifdef AERO_HAS_VISCOSITY
    // Сообщает об актуальности кэшированных газовых параметров.
    const EulerState* current_gas() const { return gas_current_ ? grid_.geometry.packed.gas : nullptr; }
#ifdef AERO_HAS_TURBULENCE
    // Проверяет наличие материальных коэффициентов стадии.
    bool has_stage_material() const { return grid_.filtered_gas!=nullptr; }
    // Помечает материальные коэффициенты стадии как актуальные.
    bool stage_material_current() const { return gas_current_ && has_stage_material(); }
#endif
    // Возвращает представление вязкого оператора для ядер GPU.
    viscosity::Operator viscosity_view() const { return viscous_grid_; }
    // Подключает вязкий оператор к геометрии GPU.
    void set_viscosity(const viscosity::Aes* aes) { aes_ = aes; }
#endif
    // Измеряет скорость вычисления потоков для балансировки работы GPU.
    static std::vector<double> measure_weights(const Processes& world, ResidualOptions options,
                                               MeshIndex budget);
    // Размечает массивы геометрии, полей и рабочих данных в едином буфере GPU.
    void layout(residual_detail::Geometry geometry,
                const std::vector<MeshIndex>& computed, const std::vector<MeshIndex>& owned,
                const double* volumes, std::size_t state_nodes, bool compact_buffer, bool whole_mesh);
    // Пробует зарезервировать рассчитанный объём и передаёт фактическую потребность планировщику.
    bool reserve(GpuMemoryFit& fit, MeshIndex implicit_bytes = 0, MeshIndex coarse_bytes = 0);
    // Копирует на карту принятое поле и исходное состояние, необходимое граничным условиям.
    void upload_state(const std::vector<double>& state, const std::vector<double>& initial);
    // Поля в памяти карты. Потоки вычисляются обычным оператором на этих же адресах.
    const partition_detail::Grid& active_grid() const { return grid_; }
    // Ожидает завершения операций GPU перед чтением результатов на CPU.
    static void synchronize();
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    PartitionGpu(const PartitionGpu&) = delete;
    // Запрещает копирование, чтобы ресурсы объекта сохраняли единственного владельца.
    PartitionGpu& operator=(const PartitionGpu&) = delete;
    // Собирает номера узлов с недопустимым газодинамическим состоянием.
    void failed_nodes(std::vector<MeshIndex>& failed, bool implicit);
    // Сохраняет поле перед попыткой шага или восстанавливает его после отказа.
    void copy_backup(std::size_t count, bool restore);
    // Обновляет коэффициенты диссипации и гибридной поправки потоков.
    void set_flux_options(const ResidualOptions& options);
    // Обновляет параметр ограничителя MP и зависимую запись ядра.
    void set_alpha(double alpha);
    // Обновляет геометрию площадок в памяти GPU.
    void update_surfaces(const std::vector<residual_detail::Surface>& surfaces);
    // Вычисляет потоки площадок с учётом режима первого порядка.
    void fluxes(bool first_order_start,bool defer_viscous_rows=false);
    // Выполняет явную стадию собственных узлов и проверяет допустимость результата.
    void update(explicit_detail::Stage stage, double dt, bool save_original);
    // Собирает только собственные узлы раздела для отчёта или многостадийного шага.
    void download_owned(std::vector<double>& compact) const;
    // Копирует текущее поле с GPU в CPU-вектор.
    void download(std::vector<double>& state) const;
    // Выделяет участок скалярных узловых величин в общей памяти GPU.
    double* layout_node_scalars(MeshIndex count, const double* initial = nullptr) {
        return storage_.allocate<double>(count, initial);
    }
    // Выделяет участок временных моментов узлов в общей памяти GPU.
    diagnostic::Moments* layout_moments(MeshIndex count) { return storage_.allocate<diagnostic::Moments>(count); }
    // Выделяет полноразмерный входной вектор линейного решателя на GPU.
    void layout_linear_input(MeshIndex count) { krylov_input_ = storage_.allocate<double>(count); }
    // Раскладывает компактный вектор собственных узлов в полную нумерацию подобласти.
    double* expand_owned(const double* compact) const;

    // Индексы и рабочие буферы обмена с соседним процессом.
    struct Peer {
        // Индексы упаковки отправляемых и размещения полученных записей.
        MeshIndex *out = nullptr, *in = nullptr;
        // Промежуточное поле обмена с этим соседом.
        double* buffer = nullptr;
    };
    // Загружает индексы отправки и приёма и готовит буферы одного соседа.
    Peer prepare(const std::vector<MeshIndex>& out, const std::vector<MeshIndex>& in, unsigned stride);
    // Упаковывает или распаковывает записи поля для обмена с соседом.
    void exchange(const Peer& peer, DoubleExchange::Buffer records, double* mapped, partition_detail::RecordField field, bool pack);
};
}
#endif
