#include "explicit.h"
#include "stage_sequence.h"
#include "../parallel/kokkos_host.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef AERO_WITH_KOKKOS_GPU
#include "../parallel/device_check.h"
#include "../parallel/device_scope.h"
#endif

namespace aero {
namespace {
#ifdef AERO_WITH_KOKKOS_GPU
using gpu::checked;
#endif
}

// Внутренние поля, буферы стадий и состояние исполнителя явного интегратора.
struct ExplicitIntegrator::Impl {
    std::size_t nodes = 0, components = 0, size = 0; // Число узлов, компонент на узел и элементов поля.
    int device = -1;                                 // Номер карты для CUDA.
    ExplicitBackend backend = ExplicitBackend::Cpu;  // Устройство, где находятся все четыре массива.
    std::vector<double> cpu;                         // Владелец общей области памяти на CPU.
    double* volumes = nullptr;                       // Неподвижные объёмы узлов.
    double* accepted = nullptr;                      // Принятое U_n, доступное на протяжении всего шага.
    double* stage_values = nullptr;                  // Сначала Y1, затем Y2 и окончательное U_next.
    double* residual = nullptr;                      // Невязка текущей стадии.
    bool busy = false;                               // Защита от повторного входа через обратный вызов.
#ifdef AERO_WITH_KOKKOS_GPU
    double* allocation = nullptr;  // Владелец непрерывной области массивов на GPU.
    int* invalid = nullptr;        // Флаг ошибки обновления в ядре CUDA.
    aero::gpu::Stream stream = nullptr; // Очередь сборки невязки и обновления этого интегратора.
#endif
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~Impl() {
#ifdef AERO_WITH_KOKKOS_GPU
        if (device < 0)
            return;
        gpu::DeviceScope scope(device);
        if (scope.status == aero::gpu::Status::Success) {
            if (stream)
                aero::gpu::wait(stream);
            if (allocation)
                aero::gpu::release(allocation);
            if (invalid)
                aero::gpu::release(invalid);
            if (stream)
                aero::gpu::destroy_stream(stream);
        }
#endif
    }
    // Размечает общий буфер объёмов и полей явных стадий.
    void bind(double* storage) {
        volumes = storage;
        // Два поля одинакового размера позволяют принять целый шаг простой перестановкой указателей.
        accepted = volumes + nodes;
        stage_values = accepted + size;
        residual = stage_values + size;
    }
    // Возвращает поток GPU либо nullptr для последовательного CPU-интерфейса.
    void* stream_handle() const {
#ifdef AERO_WITH_KOKKOS_GPU
        return stream;
#else
        return nullptr;
#endif
    }
    // Сборщик и наблюдатель получают одинаковое представление выбранного поля.
    ExplicitField view(const double* values) const {
        return {nodes, components, values, volumes, backend, device, stream_handle()};
    }

    // Обратный вызов может завершиться исключением. Освобождаем интегратор для следующего вызова в обоих случаях.
    template <class Operation>
    void with_exclusive_access(const char* nested_error, const Operation& operation) {
        if (busy)
            throw std::logic_error(nested_error);
#ifdef AERO_WITH_KOKKOS_GPU
        gpu::DeviceScope scope(device);
        checked(scope.status);
#endif
        busy = true;
        try {
            operation();
        } catch (...) {
#ifdef AERO_WITH_KOKKOS_GPU
            // Вызывающий код мог поставить работу в очередь перед ошибкой. Дожидаемся её перед повторной попыткой.
            if (stream)
                aero::gpu::wait(stream);
#endif
            busy = false;
            throw;
        }
        busy = false;
    }

    // Обновляет рабочее поле целиком. Принятое U_n остаётся источником весовой суммы SSPRK3.
    void update_stage(explicit_detail::Stage formula, const double* current, const double* rhs, double dt) {
        const explicit_detail::FieldStageUpdate update = {formula, components, volumes, accepted,
                                                          current, rhs,        dt,      stage_values};
        int bad = 0;
#ifdef AERO_WITH_KOKKOS_GPU
        if (backend == ExplicitBackend::Cuda) {
            checked(aero::gpu::fill_async(invalid, 0, sizeof(int), stream));
            explicit_detail::launch_update(update, size, invalid, stream);

            // До следующей стадии нужно знать, получились ли конечные значения во всём поле.
            checked(aero::gpu::copy_async(&bad, invalid, sizeof(int), aero::gpu::Copy::ToHost, stream));
            checked(aero::gpu::wait(stream));
        } else
#endif
        {
            const auto count = static_cast<std::ptrdiff_t>(size);
            if (size >= 4096) {
                // Каждая компонента обновляется ровно один раз. Любая ошибка отклоняет стадию.
                host_parallel::ensure_runtime();
                Kokkos::parallel_reduce("Aero: update stage", host_parallel::Range(0, size),
                    [&](std::size_t k, int& flags) { flags |= !update.apply(k); }, Kokkos::BOr<int>(bad));
            } else
            for (std::ptrdiff_t k = 0; k < count; ++k) {
                if (!update.apply(static_cast<std::size_t>(k)))
                    bad = 1;
            }
        }
        if (bad)
            throw std::domain_error("non-finite explicit stage");
    }
};

// Возвращает число доступных CUDA-устройств.
int ExplicitIntegrator::cuda_device_count() {
#ifdef AERO_WITH_KOKKOS_GPU
    int count = 0;
    const aero::gpu::Status code = aero::gpu::count(&count);
    if (code == aero::gpu::Status::NoDevice || code == aero::gpu::Status::Unavailable) {

        return 0;
    }
    checked(code);
    return count;
#else
    return 0;
#endif
}

// Создаёт буферы явных стадий и копирует исходные объёмы и поле.
ExplicitIntegrator::ExplicitIntegrator(std::size_t nodes, std::size_t components, const double* volumes,
                                       const double* initial, ExplicitBackend backend, int device)
    : impl_(new Impl) {
    const std::size_t limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (!nodes || !components || !volumes || !initial || nodes > limit ||
        components > (limit - nodes) / nodes / 3)
        throw std::invalid_argument("invalid explicit field dimensions or storage");
    if (device < 0 || (backend != ExplicitBackend::Auto && backend != ExplicitBackend::Cpu &&
                       backend != ExplicitBackend::Cuda))
        throw std::invalid_argument("invalid explicit backend or device");
    Impl& p = *impl_;
    p.nodes = nodes;
    p.components = components;
    p.size = nodes * components;
    for (std::size_t n = 0; n < nodes; ++n)
        if (!(volumes[n] > 0) || !std::isfinite(volumes[n]))
            throw std::invalid_argument("invalid control volume");
    for (std::size_t k = 0; k < p.size; ++k)
        if (!std::isfinite(initial[k]))
            throw std::invalid_argument("non-finite initial state");
    const int devices = backend == ExplicitBackend::Cpu ? 0 : cuda_device_count();
    if (backend == ExplicitBackend::Cuda || (backend == ExplicitBackend::Auto && devices > 0)) {
        if (device >= devices)
            throw std::invalid_argument("requested CUDA device unavailable");
#ifdef AERO_WITH_KOKKOS_GPU
        gpu::DeviceScope scope(device);
        checked(scope.status);
        p.backend = ExplicitBackend::Cuda;
        p.device = device;
        checked(aero::gpu::create_stream(&p.stream, true));
        checked(aero::gpu::allocate(reinterpret_cast<void**>(&p.allocation), (nodes + 3 * p.size) * sizeof(double)));
        checked(aero::gpu::allocate(reinterpret_cast<void**>(&p.invalid), sizeof(int)));
        p.bind(p.allocation);
        checked(
            aero::gpu::copy_async(p.volumes, volumes, nodes * sizeof(double), aero::gpu::Copy::ToDevice, p.stream));
        checked(
            aero::gpu::copy_async(p.accepted, initial, p.size * sizeof(double), aero::gpu::Copy::ToDevice, p.stream));
        checked(aero::gpu::wait(p.stream));
#endif
    } else {
        p.cpu.resize(nodes + 3 * p.size);
        p.bind(p.cpu.data());
        std::copy(volumes, volumes + nodes, p.volumes);
        std::copy(initial, initial + p.size, p.accepted);
    }
}
// Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
ExplicitIntegrator::~ExplicitIntegrator() {}
// Возвращает выбранный исполнитель вычислений.
ExplicitBackend ExplicitIntegrator::backend() const {
    return impl_->backend;
}

// Даёт доступ к принятому полю между шагами. Для CUDA на время чтения активируется выбранное устройство.
void ExplicitIntegrator::inspect(const std::function<void(const ExplicitField&)>& reader) const {
    Impl& p = *impl_;
    if (!reader)
        throw std::invalid_argument("missing field reader");
    p.with_exclusive_access("field inspection requires an idle integrator", [&] {
        reader(p.view(p.accepted));
#ifdef AERO_WITH_KOKKOS_GPU
        if (p.stream)
            checked(aero::gpu::wait(p.stream));
#endif
    });
}

// Преобразует имя временной схемы в её внутреннее обозначение.
ExplicitScheme explicit_detail::scheme_from_name(const std::string& name) {
    if (name == "euler")
        return ExplicitScheme::Euler;
    if (name == "ssprk3")
        return ExplicitScheme::SSPRK3;
    throw std::invalid_argument("unknown explicit scheme: " + name);
}
// Выбирает таблицу стадий указанной явной схемы.
explicit_detail::Stages explicit_detail::stages_for(ExplicitScheme scheme) {
    switch (scheme) {
    case ExplicitScheme::Euler:
        return euler_stages();
    case ExplicitScheme::SSPRK3:
        return ssprk3_stages();
    default:
        throw std::invalid_argument("unknown explicit scheme");
    }
}

// Выполняет полный явный шаг выбранной схемы и принимает поле после всех стадий.
void ExplicitIntegrator::step(ExplicitScheme scheme, double time, double dt,
                              const ExplicitResidual& assemble) {
    step(scheme, time, dt, assemble, nullptr);
}
// Выполняет полный явный шаг выбранной схемы и принимает поле после всех стадий.
void ExplicitIntegrator::step(ExplicitScheme scheme, double time, double dt, const ExplicitResidual& assemble,
                              const double* first_residual) {
    if (!assemble || !std::isfinite(time) || !(dt > 0) || !std::isfinite(time + dt))
        throw std::invalid_argument("invalid explicit step time, dt or residual callback");
    const auto stages = explicit_detail::stages_for(scheme);
    Impl& p = *impl_;
    p.with_exclusive_access("recursive explicit step", [&] {
        // На первой стадии читаем принятое U_n. Дальше current указывает на результат предыдущей стадии.
        const double* current = p.accepted;
        // Вычисляет очередной этап оператора для текущего состояния поля.
        const auto compute = [&](double stage_time) { assemble(stage_time, p.view(current), p.residual); };
        // При наличии готовой первой невязки используем её ровно один раз. Следующие пишет compute.
        const double* rhs = first_residual ? first_residual : p.residual;
        // Применяет выбранное обновление рабочих полей текущего этапа решателя.
        const auto update = [&](explicit_detail::StepStage stage) {
            p.update_stage(stage.formula, current, rhs, dt);
            current = p.stage_values;
            rhs = p.residual;
        };
        using explicit_detail::FirstResidual;
        const auto first = first_residual ? FirstResidual::Prepared : FirstResidual::NeedsAssembly;
        explicit_detail::advance_stages(stages, time, dt, first, compute, update, [] {});

        // Принимаем U_next только после успеха всех стадий. Если одна сорвалась, p.accepted всё ещё содержит U_n.
        std::swap(p.accepted, p.stage_values);
    });
}
// Возвращает CPU-копию последнего принятого поля.
std::vector<double> ExplicitIntegrator::state() const {
    const Impl& p = *impl_;
#ifdef AERO_WITH_KOKKOS_GPU
    if (p.backend == ExplicitBackend::Cuda) {
        gpu::DeviceScope scope(p.device);
        checked(scope.status);
        std::vector<double> out(p.size);
        checked(aero::gpu::copy_async(out.data(), p.accepted, p.size * sizeof(double), aero::gpu::Copy::ToHost,
                                p.stream));
        checked(aero::gpu::wait(p.stream));
        return out;
    }
#endif
    return std::vector<double>(p.accepted, p.accepted + p.size);
}
}
