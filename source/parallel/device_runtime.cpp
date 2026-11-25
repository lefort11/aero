#define AERO_BUILD_DEVICE_WORK
#include "device_work.h"
#include "device_execution.h"
#include "device_platform.h"
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>

#if defined(_WIN32) && defined(AERO_STATIC_DEVICE_WORK) && !defined(__CUDA_ARCH__)
// cudart_static собрана MSVC с проверкой стека. При запуске через MinGW выполняем
// инициализацию её cookie до конструкторов CUDA, как это делает штатный CRT MSVC.
extern "C" void __cdecl __security_init_cookie();
namespace {
// Подготавливает CUDA Runtime до создания контекстов Kokkos.
int __cdecl initialize_cuda_crt() { __security_init_cookie(); return 0; }
__attribute__((used, section(".CRT$XIB"))) int (__cdecl* const cuda_crt_initializer)() = initialize_cuda_crt;
}
#endif

namespace aero {
namespace device_work {
namespace {
// Запускаем Kokkos после выбора карты для данного MPI-процесса.
// Стандартный поток используется на каждой стадии, поэтому сохраняем исполнителя между вызовами.
class DeviceRuntime {
    int primary_; // Карта первого обращения. Ей принадлежит стандартный исполнитель Kokkos.
    std::map<int, Execution> defaults_; // Остальные карты для библиотечных вызовов из одного процесса.
    std::mutex defaults_mutex_; // Защищает создание исполнителя при обращениях из разных потоков CPU.
public:
    // Инициализирует Kokkos и контексты выполнения на заданной карте.
    explicit DeviceRuntime(int device) : primary_(device) {
        Kokkos::InitArguments args;
        args.device_id = device;
        args.disable_warnings = true;
        Kokkos::initialize(args);
    }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~DeviceRuntime() {
        int original = primary_;
        gpu::native::get_device(&original);
        while (!defaults_.empty()) {
            auto pos = defaults_.begin();
            gpu::native::set_device(pos->first);
            defaults_.erase(pos);
        }
        gpu::native::set_device(primary_);
        Kokkos::finalize();
        gpu::native::set_device(original);
    }
    // Возвращает контекст исполнения для требуемого устройства или потока.
    Execution execution(int device, gpu::native::Stream stream) {
        // Чужой поток может быть уничтожен сразу после возврата из библиотечного вызова.
        if (stream)
            return Execution(stream);
        if (device == primary_)
            return Execution();
        std::lock_guard<std::mutex> guard(defaults_mutex_);
        auto pos = defaults_.find(device);
        if (pos == defaults_.end())
            pos = defaults_.emplace(device, Execution(gpu::native::default_stream())).first;
        return pos->second;
    }
};

// Возвращает единственный объект управления средой GPU в текущем модуле.
std::unique_ptr<DeviceRuntime>& runtime_holder() {
    static std::unique_ptr<DeviceRuntime> holder;
    return holder;
}
// Выбирает устройство и соответствующий контекст выполнения ядра.
Execution select_execution(int device, gpu::native::Stream stream) {
    auto& runtime = runtime_holder();
    static std::once_flag initialized;
    std::call_once(initialized, [&] {
        runtime.reset(new DeviceRuntime(device));
        // Копирование полей может быть единственной операцией на GPU. В этом
        // случае host-обёртки вычислительных ядер вообще не вызываются. Регистрируем
        // завершение после инициализации внутренних объектов Kokkos, пока они ещё живы.
        if (std::atexit(aero_device_shutdown)) {
            runtime.reset();
            throw std::runtime_error("Unable to register Kokkos finalizer");
        }
    });
    return runtime->execution(device, stream);
}
}


// Возвращает контекст Kokkos для выбранного устройства и потока устройства.
Execution current_execution(void* stream) {
    int device;
    const auto status = gpu::native::get_device(&device);
    if (status != gpu::native::success)
        throw std::runtime_error(gpu::native::error_string(status));
    return select_execution(device, static_cast<gpu::native::Stream>(stream));
}
// Завершает среду выполнения вычислительных ядер GPU.
void aero_device_shutdown() {
    runtime_holder().reset();
}
}
}
