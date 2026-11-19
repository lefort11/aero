#ifndef AERO_BC_H
#define AERO_BC_H
#include "../parallel/kernel_math.h"

#include "../eulersolver/euler.h"

namespace aero {
enum class BoundaryType { Wall, Free, Periodic };
enum class BoundaryMode { Pressure, NormalVelocity, Density };
enum class BoundarySource { Current, Initial, Constant };

// На дозвуковой границе задаётся одна величина. Остальные получаем из характеристик.
struct BoundaryRule {
    BoundaryMode mode;     // Давление, скорость вдоль наружной нормали или плотность.
    BoundarySource source; // Откуда взять эту величину при расчёте потока.
    double value;          // Значение для источника Constant.
};

// На сверхзвуковом входе все характеристики входят, поэтому задавать можно всё состояние.
struct SupersonicInflow {
    bool use_initial;    // Брать за основу начальное состояние площадки вместо текущего.
    unsigned mask;       // Установленный бит отмечает явно заданную компоненту.
    double primitive[5]; // Значения по номерам битов: rho, p, ux, uy, uz.
};
struct FreeBoundaryRules {
    SupersonicInflow superin; // Полное состояние для сверхзвукового входа.
    BoundaryRule inflow;      // Дозвуковой вход.
    BoundaryRule outflow;     // Дозвуковой выход, включая нулевую нормальную скорость.
};
// В сетке у метки хранится одно число. Только у free это индекс в массиве FreeBoundaryRules.
// Два специальных значения обозначают стенку и периодическую метку, у которых правил течения нет.
class BoundaryConfig {
    enum : long long { Wall = -1, Periodic = -2 };
    long long index_; // Номер комплекта free или одно из двух значений выше.
    AERO_HOST_DEVICE explicit BoundaryConfig(long long index) : index_(index) {}
    friend struct BoundaryPatches; // Только владелец массива назначает индексы free.
public:
    AERO_HOST_DEVICE BoundaryConfig() : index_(Wall) {}
    AERO_HOST_DEVICE static BoundaryConfig periodic() { return BoundaryConfig(Periodic); }
    AERO_HOST_DEVICE BoundaryType type() const {
        if (index_ == Wall)
            return BoundaryType::Wall;
        if (index_ == Periodic)
            return BoundaryType::Periodic;
        return BoundaryType::Free;
    }
    // Возвращает правила только для открытой метки. Массив живёт на том же устройстве, что и расчёт.
    AERO_HOST_DEVICE const FreeBoundaryRules* rules(const FreeBoundaryRules* table) const {
        return index_ >= 0 && table ? table + index_ : nullptr;
    }
};

// Этот вид таблиц передаётся в вычислительные ядра и не владеет памятью.
struct BoundaryPatchView {
    const BoundaryConfig* by_patch;             // Одна короткая запись на метку сетки.
    const FreeBoundaryRules* free_rules;        // Комплекты только для открытых меток.
    unsigned long long patch_count, free_count; // Размеры двух массивов при передаче на GPU.
};

enum class BoundaryStatus {
    Success,
    InvalidArgument,
    InvalidState,
    MissingInitial,
    IncompatibleRegime,
    InvalidTarget
};

namespace bc_detail {
// Переводим сохранённые консервативные переменные в примитивные только по запросу активного правила.
AERO_HOST_DEVICE inline BoundaryStatus initial_state(unsigned d, const double* initial, double gamma,
                                                     EulerState& state) {
    if (!initial)
        return BoundaryStatus::MissingInitial;
    return euler_state(d, initial, gamma, &state) == EulerStatus::Success ? BoundaryStatus::Success
                                                                          : BoundaryStatus::InvalidState;
}

// На входе boundary уже содержит внутреннее состояние. Сначала выбираем основу, затем заменяем отмеченные поля.
AERO_HOST_DEVICE inline BoundaryStatus apply_supersonic_inflow(unsigned d, const SupersonicInflow& rule,
                                                               const double* initial, double gamma,
                                                               EulerState& boundary) {
    if (rule.use_initial) {
        const auto status = initial_state(d, initial, gamma, boundary);
        if (status != BoundaryStatus::Success)
            return status;
    }
    if (rule.mask >> (2 + d))
        return BoundaryStatus::InvalidTarget; // Компонента вне размерности сетки.
    if (rule.mask & 1u)
        boundary.density = rule.primitive[0];
    if (rule.mask & 2u)
        boundary.pressure = rule.primitive[1];
    for (unsigned k = 0; k < d; ++k) {
        if (rule.mask & (4u << k))
            boundary.velocity[k] = rule.primitive[2 + k];
    }
    return BoundaryStatus::Success;
}

// Для initial берём только заданную величину из начального состояния этой площадки.
AERO_HOST_DEVICE inline double scalar_target(unsigned d, BoundaryMode mode, const EulerState& state,
                                             const double* normal) {
    if (mode == BoundaryMode::Pressure)
        return state.pressure;
    if (mode == BoundaryMode::Density)
        return state.density;
    return euler_detail::dot(d, state.velocity, normal);
}

// Восстанавливает дозвуковое состояние по одному условию, внутренней энтропии и исходящей характеристике.
AERO_HOST_DEVICE inline BoundaryStatus apply_subsonic_rule(unsigned d, const BoundaryRule& rule,
                                                           const EulerState& inside, const double* normal,
                                                           double un, double gamma, const double* initial,
                                                           EulerState& boundary) {
    double target = rule.value;
    if (rule.source == BoundarySource::Initial) {
        EulerState reference;
        const auto status = initial_state(d, initial, gamma, reference);
        if (status != BoundaryStatus::Success)
            return status;
        target = scalar_target(d, rule.mode, reference, normal);
    }
    if (!euler_detail::finite(target) || (rule.mode != BoundaryMode::NormalVelocity && target <= 0))
        return BoundaryStatus::InvalidTarget;

    const double c = inside.sound_speed;
    double boundary_un = un;
    if (rule.mode == BoundaryMode::NormalVelocity) {
        // Задана скорость. Из инварианта J+ находим скорость звука, из изэнтропы — плотность и давление.
        boundary_un = target;
        const double boundary_c = c + 0.5 * (gamma - 1) * (un - boundary_un);
        if (!(boundary_c > 0))
            return BoundaryStatus::InvalidTarget;
        boundary.density *= ::pow(boundary_c / c, 2 / (gamma - 1));
        boundary.pressure *= ::pow(boundary_c / c, 2 * gamma / (gamma - 1));
    } else {
        // Задано давление или плотность. Вторую величину находим из p/rho^gamma = const.
        if (rule.mode == BoundaryMode::Pressure) {
            boundary.pressure = target;
            boundary.density *= ::pow(target / inside.pressure, 1 / gamma);
        } else {
            boundary.density = target;
            boundary.pressure *= ::pow(target / inside.density, gamma);
        }
        const double boundary_c = ::sqrt(gamma * (boundary.pressure / boundary.density));
        boundary_un += 2 * (c - boundary_c) / (gamma - 1); // Сохраняем J+ = u_n + 2c/(gamma-1).
    }
    // Добавка вдоль единичной нормали сохраняет касательную скорость внутреннего газа.
    for (unsigned k = 0; k < d; ++k)
        boundary.velocity[k] += (boundary_un - un) * normal[k];
    return BoundaryStatus::Success;
}

// Проверяем, что предписанное состояние физично и совместимо с режимом, выбранным по внутреннему газу.
AERO_HOST_DEVICE inline BoundaryStatus check_boundary_state(unsigned d, const EulerState& boundary,
                                                            const double* normal, double gamma,
                                                            double inside_un, bool supersonic_inflow) {
    if (!(boundary.pressure > 0 && boundary.density > 0) || !euler_detail::finite(boundary.pressure) ||
        !euler_detail::finite(boundary.density))
        return BoundaryStatus::InvalidTarget;
    const double c = ::sqrt(gamma * (boundary.pressure / boundary.density));
    const double un = euler_detail::dot(d, boundary.velocity, normal);
    if (!(c > 0 && euler_detail::finite(c) && euler_detail::finite(un)))
        return BoundaryStatus::InvalidTarget;
    if (supersonic_inflow) {
        if (un > -c)
            return BoundaryStatus::IncompatibleRegime;
    } else if (::fabs(un) >= c || (inside_un < 0 ? un > 0 : un < 0)) {
        return BoundaryStatus::IncompatibleRegime;
    }
    return BoundaryStatus::Success;
}

// Выбирает одно правило free для текущей площадки при каждом вычислении потока.
AERO_HOST_DEVICE inline BoundaryStatus free_boundary_state(unsigned d, const FreeBoundaryRules& config,
                                                           const EulerState& inside, const double* normal,
                                                           double un, double gamma, const double* initial,
                                                           EulerState& boundary) {
    boundary = inside;
    // Нормаль направлена наружу. При u_n >= c все волны уходят, внутреннее состояние уже готово.
    if (un >= inside.sound_speed)
        return BoundaryStatus::Success;

    const bool supersonic_inflow = un <= -inside.sound_speed;
    BoundaryStatus status;
    if (supersonic_inflow) {
        // При u_n <= -c все волны приходят извне. Используем отдельное полное состояние входа.
        status = apply_supersonic_inflow(d, config.superin, initial, gamma, boundary);
    } else {
        // При |u_n| < c знак скорости выбирает дозвуковое правило. Нуль относится к выходу.
        const auto& rule = un < 0 ? config.inflow : config.outflow;
        if (rule.source == BoundarySource::Current)
            return BoundaryStatus::Success;
        status = apply_subsonic_rule(d, rule, inside, normal, un, gamma, initial, boundary);
    }
    if (status != BoundaryStatus::Success)
        return status;
    return check_boundary_state(d, boundary, normal, gamma, un, supersonic_inflow);
}

// Интегрирует физический поток Эйлера по площадке. Вектор area уже содержит её площадь.
AERO_HOST_DEVICE inline void open_boundary_flux(unsigned d, const EulerState& state, const double* normal,
                                                const double* area, double gamma, EulerFluxResult& flux) {
    const double volume_flux = euler_detail::dot(d, state.velocity, area);
    flux.flux[0] = state.density * volume_flux; // Расход массы наружу через площадку.
    for (unsigned k = 0; k < d; ++k)
        flux.flux[k + 1] = flux.flux[0] * state.velocity[k] + state.pressure * area[k];
    const double enthalpy_density =
        gamma / (gamma - 1) * state.pressure +
        0.5 * state.density * euler_detail::dot(d, state.velocity, state.velocity);
    flux.flux[d + 1] = enthalpy_density * volume_flux;
    // В оценке шага CFL учитываем как внутреннее состояние, так и состояние на границе.
    const double un = euler_detail::dot(d, state.velocity, normal);
    flux.max_wave_speed =
        ::fmax(flux.max_wave_speed, ::fabs(un) + ::sqrt(gamma * (state.pressure / state.density)));
}
} // namespace bc_detail

// Общий для CPU и CUDA поток одной физической площадки. При ошибке result сохраняет прежнее значение.
AERO_HOST_DEVICE inline BoundaryStatus
boundary_flux(unsigned d, BoundaryConfig patch, const FreeBoundaryRules* free_rules, const double* current,
              const double* initial, const double* area, double gamma, EulerFluxResult* result) {
    double normal[3] = {};
    if (!result || !current || euler_unit_direction(d, area, normal) != EulerStatus::Success)
        return BoundaryStatus::InvalidArgument;
    EulerState inside;
    if (euler_state(d, current, gamma, &inside) != EulerStatus::Success)
        return BoundaryStatus::InvalidState;
    const double un = euler_detail::dot(d, inside.velocity, normal);

    EulerFluxResult flux = {};
    flux.components = d + 2;
    flux.max_wave_speed = ::fabs(un) + inside.sound_speed;
    if (patch.type() == BoundaryType::Wall) {
        // Непроницаемая стенка передаёт только давление. Потоки массы и энергии уже равны нулю.
        for (unsigned k = 0; k < d; ++k)
            flux.flux[k + 1] = inside.pressure * area[k];
    } else if (patch.type() == BoundaryType::Free) {
        if (!free_rules)
            return BoundaryStatus::InvalidArgument;
        EulerState boundary;
        const auto status = bc_detail::free_boundary_state(d, *patch.rules(free_rules), inside, normal, un,
                                                           gamma, initial, boundary);
        if (status != BoundaryStatus::Success)
            return status;
        bc_detail::open_boundary_flux(d, boundary, normal, area, gamma, flux);
    } else {
        return BoundaryStatus::InvalidArgument; // Периодические пары обрабатываются как внутренние связи.
    }
    // Записываем результат только после проверки всех компонент.
    if (!euler_detail::finite(flux.max_wave_speed))
        return BoundaryStatus::InvalidState;
    for (unsigned k = 0; k < flux.components; ++k)
        if (!euler_detail::finite(flux.flux[k]))
            return BoundaryStatus::InvalidState;
    *result = flux;
    return BoundaryStatus::Success;
}
}
#if !defined(AERO_BC_KERNEL_ONLY) && !defined(__CUDACC_RTC__)
#include <iosfwd>
#include <vector>
#include <string>
namespace aero {
// Владеет настройками расчёта на CPU. Порядок by_patch соответствует меткам исходной сетки.
struct BoundaryPatches {
    std::vector<BoundaryConfig> by_patch;      // Метки в порядке сетки.
    std::vector<FreeBoundaryRules> free_rules; // По одному комплекту на настроенную секцию free.

    // Создаёт одну запись в таблице правил и возвращает метку, которая на неё ссылается.
    BoundaryConfig add_free(const FreeBoundaryRules& rules) {
        const auto index = free_rules.size();
        free_rules.push_back(rules);
        return BoundaryConfig(static_cast<long long>(index));
    }
    // Представление тех же данных для CPU или для копирования в память видеокарты.
    BoundaryPatchView view() const {
        return {by_patch.data(), free_rules.data(), by_patch.size(), free_rules.size()};
    }
};
// Одна раскладка для обычного и MPI-расчёта на GPU. Нулевое число free не требует памяти под правила.
template <class DeviceStorage>
BoundaryPatchView upload_boundary_patches(DeviceStorage& storage, BoundaryPatchView host) {
    host.by_patch = storage.allocate(host.patch_count, host.by_patch);
    host.free_rules = storage.allocate(host.free_count, host.free_rules);
    return host;
}
struct PeriodicPair {
    std::string patch, pair;
    double translation[3]; // Сдвиг от patch к pair в исходных единицах координат сетки.
};
// Читает условия одного участка. При ошибке в его описании выбрасывает std::invalid_argument.
// bc_type wall
// bc_type free требует три отдельные строки в любом порядке:
// superin [initial] [rho=...] [p=...] [ux=...] [uy=...] [uz=...]
// inflow  base_p|base_u|base_rho [initial|число]
// outflow base_p|base_u|base_rho [initial|число]
BoundaryConfig read_boundary(std::istream& input, unsigned dimension, BoundaryPatches& patches);
BoundaryConfig read_boundary(std::istream& input, unsigned dimension, BoundaryPatches& patches,
                             PeriodicPair& periodic);
}
#endif
#endif
