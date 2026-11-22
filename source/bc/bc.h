#ifndef AERO_BC_H
#define AERO_BC_H
#include "../parallel/kernel_math.h"

#include "../eulersolver/euler.h"

namespace aero {
enum class BoundaryType { Wall, Free, Periodic, SlipWall };
// Давление действует на обе непроницаемые стенки. Только Wall закрепляет
// касательную скорость и рабочую вязкость SA при включённой вязкости.
AERO_HOST_DEVICE inline bool solid_wall(BoundaryType type) {
    return type==BoundaryType::Wall || type==BoundaryType::SlipWall;
}
enum class BoundaryMode { Pressure, NormalVelocity, Density };
enum class BoundarySource { Current, Initial, Constant, Profile, Reference, InteriorPressure, Characteristic, ExternalScalar };
enum class ProfileFallback { Full, Reference, Interior, Initial };
AERO_HOST_DEVICE inline bool takes_reference(BoundarySource source) {
    return source == BoundarySource::Reference || source == BoundarySource::InteriorPressure ||
           source == BoundarySource::Characteristic || source == BoundarySource::ExternalScalar;
}

// На дозвуковой границе задаётся одна величина. Остальные получаем из характеристик.
struct BoundaryRule {
    BoundaryMode mode;     // Давление, скорость вдоль наружной нормали или плотность.
    BoundarySource source; // Откуда взять эту величину при расчёте потока.
    double value;          // Значение для источника Constant.
};

// На сверхзвуковом входе все характеристики входят, поэтому задавать можно всё состояние.
struct PrescribedState {
    bool use_initial;    // Брать за основу начальное состояние площадки вместо текущего.
    unsigned mask;       // Установленный бит отмечает явно заданную компоненту.
    double primitive[5]; // Значения по номерам битов: rho, p, ux, uy, uz.
};
struct SupersonicInflow : PrescribedState {
    BoundarySource source; // Для reference_state сверхзвуковой вход определяется по скорости и звуку опорного газа.
};
struct FreeBoundaryRules {
    SupersonicInflow superin; // Полное состояние для сверхзвукового входа.
    BoundaryRule inflow;      // Дозвуковой вход.
    BoundaryRule outflow;     // Дозвуковой выход, включая нулевую нормальную скорость.
    PrescribedState reference; // Общие rho, p и скорости reference_state. Нулевая маска означает, что чисел в секции нет.
    bool uses_reference() const {
        return takes_reference(superin.source) || takes_reference(inflow.source) || takes_reference(outflow.source);
    }
    ProfileFallback external_base; // Для Linear без таблицы: пульсации из настройки, начального или внутреннего поля.
};
// В сетке у метки хранится одно число. Только у free это индекс в массиве FreeBoundaryRules.
// Отрицательные значения обозначают варианты стенки и периодическую метку.
class BoundaryConfig {
    enum : long long { Wall = -1, Periodic = -2, SlipWall = -3 };
    long long index_; // Номер комплекта free или код физической границы.
    AERO_HOST_DEVICE explicit BoundaryConfig(long long index) : index_(index) {}
    friend struct BoundaryPatches; // Только владелец массива назначает индексы free.
public:
    AERO_HOST_DEVICE BoundaryConfig() : index_(Wall) {}
    AERO_HOST_DEVICE static BoundaryConfig slip_wall() { return BoundaryConfig(SlipWall); }
    AERO_HOST_DEVICE static BoundaryConfig periodic() { return BoundaryConfig(Periodic); }
    AERO_HOST_DEVICE BoundaryType type() const {
        if (index_ == Wall)
            return BoundaryType::Wall;
        if (index_ == SlipWall) return BoundaryType::SlipWall;
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
enum ProfileColumn : unsigned { Radius=1, ProfileRho=2, Axial=4, Radial=8, ProfileP=16, NuTilde=32,
                                GasColumns=ProfileRho|Axial|Radial|ProfileP, AllColumns=Radius|GasColumns|NuTilde };
struct ProfileState {
    EulerState gas; // Нулевая плотность означает обычную площадку без профиля.
    double nutilde;
    unsigned columns; // Маска столбцов исходной таблицы; нуль допустим для заранее готового полного состояния.
    ProfileFallback fallback;
    double direction[3]; // Направление единственной заданной компоненты скорости в осях расчёта.
    AERO_HOST_DEVICE bool present() const { return columns || gas.density > 0.; }
    AERO_HOST_DEVICE bool partial_gas() const { return columns && (columns & GasColumns) != GasColumns; }
    // Из внутреннего или начального вектора оставляем проекцию, которой нет в профиле.
    AERO_HOST_DEVICE void missing_velocity(unsigned d, const double* velocity, double* missing) const {
        const double along = euler_detail::dot(d, velocity, direction);
        for (unsigned k=0; k<d; ++k)
            missing[k] = (columns & (Axial|Radial)) == (Axial|Radial) ? 0. : velocity[k] - along*direction[k];
    }
};
struct BoundaryPatchView {
    const BoundaryConfig* by_patch;             // Одна короткая запись на метку сетки.
    const FreeBoundaryRules* free_rules;        // Комплекты только для открытых меток.
    unsigned long long patch_count, free_count;
    const ProfileState* profiles; // По одной записи на физическую площадку.
    unsigned long long profile_count;
    AERO_HOST_DEVICE const ProfileState* profile(unsigned long long piece) const {
        return profiles && profiles[piece].present() ? profiles + piece : nullptr;
    } // Размеры двух массивов при передаче на GPU.
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
AERO_HOST_DEVICE inline BoundaryStatus apply_prescribed_state(unsigned d, const PrescribedState& rule,
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
enum class FreeRegime { SuperIn, In, Out, SuperOut };
AERO_HOST_DEVICE inline BoundarySource boundary_source(FreeRegime regime, const FreeBoundaryRules& rules) {
    if (regime == FreeRegime::In) return rules.inflow.source;
    if (regime == FreeRegime::Out) return rules.outflow.source;
    if (regime == FreeRegime::SuperIn) {
        if (rules.superin.use_initial) return BoundarySource::Initial;
        return rules.superin.source;
    }
    return BoundarySource::Current;
}
AERO_HOST_DEVICE inline FreeRegime free_regime(double un, double sound) {
    if (un >= sound)
        return FreeRegime::SuperOut;
    if (un <= -sound)
        return FreeRegime::SuperIn;
    return un < 0 ? FreeRegime::In : FreeRegime::Out;
}

// Восстанавливаем состояние из профиля в уже выбранном режиме.
AERO_HOST_DEVICE inline BoundaryStatus profile_state(unsigned d, const EulerState& ref, const EulerState& inside,
        const double* normal, double un, double gamma, EulerState& boundary, FreeRegime regime) {
    boundary = inside;
    if (regime == FreeRegime::SuperOut) return BoundaryStatus::Success;
    if (regime == FreeRegime::Out) {
        BoundaryRule rule{BoundaryMode::Pressure,BoundarySource::Constant,ref.pressure};
        return apply_subsonic_rule(d,rule,inside,normal,un,gamma,nullptr,boundary);
    }
    boundary = ref;
    if (regime == FreeRegime::SuperIn) return BoundaryStatus::Success;
    // Наружная нормаль. Из области приходит J+, из сопла J−, энтропия и касательная скорость.
    const double ref_un = euler_detail::dot(d,ref.velocity,normal);
    const double plus = un + 2.*inside.sound_speed/(gamma-1.);
    const double minus = ref_un - 2.*ref.sound_speed/(gamma-1.);
    const double c = .25*(gamma-1.)*(plus-minus);
    if (!(c > 0.)) return BoundaryStatus::InvalidTarget;
    boundary.density *= ::pow(c/ref.sound_speed,2./(gamma-1.));
    boundary.pressure = boundary.density*c*c/gamma;
    for (unsigned k=0;k<d;++k) boundary.velocity[k] += (.5*(plus+minus)-ref_un)*normal[k];
    return BoundaryStatus::Success;
}

// Заполняем пропуски в примитивных величинах, затем согласованно получаем звук и энтальпию для Роу.
AERO_HOST_DEVICE inline BoundaryStatus complete_profile(unsigned d, const ProfileState& sample, const PrescribedState& fixed,
        const EulerState& inside, const double* initial, double gamma, EulerState& gas) {
    gas = inside;
    BoundaryStatus status = BoundaryStatus::Success;
    if (sample.fallback == ProfileFallback::Initial) status = initial_state(d, initial, gamma, gas);
    if (sample.fallback == ProfileFallback::Reference) status = apply_prescribed_state(d,fixed,initial,gamma,gas);
    if (status != BoundaryStatus::Success) return status;
    double missing[3]{};
    sample.missing_velocity(d, gas.velocity, missing);
    for (unsigned k=0;k<d;++k) gas.velocity[k] = sample.gas.velocity[k] + missing[k];
    if (sample.columns & ProfileRho) gas.density = sample.gas.density;
    if (sample.columns & ProfileP) gas.pressure = sample.gas.pressure;
    gas.sound_speed = ::sqrt(gamma*(gas.pressure/gas.density));
    gas.density_root = ::sqrt(gas.density);
    gas.enthalpy = gamma/(gamma-1.)*(gas.pressure/gas.density) + .5*euler_detail::dot(d,gas.velocity,gas.velocity);
    return gas.sound_speed > 0. && euler_detail::finite(gas.enthalpy) ? BoundaryStatus::Success : BoundaryStatus::InvalidTarget;
}

AERO_HOST_DEVICE inline BoundaryStatus free_boundary_state(unsigned d, const FreeBoundaryRules& config,
                                                           const EulerState& inside, const double* normal,
                                                           double un, double gamma, const double* initial,
                                                           EulerState& boundary, FreeRegime& regime,
                                                           const ProfileState* profile, EulerState* used_profile=nullptr) {
    boundary = inside;
    EulerState completed;
    const EulerState* ref_gas = profile ? &profile->gas : nullptr;
    if (profile && profile->partial_gas()) {
        const auto status = complete_profile(d,*profile,config.reference,inside,initial,gamma,completed);
        if (status != BoundaryStatus::Success) return status;
        ref_gas = &completed;
    }
    if (ref_gas && used_profile) *used_profile = *ref_gas;
    bool reference_inlet = false;
    if (config.superin.source == BoundarySource::Reference) {
        // Сравниваем проекцию скорости на наружную нормаль со звуком в том же опорном газе.
        // Если все характеристики входят, даже при пуске из покоя задаём все компоненты reference_state.
        const auto& rule = config.reference;
        if (!rule.mask && !profile) return BoundaryStatus::InvalidTarget;
        const double* velocity = ref_gas ? ref_gas->velocity : rule.primitive + 2;
        const double c = ref_gas ? ref_gas->sound_speed : ::sqrt(gamma * (rule.primitive[1] / rule.primitive[0]));
        if (!(c > 0.) || !euler_detail::finite(c)) return BoundaryStatus::InvalidTarget;
        reference_inlet = euler_detail::dot(d, velocity, normal) <= -c;
    }
    if (reference_inlet) {
        regime = FreeRegime::SuperIn;
    } else if (profile) {
        euler_detail::RoeAverage roe;
        if (euler_detail::average_roe_state(d,inside,*ref_gas,normal,gamma,roe) != EulerStatus::Success)
            return BoundaryStatus::InvalidState;
        regime = free_regime(roe.normal_speed, roe.sound_speed);
    } else {
        regime = free_regime(un, inside.sound_speed);
    }
    // Для reference_state полное навязывание разрешено только проверкой выше. Если внутренний
    // (или осреднённый по Роу) газ втекает быстрее звука, остаётся правило inflow.
    if (config.superin.source == BoundarySource::Reference && !reference_inlet && regime == FreeRegime::SuperIn) regime = FreeRegime::In;
    const auto source = boundary_source(regime, config);
    if (takes_reference(source)) {
        if (ref_gas) boundary = *ref_gas;
        else {
            if (!config.reference.mask) return BoundaryStatus::InvalidTarget;
            const auto status = apply_prescribed_state(d, config.reference, initial, gamma, boundary);
            if (status != BoundaryStatus::Success) return status;
        }
        if (source == BoundarySource::InteriorPressure) boundary.pressure = inside.pressure;
        if (source == BoundarySource::ExternalScalar) {
            auto rule = regime == FreeRegime::In ? config.inflow : config.outflow;
            rule.value = scalar_target(d, rule.mode, boundary, normal);
            rule.source = BoundarySource::Constant;
            boundary = inside;
            return apply_subsonic_rule(d,rule,inside,normal,un,gamma,initial,boundary);
        }
        if (source == BoundarySource::Characteristic) {
            // То же J+ из области и те же входящие данные, что у дозвукового входа с профилем.
            EulerState ref = boundary;
            ref.sound_speed = ::sqrt(gamma * (ref.pressure / ref.density));
            return profile_state(d, ref, inside, normal, un, gamma, boundary, regime);
        }
        return BoundaryStatus::Success;
    }
    if (source == BoundarySource::Profile) {
        if (!profile) return BoundaryStatus::InvalidTarget;
        return profile_state(d, *ref_gas, inside, normal, un, gamma, boundary, regime);
    }
    // Нормаль направлена наружу. При u_n >= c все волны уходят, внутреннее состояние уже готово.
    if (regime == FreeRegime::SuperOut)
        return BoundaryStatus::Success;

    const bool supersonic_inflow = regime == FreeRegime::SuperIn;
    BoundaryStatus status;
    if (supersonic_inflow) {
        // При u_n <= -c все волны приходят извне. Используем отдельное полное состояние входа.
        status = apply_prescribed_state(d, config.superin, initial, gamma, boundary);
    } else {
        // При |u_n| < c знак скорости выбирает дозвуковое правило. Нуль относится к выходу.
        const auto& rule = regime == FreeRegime::In ? config.inflow : config.outflow;
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
              const double* initial, const double* area, double gamma, EulerFluxResult* result, const ProfileState* profile = nullptr) {
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
    if (solid_wall(patch.type())) {
        // Непроницаемая стенка передаёт только давление. Потоки массы и энергии уже равны нулю.
        for (unsigned k = 0; k < d; ++k)
            flux.flux[k + 1] = inside.pressure * area[k];
    } else if (patch.type() == BoundaryType::Free) {
        if (!free_rules)
            return BoundaryStatus::InvalidArgument;
        EulerState boundary;
        bc_detail::FreeRegime regime;
        const auto status = bc_detail::free_boundary_state(d, *patch.rules(free_rules), inside, normal, un,
                                                           gamma, initial, boundary, regime, profile);
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
#if !defined(AERO_BC_KERNEL_ONLY)
#include "../mesh/coordinate.h"
#include "../io/freestream.h"
#include <iosfwd>
#include <vector>
#include <string>
namespace aero {
// Владеет настройками расчёта на CPU. Порядок by_patch соответствует меткам исходной сетки.
struct ProfileRow {
    double radius, density, axial, radial, pressure, nutilde;
};
struct BoundaryProfile {
    Coordinate center{}, axis{{1,0,0}};
    unsigned columns = AllColumns;
    ProfileFallback fallback = ProfileFallback::Full;
    std::vector<ProfileRow> rows;
};
struct BoundaryPatches {
    std::vector<BoundaryProfile> profiles; // Индекс таблицы по номеру секции free.

    std::vector<BoundaryConfig> by_patch;      // Метки в порядке сетки.
    std::vector<FreeBoundaryRules> free_rules; // По одному комплекту на настроенную секцию free.

    // Создаёт одну запись в таблице правил и возвращает метку, которая на неё ссылается.
    BoundaryConfig add_free(const FreeBoundaryRules& rules) {
        const auto index = free_rules.size();
        free_rules.push_back(rules);
        return BoundaryConfig(static_cast<long long>(index));
    }
    // Представление тех же данных для CPU или для копирования в память видеокарты.
    BoundaryPatchView view(const std::vector<ProfileState>& samples = {}) const {
        return {by_patch.data(), free_rules.data(), by_patch.size(), free_rules.size(), samples.empty() ? nullptr : samples.data(), samples.size()};
    }
};
// Одна раскладка для обычного и MPI-расчёта на GPU. Нулевое число free не требует памяти под правила.
template <class DeviceStorage>
BoundaryPatchView upload_boundary_patches(DeviceStorage& storage, BoundaryPatchView host) {
    host.by_patch = storage.allocate(host.patch_count, host.by_patch);
    host.free_rules = storage.allocate(host.free_count, host.free_rules);
    host.profiles = storage.allocate(host.profile_count,host.profiles);
    return host;
}
struct PeriodicPair {
    std::string patch, pair;
    Coordinate translation; // Сдвиг от patch к pair в исходных единицах координат сетки.
};
// Читает условия одного участка. При ошибке в его описании выбрасывает std::invalid_argument.
struct BoundaryTemperature {
    FreestreamGas gas; // T/T_amb = gamma*M_inf^2*p/rho.
    double background_rho, background_p; // Для пересчёта пульсации температуры в режиме Linear.
    double background_velocity[3];
    double density(double pressure, double temperature, bool linear) const;
};
// bc_type adiabatic_wall или bc_type novisc_wall
// bc_type free требует три отдельные строки в любом порядке:
// superin [initial] [rho=...] [p=...] [ux=...] [uy=...] [uz=...]
// superin reference_state rho=... p=... ux=... [uy=...] [uz=...] либо superin reference_state с файлом профиля
// inflow  base_p|base_u|base_rho [initial|число]
// outflow base_p|base_u|base_rho [initial|число]
BoundaryConfig read_boundary(std::istream& input, unsigned dimension, BoundaryPatches& patches);
BoundaryConfig read_boundary(std::istream& input, unsigned dimension, BoundaryPatches& patches,
                             PeriodicPair& periodic, bool linear = false, BoundaryTemperature temperature = {});
}
#endif
#endif
