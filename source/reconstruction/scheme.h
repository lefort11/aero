#ifndef AERO_RECONSTRUCTION_SCHEME_H
#define AERO_RECONSTRUCTION_SCHEME_H
#include <array>
#include <stdexcept>
#include <string>

namespace aero {
enum class ReconstructionScheme { FirstOrder, Ebr3, Ebr5, Weno3, Weno5, Mp5, MPWENO5 };
enum class ReconstructionFamily { Constant, Ebr, Weno, Mp };

// Настройки, построение шаблона и разбиение MPI читают одно описание схемы.
struct ReconstructionSchemeInfo {
    ReconstructionScheme scheme;  // Значение в программном интерфейсе.
    const char* name;             // Значение reconstruction в main.txt.
    ReconstructionFamily family;  // Способ поиска опорных точек и применения весов.
    unsigned layers;              // Сколько слоёв соседей потребуется при полной реконструкции.
    double default_memory_factor; // Начальная оценка памяти при разбиении MPI, до переопределения в main.txt.
};

// MP5 и WENO используют одни и те же строки направленных наклонов.
inline bool uses_slope_rows(ReconstructionFamily family) {
    return family == ReconstructionFamily::Weno || family == ReconstructionFamily::Mp;
}

namespace detail {
// Единственный список имён и требований к геометрии.
inline const std::array<ReconstructionSchemeInfo, 7>& reconstruction_schemes() {
    static const std::array<ReconstructionSchemeInfo, 7> schemes = {
        {{ReconstructionScheme::FirstOrder, "first_order", ReconstructionFamily::Constant, 0, 4},
         {ReconstructionScheme::Ebr3, "ebr3", ReconstructionFamily::Ebr, 1, 12},
         {ReconstructionScheme::Ebr5, "ebr5", ReconstructionFamily::Ebr, 2, 16},
         {ReconstructionScheme::Weno3, "weno3", ReconstructionFamily::Weno, 1, 16},
         {ReconstructionScheme::Weno5, "weno5", ReconstructionFamily::Weno, 2, 24},
         {ReconstructionScheme::Mp5, "mp5", ReconstructionFamily::Mp, 2, 24},
         {ReconstructionScheme::MPWENO5, "MPWENO5", ReconstructionFamily::Mp, 2, 24}}};
    return schemes;
}
}

// Преобразует имя в описание схемы. При ошибке сохраняет место в файле настроек.
inline ReconstructionSchemeInfo reconstruction_scheme(const std::string& name,
                                                      const std::string& where = "reconstruction") {
    for (const auto& info : detail::reconstruction_schemes()) {
        if (name == info.name || (name == "mpweno5" && info.scheme == ReconstructionScheme::MPWENO5) ||
            (name == "MP5" && info.scheme == ReconstructionScheme::Mp5)) {
            return info;
        }
    }
    throw std::invalid_argument(where + ": unknown value " + name);
}

class ReconstructionError : public std::runtime_error {
public:
    explicit ReconstructionError(const std::string& message) : std::runtime_error(message) {}
};

// Программные вызовы и чтение настроек поддерживают один набор схем.
inline ReconstructionSchemeInfo reconstruction_scheme(ReconstructionScheme scheme) {
    for (const auto& info : detail::reconstruction_schemes()) {
        if (info.scheme == scheme) {
            return info;
        }
    }
    throw ReconstructionError("Invalid reconstruction scheme");
}
}
#endif
