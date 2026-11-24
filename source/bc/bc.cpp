#include "bc.h"
#include "../io/settings_text.h"

#include <sstream>
#include <locale>
#include <memory>
#include <string>

namespace aero {
// Восстанавливает плотность по параметрам граничного газа.
double BoundaryTemperature::density(double pressure, double temperature, bool linear) const {
    const double scale = gas.gamma * gas.mach * gas.mach;
    if (!(scale > 0.) || !std::isfinite(scale)) throw std::invalid_argument("reference_state T requires a positive mach in gas.txt");
    if (!linear && !(temperature > 0.)) throw std::invalid_argument("reference_state T must be positive");
    const double rho = linear ? background_rho/background_p * (pressure - background_rho*temperature/scale)
                              : scale * pressure / temperature;
    if (!std::isfinite(rho) || (!linear && !(rho > 0.))) throw std::invalid_argument("invalid density from reference_state p and T");
    return rho;
}
namespace {
using Tokens = std::vector<std::string>;

// Разбирает весь токен в локали с десятичной точкой и отвергает NaN и бесконечность.
double number(const std::string& token) {
    return settings_detail::finite_real(token, "boundary value " + token);
}

// Общий синтаксис компонент для опорного состояния и замен в обычном superin.
void read_components(const Tokens& words, size_t first, unsigned dimension, bool linear, PrescribedState& rule) {
    const char* fields[] = {"rho", "p", "ux", "uy", "uz"};
    for (size_t i = first; i < words.size(); ++i) {
        const auto equals = words[i].find('=');
        const auto name = words[i].substr(0, equals);
        // В 1D допустима только ux, в 2D добавляется uy, в 3D также uz.
        unsigned field = 0;
        while (field < 2 + dimension && name != fields[field])
            ++field;
        if (equals == std::string::npos || field == 2 + dimension || (rule.mask & (1u << field)))
            throw std::invalid_argument("invalid or repeated boundary field: " + words[i]);
        rule.primitive[field] = number(words[i].substr(equals + 1));
        if (!linear && field < 2 && rule.primitive[field] <= 0)
            throw std::invalid_argument("expected positive " + name);
        rule.mask |= 1u << field; // Остальные компоненты солвер возьмёт из выбранного состояния.
    }
}

// Оба дозвуковых режима используют один синтаксис и одну проверку численного значения.
BoundaryRule read_subsonic_rule(const Tokens& words, bool linear) {
    if (words.size() < 2 || words.size() > 3)
        throw std::invalid_argument("expected " + words[0] + " base_p|base_u|base_rho [initial|number]");
    const auto& name = words[1];
    BoundaryRule rule = {};
    if (name == "p_interior" && words.size() == 2) { rule.source = BoundarySource::InteriorPressure; return rule; }
    if (name == "in_char") {
        if (words[0] != "inflow" || words.size() != 2)
            throw std::invalid_argument("expected 'inflow in_char' without additional arguments");
        rule.source = BoundarySource::Characteristic;
        return rule;
    }
    if ((name == "reference_state" || name == "external") && words.size() == 2) { rule.source = BoundarySource::Reference; return rule; }
    if (name == "profile" && words.size()==2 && words[0]=="inflow") { rule.source=BoundarySource::Profile; return rule; }
    if (name == "base_p")
        rule.mode = BoundaryMode::Pressure;
    else if (name == "base_u")
        rule.mode = BoundaryMode::NormalVelocity;
    else if (name == "base_rho")
        rule.mode = BoundaryMode::Density;
    else
        throw std::invalid_argument("unknown subsonic boundary rule: " + name);

    // Без аргумента используем значение величины непосредственно внутри границы.
    if (words.size() == 2)
        rule.source = BoundarySource::Current;
    else if (words[2] == "external" && rule.mode == BoundaryMode::Pressure) rule.source = BoundarySource::ExternalScalar;
    else if (words[2] == "profile") rule.source = BoundarySource::Profile;
    else if (words[2] == "initial")
        rule.source = BoundarySource::Initial;
    else {
        rule.source = BoundarySource::Constant;
        rule.value = number(words[2]);
        if (!linear && rule.mode != BoundaryMode::NormalVelocity && rule.value <= 0)
            throw std::invalid_argument("expected positive value for " + name);
    }
    return rule;
}

// Тип метки описывает её назначение. Режим течения для free определится во время расчёта.
BoundaryType read_type(const Tokens& words) {
    if (words.size() == 2) {
        if (words[1] == "free")
            return BoundaryType::Free;
        if (words[1] == "novisc_wall") return BoundaryType::SlipWall;
        if (words[1] == "adiabatic_wall" || words[1] == "wall")
            return BoundaryType::Wall;
        if (words[1] == "periodic")
            return BoundaryType::Periodic;
    }
    throw std::invalid_argument("expected bc_type adiabatic_wall, novisc_wall, free or periodic");
}

// Промежуточное состояние одной секции позволяет задавать строки в любом порядке.
struct BoundarySection {
    // Температурное условие для данного участка границы.
    BoundaryTemperature temperature{};
    BoundaryType type = BoundaryType::Wall;   // Тип, прочитанный из bc_type.
    std::unique_ptr<FreeBoundaryRules> rules; // Создаётся при первой строке superin, inflow или outflow.
    PeriodicPair periodic{};                  // Имена и перенос, если метка периодическая.
    bool has_type = false;                    // Тип обязателен для всех меток.
    bool has_superin = false, has_inflow = false, has_outflow = false; // Наличие каждой строки free.
    bool has_translation = false; // Нулевой вектор тоже может быть явно записан.

    // При чтении стенки или периодики даже временный набор правил не нужен.
    FreeBoundaryRules& free_rules() {
        if (!rules)
            rules.reset(new FreeBoundaryRules{});
        return *rules;
    }

    // reference_state задаётся один раз на всю секцию независимо от числа использующих его режимов.
    void read_reference(const Tokens& words, size_t first, unsigned dimension, bool linear) {
        auto& reference = free_rules().reference;
        if (reference.mask) throw std::invalid_argument("repeated reference_state state");
        Tokens components;
        bool has_temperature = false;
        double theta = 0.;
        for (size_t i = first; i < words.size(); ++i) {
            if (words[i].compare(0, 2, "T=") != 0) { components.push_back(words[i]); continue; }
            if (has_temperature) throw std::invalid_argument("repeated reference_state T");
            theta = number(words[i].substr(2));
            has_temperature = true;
        }
        read_components(components, 0, dimension, linear, reference);
        if (has_temperature) {
            if ((reference.mask & 3u) != 2u) throw std::invalid_argument("reference_state T requires p and replaces rho");
            reference.primitive[0] = temperature.density(reference.primitive[1], theta, linear);
            reference.mask |= 1u;
        }
        if (!reference.mask) throw std::invalid_argument("empty reference_state state");
    }

    // Назначает единственную строку секции. Дубликаты являются ошибкой даже при одинаковых значениях.
    void read_line(const Tokens& words, unsigned dimension, bool linear) {
        const auto& key = words.front();
        if (key == "bc_type" && !has_type) {
            type = read_type(words);
            has_type = true;
        } else if (key == "reference_state") {
            read_reference(words, 1, dimension, linear);
        } else if (key == "superin" && !has_superin) {
            auto& inlet = free_rules().superin;
            if (words.size() == 2 && words[1] == "profile") inlet.source = BoundarySource::Profile;
            else if (words.size() == 2 && words[1] == "p_interior") inlet.source = BoundarySource::InteriorPressure;
            else if ((words.size() > 1 && words[1] == "reference_state") || (words.size() == 2 && words[1] == "external")) {
                inlet.source = BoundarySource::Reference;
                if (words.size() > 2) read_reference(words, 2, dimension, linear);
            } else {
                inlet.use_initial = words.size() > 1 && words[1] == "initial";
                read_components(words, inlet.use_initial ? 2 : 1, dimension, linear, inlet);
            }
            has_superin = true;
        } else if (key == "inflow" && !has_inflow) {
            free_rules().inflow = read_subsonic_rule(words, linear);
            has_inflow = true;
        } else if (key == "outflow" && !has_outflow) {
            free_rules().outflow = read_subsonic_rule(words, linear);
            has_outflow = true;
        } else if (key == "patch" && words.size() == 2 && periodic.patch.empty()) {
            periodic.patch = words[1];
        } else if (key == "pair" && words.size() == 2 && periodic.pair.empty()) {
            periodic.pair = words[1];
        } else if (key == "translation" && words.size() == 4 && !has_translation) {
            for (unsigned k = 0; k < 3; ++k)
                periodic.translation[k] = number(words[k + 1]);
            has_translation = true;
        } else {
            throw std::invalid_argument("invalid or repeated boundary setting: " + key);
        }
    }

    // Проверяем комплектность после последней строки, когда известны тип и все параметры.
    void validate(unsigned dimension) const {
        if (!has_type)
            throw std::invalid_argument("missing bc_type");
        if (type == BoundaryType::Free) {
            if (!has_superin)
                throw std::invalid_argument("free boundary is missing superin");
            if (!has_inflow)
                throw std::invalid_argument("free boundary is missing inflow");
            if (!has_outflow)
                throw std::invalid_argument("free boundary is missing outflow");
            if (rules->reference.mask) {
                unsigned needed = (1u << (dimension + 2)) - 1;
                // Проверяет, требует ли граничное правило явного задания давления.
                const auto needs_pressure = [](BoundarySource s) {
                    return takes_reference(s) && s != BoundarySource::InteriorPressure;
                };
                if (!needs_pressure(rules->superin.source) && !needs_pressure(rules->inflow.source) && !needs_pressure(rules->outflow.source))
                    needed &= ~2u;
                if ((rules->reference.mask & needed) != needed)
                    throw std::invalid_argument("incomplete explicit reference_state: specify density, velocity components and required pressure");
            }
        } else if (rules) {
            throw std::invalid_argument("superin, inflow and outflow require bc_type free");
        }
        if (type == BoundaryType::Periodic) {
            if (!has_translation || periodic.patch.empty() || periodic.pair.empty() ||
                periodic.patch == periodic.pair)
                throw std::invalid_argument(
                    "periodic requires different patch and pair names and translation x y z");
        } else if (has_translation || !periodic.patch.empty() || !periodic.pair.empty()) {
            throw std::invalid_argument("periodic keywords on a nonperiodic boundary");
        }
    }
};
} // namespace

// Читает одну секцию bc.txt и возвращает разобранную пару вызывающему коду.
BoundaryConfig read_boundary(std::istream& input, unsigned dimension, BoundaryPatches& patches,
                             PeriodicPair& periodic, bool linear, BoundaryTemperature temperature) {
    if (dimension < 1 || dimension > 3)
        throw std::invalid_argument("invalid boundary dimension");
    BoundarySection section;
    section.temperature = temperature;
    std::string line;
    size_t row = 0;
    while (std::getline(input, line)) {
        ++row;
        // Всё после # — комментарий, в том числе в конце строки с правилом.
        std::istringstream words(line.substr(0, line.find('#')));
        Tokens tokens;
        for (std::string word; words >> word;)
            tokens.push_back(word);
        if (tokens.empty())
            continue;
        try {
            section.read_line(tokens, dimension, linear);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("boundary line " + std::to_string(row) + ": " + e.what());
        }
    }
    if (input.bad() || !input.eof())
        throw std::runtime_error("boundary settings read failed");
    section.validate(dimension);
    periodic = section.periodic;
    // Стенка и периодическая пара занимают только запись метки. Для них комплект правил не создаётся.
    if (section.type == BoundaryType::Free)
        return patches.add_free(*section.rules);
    if (section.type == BoundaryType::SlipWall) return BoundaryConfig::slip_wall();
    if (section.type == BoundaryType::Periodic)
        return BoundaryConfig::periodic();
    return BoundaryConfig{};
}

// Сокращённый вызов для физической границы, которой не требуется парная метка.
BoundaryConfig read_boundary(std::istream& input, unsigned dimension, BoundaryPatches& patches) {
    PeriodicPair spec;
    BoundaryConfig result = read_boundary(input, dimension, patches, spec);
    if (result.type() == BoundaryType::Periodic)
        throw std::invalid_argument("read_boundary requires a PeriodicPair output for periodic settings");
    return result;
}
}
