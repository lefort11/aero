#include "bc.h"
#include "../io/settings_text.h"

#include <sstream>
#include <locale>
#include <memory>
#include <string>

namespace aero {
namespace {
using Tokens = std::vector<std::string>;

// Разбирает весь токен в локали с десятичной точкой и отвергает NaN и бесконечность.
double number(const std::string& token) {
    return settings_detail::finite_real(token, "boundary value " + token);
}

// После superin разрешены начальное состояние и независимые замены его компонент.
SupersonicInflow read_supersonic_inflow(const Tokens& words, unsigned dimension, bool linear) {
    SupersonicInflow rule = {};
    if (words.size() == 2 && words[1] == "profile") { rule.profile = true; return rule; }
    rule.use_initial = words.size() > 1 && words[1] == "initial";
    const char* fields[] = {"rho", "p", "ux", "uy", "uz"};
    for (size_t i = rule.use_initial ? 2 : 1; i < words.size(); ++i) {
        const auto equals = words[i].find('=');
        const auto name = words[i].substr(0, equals);
        // В 1D допустима только ux, в 2D добавляется uy, в 3D также uz.
        unsigned field = 0;
        while (field < 2 + dimension && name != fields[field])
            ++field;
        if (equals == std::string::npos || field == 2 + dimension || (rule.mask & (1u << field)))
            throw std::invalid_argument("invalid or repeated superin field: " + words[i]);
        rule.primitive[field] = number(words[i].substr(equals + 1));
        if (!linear && field < 2 && rule.primitive[field] <= 0)
            throw std::invalid_argument("expected positive " + name);
        rule.mask |= 1u << field; // Остальные компоненты солвер возьмёт из выбранного состояния.
    }
    return rule;
}

// Оба дозвуковых режима используют один синтаксис и одну проверку численного значения.
BoundaryRule read_subsonic_rule(const Tokens& words, bool linear) {
    if (words.size() < 2 || words.size() > 3)
        throw std::invalid_argument("expected " + words[0] + " base_p|base_u|base_rho [initial|number]");
    const auto& name = words[1];
    BoundaryRule rule = {};
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

    // Назначает единственную строку секции. Дубликаты являются ошибкой даже при одинаковых значениях.
    void read_line(const Tokens& words, unsigned dimension, bool linear) {
        const auto& key = words.front();
        if (key == "bc_type" && !has_type) {
            type = read_type(words);
            has_type = true;
        } else if (key == "superin" && !has_superin) {
            free_rules().superin = read_supersonic_inflow(words, dimension, linear);
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
    void validate() const {
        if (!has_type)
            throw std::invalid_argument("missing bc_type");
        if (type == BoundaryType::Free) {
            if (!has_superin)
                throw std::invalid_argument("free boundary is missing superin");
            if (!has_inflow)
                throw std::invalid_argument("free boundary is missing inflow");
            if (!has_outflow)
                throw std::invalid_argument("free boundary is missing outflow");
        } else if (has_superin || has_inflow || has_outflow) {
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
                             PeriodicPair& periodic, bool linear) {
    if (dimension < 1 || dimension > 3)
        throw std::invalid_argument("invalid boundary dimension");
    BoundarySection section;
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
    section.validate();
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
