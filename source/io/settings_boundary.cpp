#include "settings_boundary.h"
#include "settings_text.h"
#include "../mesh/mesh.h"
#include <map>
#include <algorithm>
#include <array>
#include <sstream>

namespace aero {
namespace {
using namespace settings_detail;
// Промежуточные правила, участки и профиль одной секции граничного файла.
struct Section {
    // Имена патчей, к которым применяется секция.
    std::vector<std::string> patches;
    // Имя пары, место определения, правила и путь к профилю.
    std::string pair, where, rules, profile_path;
    // Разобранный радиальный профиль секции.
    BoundaryProfile profile;
    // Признак явного задания внешних величин.
    bool external_set = false;
    // Отмечают явно заданные ось и центр профиля.
    bool has_axis = false, has_center = false;
};

// Запятая разделяет метки только вне кавычек. Кавычки внутри имени
// и пустые имена проверяет общий разборщик меток.
std::vector<std::string> patch_names(const std::string& text, const std::string& where) {
    std::vector<std::string> names;
    std::size_t begin = 0;
    bool quoted = false;
    for (std::size_t pos = 0; pos < text.size(); ++pos) {
        if (text[pos] == '"') quoted = !quoted;
        if (text[pos] == ',' && !quoted) {
            names.push_back(label(trim(text.substr(begin, pos - begin)), where));
            begin = pos + 1;
        }
    }
    names.push_back(label(trim(text.substr(begin)), where));
    return names;
}

// Читаем профиль один раз, пока открывается bc.txt. Координаты и газовые
// величины даны в тех же единицах, что в сетке и начальных условиях.
void read_profile(Section& section, const std::string& bc_path) {
    std::string path = section.profile_path;
    if (path.front() != '/' && path.front() != '\\' && !(path.size() > 1 && path[1] == ':'))
        path = bc_path.substr(0,bc_path.find_last_of("/\\")+1) + path;
    auto input = open(path);
    std::string line;
    std::size_t row = 0;
    std::vector<unsigned> columns;
    auto& mask = section.profile.columns;
    mask = 0;
    while (std::getline(input,line)) {
        ++row;
        const auto where = path + ":" + std::to_string(row);
        std::istringstream in(clean(line,where,row));
        std::vector<std::string> words;
        for (std::string word; in >> word;) words.push_back(word);
        if (words.empty()) continue;
        if (columns.empty()) {
            const std::vector<std::string> names{"R_yz","ro","u_x","u_r","p","nutilde"};
            for (const auto& word : words) {
                const auto it = std::find(names.begin(), names.end(), word);
                if (it == names.end()) throw std::invalid_argument(where + ": unknown profile column " + word);
                const unsigned index = static_cast<unsigned>(it - names.begin());
                if (mask & (1u << index)) throw std::invalid_argument(where + ": duplicate profile column " + word);
                mask |= 1u << index;
                columns.push_back(index);
            }
            if (!(mask & Radius) || mask == Radius) throw std::invalid_argument(where + ": expected R_yz and field columns");
            continue;
        }
        if (words.size()!=columns.size()) throw std::invalid_argument(where + ": row size differs from profile header");
        double v[6]{};
        for (size_t k=0;k<columns.size();++k) v[columns[k]]=finite_real(words[k],where);
        auto& rows = section.profile.rows;
        if (v[0]<0. || (!rows.empty() && rows.back().radius>=v[0]) || ((mask & ProfileRho) && v[1]<=0.) ||
            ((mask & ProfileP) && v[4]<=0.) || v[5]<0.)
            throw std::invalid_argument(where + ": invalid radius, density, pressure or nutilde");
        rows.push_back({v[0],v[1],v[2],v[3],v[4],v[5]});
    }
    if (input.bad() || section.profile.rows.empty()) throw std::invalid_argument(path + ": incomplete profile");
}

// Разборщик условий одного участка принимает однословные имена. Подставляем локальные идентификаторы,
// затем возвращаем точные имена участков из заголовков секций.
void add_rule(Section& section, const std::string& line, const std::string& where) {
    const auto split = line.find_first_of(" \t=");
    const std::string key = line.substr(0, split);
    if (key == "profile" || key == "external" || key == "profile_ref" || key == "profile_init" || key == "profile_interior") {
        std::string value = split == std::string::npos ? "" : trim(line.substr(split));
        if (!value.empty() && value[0] == '=') value = trim(value.substr(1));
        if (key != "profile") {
            if (section.external_set) throw std::invalid_argument(where + ": repeated external source");
            section.external_set = true;
            const auto source = key == "external" ? value : key.substr(8);
            if (source == "reference_state" || source == "ref") section.profile.fallback = ProfileFallback::Reference;
            else if (source == "initial" || source == "init") section.profile.fallback = ProfileFallback::Initial;
            else if (source == "interior") section.profile.fallback = ProfileFallback::Interior;
            else throw std::invalid_argument(where + ": expected external = reference_state, initial or interior");
            if (key == "external") return;
        }
        if (split == std::string::npos || !section.profile_path.empty()) throw std::invalid_argument(where + ": expected one profile file");
        section.profile_path = label(value,where);
        return;
    }
    if (key == "profile_axis" || key == "profile_center") {
        const bool axis = key == "profile_axis";
        bool& seen = axis ? section.has_axis : section.has_center;
        if (seen) throw std::invalid_argument(where + ": duplicate " + key);
        seen = true;
        auto& vector = axis ? section.profile.axis : section.profile.center;
        std::istringstream in(line);
        std::string token;
        in >> token;
        for (double& x : vector) {
            if (!(in >> token)) throw std::invalid_argument(where + ": expected three coordinates");
            x = finite_real(token,where);
        }
        if (in >> token) throw std::invalid_argument(where + ": unexpected coordinate");
        if (axis) {
            const double length = coordinate_length(vector);
            if (!(length>0.) || !std::isfinite(length)) throw std::invalid_argument(where + ": invalid profile axis");
            for (double& x : vector) x /= length;
        }
        return;
    }
    if (key == "patch")
        throw std::invalid_argument(where + ": patch is given by the section header");
    if (key == "pair") {
        if (!section.pair.empty() || split == std::string::npos)
            throw std::invalid_argument(where + ": expected one periodic partner");
        section.pair = label(trim(line.substr(split)), where);
        section.rules += "pair partner\npatch owner\n";
    } else {
        section.rules += line + "\n";
    }
}
}

// Связывает граничные секции постановки с участками загруженной сетки.
CaseBoundaries read_case_boundaries(const std::string& path, const MeshData& mesh, bool linear,
                                   std::vector<std::string>* warnings, BoundaryTemperature temperature) {
    auto input = settings_detail::open(path);
    CaseBoundaries result;
    result.patches.by_patch.resize(mesh.patch_names.size());
    std::map<std::string, std::size_t> unassigned;
    for (std::size_t i = 0; i < mesh.patch_names.size(); ++i)
        if (!unassigned.emplace(mesh.patch_names[i], i).second)
            throw std::invalid_argument(path + ": ambiguous mesh label '" + mesh.patch_names[i] + "'");
    Section section;
    // Связывает разобранные граничные правила с соответствующими участками.
    auto assign = [&](const std::string& name, const BoundaryConfig& config) {
        const auto it = unassigned.find(name);
        if (it == unassigned.end())
            throw std::invalid_argument(section.where + ": unknown or already assigned mesh label '" + name +
                                        "'");
        result.patches.by_patch[it->second] = config;
        unassigned.erase(it);
    };
    // Завершает разбор текущей граничной секции и проверяет её полноту.
    auto finish = [&]() {
        if (section.patches.empty())
            return;
        std::istringstream rules(section.rules);
        PeriodicPair pair{};
        BoundaryConfig config;
        try {
            config = read_boundary(rules, mesh.dimension, result.patches, pair, linear, temperature);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(section.where + " [" + section.patches.front() + "]: " + e.what());
        }
        auto* rule = config.type() == BoundaryType::Free ? &result.patches.free_rules.back() : nullptr;
        const bool table = !section.profile_path.empty();
        const bool dynamic = section.profile.fallback == ProfileFallback::Interior || section.profile.fallback == ProfileFallback::Initial;
        if (rule) rule->external_base = section.profile.fallback;
        if (table || (dynamic && !linear) || section.has_axis || section.has_center) {
            if (!rule || linear || (!table && (section.has_axis || section.has_center)))
                throw std::invalid_argument(section.where + ": profile requires an Euler free boundary");
            if (rule->outflow.source == BoundarySource::Profile && rule->outflow.mode != BoundaryMode::Pressure)
                throw std::invalid_argument(section.where + ": profile outflow requires base_p profile or reference_state");
            if (!rule->uses_reference() && (rule->superin.source != BoundarySource::Profile || rule->inflow.source != BoundarySource::Profile ||
                                            rule->outflow.source != BoundarySource::Profile))
                throw std::invalid_argument(section.where + ": incomplete free boundary profile rules");
            if (table) read_profile(section,path);
            else section.profile.columns = Radius; // Взятое целиком начальное или внутреннее состояние.
            if (section.profile.fallback == ProfileFallback::Full) section.profile.fallback = ProfileFallback::Reference;
            result.patches.profiles.resize(result.patches.free_rules.size());
            result.patches.profiles.back() = std::move(section.profile);
        } else if (rule && (rule->superin.source == BoundarySource::Profile || rule->inflow.source == BoundarySource::Profile ||
                            rule->outflow.source == BoundarySource::Profile)) {
            throw std::invalid_argument(section.where + ": missing profile file");
        }
        if (section.external_set && !rule) throw std::invalid_argument(section.where + ": external requires free");
        // Полная таблица и динамические источники обходятся без набегающего газа.
        const bool needs_ref = table ? !dynamic && (section.profile.columns & GasColumns) != GasColumns :
                                       !dynamic && rule && rule->uses_reference();
        if (needs_ref && rule && !rule->reference.mask) {
            const auto q = temperature.gas.primitive(mesh.dimension);
            auto& ref = rule->reference;
            ref.mask = (1u << (mesh.dimension+2)) - 1;
            ref.primitive[0] = q.front() - (linear ? temperature.background_rho : 0.);
            ref.primitive[1] = q.back() - (linear ? temperature.background_p : 0.);
            for (unsigned j=0;j<mesh.dimension;++j)
                ref.primitive[2+j] = q[1+j] - (linear ? temperature.background_velocity[j] : 0.);
        }
        // Все метки free ссылаются на один уже разобранный комплект правил.
        for (const auto& patch : section.patches) assign(patch, config);
        if (config.type() == BoundaryType::Periodic) {
            if (section.patches.size() != 1)
                throw std::invalid_argument(section.where + ": periodic requires one patch and one partner");
            assign(section.pair, config);
            pair.patch = section.patches.front();
            pair.pair = section.pair;
            result.periodic_pairs.push_back(pair);
        }
    };
    std::string line;
    std::size_t row = 0;
    while (std::getline(input, line)) {
        const std::string where = path + ":" + std::to_string(++row);
        line = settings_detail::clean(line, where, row);
        if (line.empty())
            continue;
        if (line.front() == '[') {
            finish();
            if (line.back() != ']')
                throw std::invalid_argument(where + ": expected [mesh label]");
            section = Section();
            section.patches = patch_names(line.substr(1, line.size() - 2), where);
            section.where = where;
        } else {
            if (section.patches.empty())
                throw std::invalid_argument(where + ": expected [mesh label] before rules");
            const auto key = line.substr(0, line.find_first_of(" \t="));
            if (warnings && key != "profile" && key != "profile_axis" && key != "profile_center" && key != "bc_type" && key != "pair" && key != "patch" && key != "translation" &&
                key != "superin" && key != "inflow" && key != "outflow" && key != "reference_state" && key != "external" &&
                key != "profile_ref" && key != "profile_interior" && key != "profile_init") {
                warnings->push_back(where + ": unknown parameter " + key);
            } else add_rule(section, line, where);
        }
    }
    if (input.bad() || !input.eof())
        throw std::runtime_error("error reading " + path);
    finish();
    if (!unassigned.empty())
        throw std::invalid_argument(path + ": missing boundary for mesh label '" + unassigned.begin()->first +
                                    "'");
    return result;
}
}
