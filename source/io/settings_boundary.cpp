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
struct Section {
    std::vector<std::string> patches;
    std::string pair, where, rules, profile_path;
    BoundaryProfile profile;
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
    bool header = false;
    while (std::getline(input,line)) {
        ++row;
        const auto where = path + ":" + std::to_string(row);
        std::istringstream in(clean(line,where,row));
        std::vector<std::string> words;
        for (std::string word; in >> word;) words.push_back(word);
        if (words.empty()) continue;
        if (!header) {
            if (words != std::vector<std::string>{"R_yz","ro","u_x","u_r","p","nutilde"})
                throw std::invalid_argument(where + ": expected R_yz ro u_x u_r p nutilde");
            header = true;
            continue;
        }
        if (words.size()!=6) throw std::invalid_argument(where + ": expected six values");
        double v[6];
        for (unsigned k=0;k<6;++k) v[k]=finite_real(words[k],where);
        auto& rows = section.profile.rows;
        if (v[0]<0. || (!rows.empty() && rows.back().radius>=v[0]) || v[1]<=0. || v[4]<=0. || v[5]<0.)
            throw std::invalid_argument(where + ": invalid radius, density, pressure or nutilde");
        rows.push_back({v[0],v[1],v[2],v[3],v[4],v[5]});
    }
    if (input.bad() || section.profile.rows.empty()) throw std::invalid_argument(path + ": incomplete profile");
}

// Разборщик условий одного участка принимает однословные имена. Подставляем локальные идентификаторы,
// затем возвращаем точные имена участков из заголовков секций.
void add_rule(Section& section, const std::string& line, const std::string& where) {
    const auto split = line.find_first_of(" \t");
    const std::string key = line.substr(0, split);
    if (key == "profile") {
        if (split == std::string::npos || !section.profile_path.empty()) throw std::invalid_argument(where + ": expected one profile file");
        section.profile_path = label(trim(line.substr(split)),where);
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

CaseBoundaries read_case_boundaries(const std::string& path, const MeshData& mesh, bool linear, std::vector<std::string>* warnings) {
    auto input = settings_detail::open(path);
    CaseBoundaries result;
    result.patches.by_patch.resize(mesh.patch_names.size());
    std::map<std::string, std::size_t> unassigned;
    for (std::size_t i = 0; i < mesh.patch_names.size(); ++i)
        if (!unassigned.emplace(mesh.patch_names[i], i).second)
            throw std::invalid_argument(path + ": ambiguous mesh label '" + mesh.patch_names[i] + "'");
    Section section;
    auto assign = [&](const std::string& name, const BoundaryConfig& config) {
        const auto it = unassigned.find(name);
        if (it == unassigned.end())
            throw std::invalid_argument(section.where + ": unknown or already assigned mesh label '" + name +
                                        "'");
        result.patches.by_patch[it->second] = config;
        unassigned.erase(it);
    };
    auto finish = [&]() {
        if (section.patches.empty())
            return;
        std::istringstream rules(section.rules);
        PeriodicPair pair{};
        BoundaryConfig config;
        try {
            config = read_boundary(rules, mesh.dimension, result.patches, pair, linear);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(section.where + " [" + section.patches.front() + "]: " + e.what());
        }
        const auto* rule = config.rules(result.patches.free_rules.data());
        const bool uses_profile = rule && (rule->superin.profile || rule->inflow.source==BoundarySource::Profile ||
                                            rule->outflow.source==BoundarySource::Profile);
        if (uses_profile || !section.profile_path.empty() || section.has_axis || section.has_center) {
            if (linear || !rule || section.profile_path.empty() || !rule->superin.profile ||
                rule->inflow.source!=BoundarySource::Profile || rule->outflow.source!=BoundarySource::Profile ||
                rule->outflow.mode!=BoundaryMode::Pressure)
                throw std::invalid_argument(section.where + ": Euler free profile requires profile FILE, superin profile, inflow profile and outflow base_p profile");
            read_profile(section,path);
            result.patches.profiles.resize(result.patches.free_rules.size());
            result.patches.profiles.back() = std::move(section.profile);
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
                key != "superin" && key != "inflow" && key != "outflow") {
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
