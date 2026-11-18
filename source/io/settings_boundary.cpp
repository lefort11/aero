#include "settings_boundary.h"
#include "settings_text.h"
#include "../mesh/mesh.h"
#include <map>
#include <sstream>

namespace aero {
namespace {
using namespace settings_detail;
struct Section {
    std::string patch, pair, where, rules;
};

// The single-patch physics parser uses single-token names. Substitute its two
// local identifiers, then restore the exact mesh labels from the section.
void add_rule(Section& section, const std::string& line, const std::string& where) {
    const auto split = line.find_first_of(" \t");
    const std::string key = line.substr(0, split);
    if (key == "patch") throw std::invalid_argument(where + ": patch is given by the section header");
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

CaseBoundaries read_case_boundaries(const std::string& path, const MeshData& mesh) {
    auto input = settings_detail::open(path);
    CaseBoundaries result;
    result.by_patch.resize(mesh.patch_names.size());
    std::map<std::string, std::size_t> unassigned;
    for (std::size_t i = 0; i < mesh.patch_names.size(); ++i)
        if (!unassigned.emplace(mesh.patch_names[i], i).second)
            throw std::invalid_argument(path + ": ambiguous mesh label '" + mesh.patch_names[i] + "'");
    Section section;
    auto assign = [&](const std::string& name, const BoundaryConfig& config) {
        const auto it = unassigned.find(name);
        if (it == unassigned.end())
            throw std::invalid_argument(section.where + ": unknown or already assigned mesh label '" + name + "'");
        result.by_patch[it->second] = config;
        unassigned.erase(it);
    };
    auto finish = [&]() {
        if (section.patch.empty()) return;
        std::istringstream rules(section.rules);
        PeriodicPair pair{};
        BoundaryConfig config;
        try {
            config = read_boundary(rules, mesh.dimension, pair);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(section.where + " [" + section.patch + "]: " + e.what());
        }
        assign(section.patch, config);
        if (config.type == BoundaryType::Periodic) {
            assign(section.pair, config);
            pair.patch = section.patch;
            pair.pair = section.pair;
            result.periodic_pairs.push_back(pair);
        }
    };
    std::string line;
    std::size_t row = 0;
    while (std::getline(input, line)) {
        const std::string where = path + ":" + std::to_string(++row);
        line = settings_detail::clean(line, where, row);
        if (line.empty()) continue;
        if (line.front() == '[') {
            finish();
            if (line.back() != ']') throw std::invalid_argument(where + ": expected [mesh label]");
            section = Section();
            section.patch = settings_detail::label(settings_detail::trim(line.substr(1, line.size()-2)), where);
            section.where = where;
        } else {
            if (section.patch.empty()) throw std::invalid_argument(where + ": expected [mesh label] before rules");
            add_rule(section, line, where);
        }
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("error reading " + path);
    finish();
    if (!unassigned.empty())
        throw std::invalid_argument(path + ": missing boundary for mesh label '" + unassigned.begin()->first + "'");
    return result;
}
}
