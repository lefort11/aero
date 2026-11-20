#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <cstdlib>
#endif
#include "scheduled_output.h"
#include "../mesh/mesh.h"
#include "../eulersolver/euler.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

namespace aero {
StagedFieldFile::StagedFieldFile(const std::string& path) : path_(path) {
    const auto directory = path.substr(0, path.find_last_of("/\\") + 1);
#ifdef _WIN32
    char temp[MAX_PATH];
    if (!GetTempFileNameA(directory.c_str(), "aer", 0, temp))
        throw FieldError("Cannot create temporary file for " + path);
    temporary_ = temp;
#else
    std::string pattern = directory + ".aero-fields-XXXXXX";
    const int fd = mkstemp(&pattern[0]);
    if (fd < 0)
        throw FieldError("Cannot create temporary file for " + path);
    temporary_ = pattern;
    if (close(fd) != 0) {
        std::remove(temporary_.c_str());
        throw FieldError("Cannot close temporary file for " + path);
    }
#endif
}
StagedFieldFile::~StagedFieldFile() {
    if (!temporary_.empty())
        std::remove(temporary_.c_str());
}
void StagedFieldFile::publish(bool replace) {
#ifdef _WIN32
    if (!MoveFileExA(temporary_.c_str(), path_.c_str(), replace ? MOVEFILE_REPLACE_EXISTING : 0))
        throw FieldError("Cannot publish " + path_ + " (Windows error " + std::to_string(GetLastError()) +
                         ")");
#else
    if (replace) {
        if (std::rename(temporary_.c_str(), path_.c_str()) != 0)
            throw FieldError("Cannot replace " + path_);
    } else {
        if (link(temporary_.c_str(), path_.c_str()) != 0)
            throw FieldError("Cannot publish " + path_);
        if (std::remove(temporary_.c_str()) != 0)
            return; // Деструктор повторит попытку удалить временный файл.
    }
#endif
    temporary_.clear();
}

Field output_field(const std::string& name) {
    Field f;
    f.name = name;
    f.location = FieldLocation::Node;
    f.components = {"value"};
    return f;
}

ScheduledOutput::ScheduledOutput(const std::vector<OutputZone>& zones, const std::string& directory,
                                 const Mesh& mesh, double gamma, std::vector<std::uint64_t> node_dof,
                                 std::size_t state_nodes)
    : OutputSchedule(zones, mesh.data().dimension) {
    if (!zones.empty())
        local_ = LocalFieldOutput(directory, mesh, gamma, std::move(node_dof), state_nodes);
}
LocalFieldOutput::LocalFieldOutput(const std::string& directory, const Mesh& mesh, double gamma,
                                   std::vector<std::uint64_t> node_dof, std::size_t state_nodes, const LinearBackground& background)
    : directory_(directory), gamma_(gamma), background_(background) {
    if (directory.empty() || !std::isfinite(gamma) || gamma <= 1)
        throw std::invalid_argument("output: invalid directory or gamma");
    metadata_ = FieldSnapshot(mesh);
    const auto count = metadata_.counts[0];
    const unsigned d = metadata_.dimension;
    const auto limit = std::vector<double>().max_size() / (d + 2);
    if (count > limit || state_nodes > limit)
        throw FieldError("output: too many nodes");
    nodes_ = static_cast<std::size_t>(count);
    if (!node_dof.empty()) {
        if (node_dof.size() != nodes_)
            throw FieldError("output: wrong node map length");
        const auto maximum = *std::max_element(node_dof.begin(), node_dof.end());
        // Массив состояния раздела может также содержать узлы соседей, отсутствующие в его выходной сетке.
        if (maximum >= (state_nodes ? state_nodes : count))
            throw FieldError("output: invalid node map");
        nodes_ = state_nodes ? state_nodes : static_cast<std::size_t>(maximum + 1);
        node_dof_ = std::move(node_dof);
    } else if (state_nodes && state_nodes != nodes_)
        throw FieldError("output: missing node map");
}
OutputSchedule::OutputSchedule(const std::vector<OutputZone>& zones, unsigned dimension) {
    validate_output_zones(zones);
    const std::vector<std::string> names = {"rho", "ux", "uy", "uz", "p", "E"};
    for (const auto& config : zones) {
        Zone z;
        z.config = config;
        for (const auto& var : config.variables) {
            const auto column =
                static_cast<unsigned>(std::find(names.begin(), names.end(), var) - names.begin());
            if (column <= 3 && column > dimension)
                throw FieldError("[" + config.name + "] " + var + " exceeds mesh dimension");
            z.columns.push_back(column);
        }
        zones_.push_back(z);
    }
}
double OutputSchedule::Zone::deadline() const {
    return static_cast<double>(next_tick) * config.every_time;
}
bool OutputSchedule::Zone::due(std::uint64_t step, double time) const {
    if (step <= last_step)
        return false;
    if (config.every_steps)
        return step % config.every_steps == 0;
    const double target = deadline();
    if (!std::isfinite(target))
        return false;
    // Объединяем сроки, отличающиеся лишь округлением, например 3 × 0.1 и 0.3.
    const double tolerance =
        std::min(config.every_time * 0.25, 4 * std::numeric_limits<double>::epsilon() * target);
    if (time - target > tolerance)
        throw FieldError("[" + config.name + "] missed output time; use limit_dt before stepping");
    return target - time <= tolerance;
}
void OutputSchedule::resume(std::uint64_t step, double time) {
    for (auto& zone : zones_) {
        zone.last_step = step;
        if (!zone.config.every_time)
            continue;
        const double passed = std::floor(time / zone.config.every_time);
        const double limit = static_cast<double>(std::numeric_limits<std::uint64_t>::max());
        if (!(passed >= 0 && passed < limit))
            throw FieldError("output: restored time exceeds schedule range");
        zone.next_tick = static_cast<std::uint64_t>(passed) + 1;
        const double target = zone.deadline();
        const double tolerance =
            std::min(zone.config.every_time * 0.25, 4 * std::numeric_limits<double>::epsilon() * target);
        if (std::isfinite(target) && target - time <= tolerance)
            ++zone.next_tick;
        if (!(zone.deadline() > time))
            throw FieldError("output: restored deadline cannot advance");
    }
}
double OutputSchedule::limit_dt(double time, double candidate) const {
    if (!(time >= 0 && std::isfinite(time) && candidate > 0 && std::isfinite(candidate)))
        throw FieldError("output: invalid time or proposed dt");
    for (const auto& zone : zones_)
        if (zone.config.every_time > 0) {
            const double remaining = zone.deadline() - time;
            if (!(remaining > 0))
                throw FieldError("[" + zone.config.name + "] pending output at current time");
            candidate = std::min(candidate, remaining);
        }
    return candidate;
}
bool OutputSchedule::due(std::uint64_t step, double time) const {
    if (!(time >= 0 && std::isfinite(time)))
        throw FieldError("output: invalid accepted time");
    bool any = false;
    for (const auto& zone : zones_)
        if (zone.due(step, time))
            any = true;
    return any;
}
void ScheduledOutput::write(std::uint64_t step, double time, const std::vector<double>& state) {
    write_due(step, time, [&](const OutputZone& zone, const std::vector<unsigned>& columns) {
        local_.write(zone, columns, step, time, state);
    });
}
void OutputSchedule::write_due(std::uint64_t step, double time, const Writer& writer, bool final_field) {
    if (!final_field && !due(step, time))
        return;
    for (auto& zone : zones_)
        if (final_field || zone.due(step, time)) {
            writer(zone.config, zone.columns);
            zone.last_step = step;
            if (!final_field && zone.config.every_time > 0)
                ++zone.next_tick;
        }
}
void LocalFieldOutput::write(const OutputZone& zone, const std::vector<unsigned>& columns, std::uint64_t step,
                             double time, const std::vector<double>& state) const {
    const unsigned width = metadata_.dimension + 2;
    if (state.size() != nodes_ * width)
        throw FieldError("output: incorrect conservative state size");
    auto snap = metadata_;
    snap.step = step;
    snap.time = time;
    const auto count = static_cast<std::size_t>(snap.counts[0]);
    for (const auto& name : zone.variables) {
        snap.fields.push_back(output_field(name));
        snap.fields.back().values.resize(count);
    }
    parallel_for(count, [&](std::size_t i) {
        const auto node = node_dof_.empty() ? i : static_cast<std::size_t>(node_dof_[i]);
        const auto values = output_values(snap.dimension, state.data() + node * width, gamma_, node, background_);
        for (std::size_t k = 0; k < columns.size(); ++k)
            snap.fields[k].values[i] = values[columns[k]];
    });
    auto name = output_filename(zone, step);
    std::replace(name.begin(), name.end(), '\\', '/');
    StagedFieldFile file(directory_ + "/" + name);
    snap.write(file.temporary(), zone.format);
    file.publish(!zone.multi);
}
}
