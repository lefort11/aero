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
// Выбирает физическую или диагностическую величину для столбца вывода.
Field output_field(const std::string& name) {
    Field f;
    f.name = name;
    f.location = FieldLocation::Node;
    f.components = {"value"};
    return f;
}

// Связывает расписание зон с геометрией и полями для записи снимков.
ScheduledOutput::ScheduledOutput(const std::vector<OutputZone>& zones, const std::string& directory,
                                 const Mesh& mesh, double gamma, std::vector<std::uint64_t> node_dof,
                                 std::size_t state_nodes)
    : OutputSchedule(zones, mesh.data().dimension) {
    if (!zones.empty())
        local_ = LocalFieldOutput(directory, mesh, gamma, std::move(node_dof), state_nodes);
}
// Подготавливает локальную запись полей с учётом нумерации узлов.
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
// Создаёт расписание зон и выбирает их столбцы по размерности задачи.
OutputSchedule::OutputSchedule(const std::vector<OutputZone>& zones, unsigned dimension) {
    validate_output_zones(zones);
    for (const auto& config : zones) {
        Zone z;
        z.config = config;
        z.config.variables.clear();
        // Добавляет выбранное поле в набор данных записываемой зоны.
        const auto append=[&](unsigned c) {
            const unsigned axis=c>=diagnostic::DensityPrime && c<=diagnostic::EnergyPrime ? c-diagnostic::DensityPrime : c;
            if(axis<=3 && axis>dimension) throw FieldError("["+config.name+"] "+diagnostic::name(c)+" exceeds mesh dimension");
            if(std::find(z.columns.begin(),z.columns.end(),c)!=z.columns.end())
                throw FieldError("["+config.name+"] duplicate expanded column "+diagnostic::name(c));
            z.columns.push_back(c); z.config.variables.emplace_back(diagnostic::name(c));
        };
        for(const auto& key:config.variables) {
            if(key=="primary" || key=="fluctuations") {
                const unsigned start=key=="primary"?diagnostic::Density:diagnostic::DensityPrime;
                for(unsigned c=0;c<6;++c) if(c<=dimension || c>=4) append(start+c);
            } else if(key=="turbulence") {append(diagnostic::WorkingNu);append(diagnostic::EddyNu);}
            else append(diagnostic::column(key));
        }
        zones_.push_back(z);
    }
}
// Вычисляет ближайшее физическое время записи зоны по целочисленному номеру срока.
double OutputSchedule::Zone::deadline() const {
    return static_cast<double>(next_tick) * config.every_time;
}
// Проверяет, наступил ли шаг или физическое время записи зоны.
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
// Переносит расписание за восстановленные шаг и время, исключая повтор старых снимков.
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
// Ограничивает шаг ближайшим сроком записи активной зоны.
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
// Проверяет, наступил ли шаг или физическое время записи зоны.
bool OutputSchedule::due(std::uint64_t step, double time) const {
    if (!(time >= 0 && std::isfinite(time)))
        throw FieldError("output: invalid accepted time");
    bool any = false;
    for (const auto& zone : zones_)
        if (zone.due(step, time))
            any = true;
    return any;
}
// Проверяет, понадобится ли столбец хотя бы одной зоне именно в этом снимке.
bool OutputSchedule::column_due(unsigned column, std::uint64_t step, double time, bool final_field) const {
    for (const auto& zone : zones_)
        if ((final_field || zone.due(step, time)) &&
            std::find(zone.columns.begin(), zone.columns.end(), column) != zone.columns.end())
            return true;
    return false;
}
// Формирует выбранные поля и записывает снимок зоны в нужном формате.
void ScheduledOutput::write(std::uint64_t step, double time, const std::vector<double>& state) {
    write_due(step, time, [&](const OutputZone& zone, const std::vector<unsigned>& columns) {
        local_.write(zone, columns, step, time, state);
    });
}
// Записывает зоны, срок которых наступил на принятом шаге.
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
// Формирует выбранные поля и записывает снимок зоны в нужном формате.
void LocalFieldOutput::write(const OutputZone& zone, const std::vector<unsigned>& columns, std::uint64_t step,
                             double time, const std::vector<double>& state,const diagnostic::Fields& scalars) const {
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
    const FieldRows rows{state.data(), snap.dimension, gamma_, background_, scalars, columns};
    parallel_for(count, [&](std::size_t i) {
        const auto node = node_dof_.empty() ? i : static_cast<std::size_t>(node_dof_[i]);
        rows.visit(node, node, [&](std::size_t k, double value) { snap.fields[k].values[i] = value; });
    });
    auto name = output_filename(zone, step);
    std::replace(name.begin(), name.end(), '\\', '/');
    StagedFieldFile file(directory_ + "/" + name);
    snap.write(file.temporary(), zone.format);
    file.publish(!zone.multi);
}
}
