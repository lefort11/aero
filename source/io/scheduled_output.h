#ifndef AERO_IO_SCHEDULED_OUTPUT_H
#define AERO_IO_SCHEDULED_OUTPUT_H
#include "diagnostic_fields.h"
#include "output_settings.h"
#include "flow_values.h"
#include <functional>

namespace aero {
// Публикует готовый снимок одной операцией. При ошибке сохраняется предыдущий выходной файл.
class StagedFieldFile {
public:
    explicit StagedFieldFile(const std::string& path);
    ~StagedFieldFile();
    StagedFieldFile(const StagedFieldFile&) = delete;
    StagedFieldFile& operator=(const StagedFieldFile&) = delete;
    const std::string& temporary() const { return temporary_; }
    void publish(bool replace);

private:
    std::string path_, temporary_;
};

Field output_field(const std::string& name);

// Все зоны используют общую последовательность принятых шагов и моментов записи.
class OutputSchedule {
public:
    OutputSchedule(const std::vector<OutputZone>& zones, unsigned dimension);
    void resume(std::uint64_t step, double time);
    double limit_dt(double time, double candidate) const;
    bool due(std::uint64_t step, double time) const;
    // Проверяет, понадобится ли столбец хотя бы одной зоне именно в этом снимке.
    bool column_due(unsigned column, std::uint64_t step, double time, bool final_field) const;
    using Writer = std::function<void(const OutputZone&, const std::vector<unsigned>& columns)>;
    bool has_zones() const { return !zones_.empty(); }
    void write_due(std::uint64_t step, double time, const Writer& writer, bool final_field = false);

private:
    struct Zone {
        OutputZone config;
        std::vector<unsigned> columns;
        std::uint64_t next_tick = 1, last_step = 0;
        double deadline() const;
        bool due(std::uint64_t step, double time) const;
    };
    std::vector<Zone> zones_;
};

class LocalFieldOutput {
public:
    LocalFieldOutput() = default;
    LocalFieldOutput(const std::string& directory, const Mesh& mesh, double gamma,
                     std::vector<std::uint64_t> node_dof, std::size_t state_nodes, const LinearBackground& background = {});
    void write(const OutputZone& zone, const std::vector<unsigned>& columns, std::uint64_t step, double time,
               const std::vector<double>& state,const diagnostic::Fields& scalars={}) const;

private:
    std::string directory_;
    FieldSnapshot metadata_;
    std::vector<std::uint64_t> node_dof_;
    std::size_t nodes_ = 0;
    double gamma_ = 0;
    LinearBackground background_;
};

class ScheduledOutput : public OutputSchedule {
public:
    ScheduledOutput(const std::vector<OutputZone>& zones, const std::string& directory, const Mesh& mesh,
                    double gamma, std::vector<std::uint64_t> node_dof = {}, std::size_t state_nodes = 0);
    void write(std::uint64_t step, double time, const std::vector<double>& state);

private:
    LocalFieldOutput local_;
};
}
#endif
