#ifndef AERO_IO_SCHEDULED_OUTPUT_H
#define AERO_IO_SCHEDULED_OUTPUT_H
#include "output_settings.h"
#include <functional>

namespace aero {
// Publish a complete snapshot in one operation, preserving older output on failure.
class StagedFieldFile {
public:
    explicit StagedFieldFile(const std::string& path);
    ~StagedFieldFile();
    StagedFieldFile(const StagedFieldFile&)=delete;
    StagedFieldFile& operator=(const StagedFieldFile&)=delete;
    const std::string& temporary() const { return temporary_; }
    void publish(bool replace);
private:
    std::string path_,temporary_;
};

std::array<double,6> output_values(unsigned dimension,const double* state,double gamma,std::uint64_t node);
Field output_field(const std::string& name);

// All zones share the same accepted steps and time deadlines.
class OutputSchedule {
public:
    OutputSchedule(const std::vector<OutputZone>& zones,unsigned dimension);
    void resume(std::uint64_t step,double time);
    double limit_dt(double time,double candidate) const;
    bool due(std::uint64_t step,double time) const;
    using Writer=std::function<void(const OutputZone&,const std::vector<unsigned>& columns)>;
    void write_due(std::uint64_t step,double time,const Writer& writer);
private:
    struct Zone {
        OutputZone config;
        std::vector<unsigned> columns;
        std::uint64_t next_tick=1, last_step=0;
        double deadline() const;
        bool due(std::uint64_t step,double time) const;
    };
    std::vector<Zone> zones_;
};

class LocalFieldOutput {
public:
    LocalFieldOutput()=default;
    LocalFieldOutput(const std::string& directory,const Mesh& mesh,double gamma,
                     std::vector<std::uint64_t> node_dof,std::size_t state_nodes);
    void write(const OutputZone& zone,const std::vector<unsigned>& columns,std::uint64_t step,double time,
               const std::vector<double>& state) const;
private:
    std::string directory_;
    FieldSnapshot metadata_;
    std::vector<std::uint64_t> node_dof_;
    std::size_t nodes_=0;
    double gamma_=0;
};

class ScheduledOutput : public OutputSchedule {
public:
    ScheduledOutput(const std::vector<OutputZone>& zones,const std::string& directory,const Mesh& mesh,double gamma,
                    std::vector<std::uint64_t> node_dof = {},std::size_t state_nodes=0);
    void write(std::uint64_t step,double time,const std::vector<double>& state);
private:
    LocalFieldOutput local_;
};
}
#endif
