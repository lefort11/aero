#ifndef AERO_IO_SCHEDULED_OUTPUT_H
#define AERO_IO_SCHEDULED_OUTPUT_H
#include "output_settings.h"

namespace aero {
// One instance per run starting at time zero. All zones currently use the full mesh.
class ScheduledOutput {
public:
    ScheduledOutput(const std::vector<OutputZone>& zones,const std::string& directory,
                    const Mesh& mesh,double gamma,
                    const std::vector<std::uint64_t>& node_dof = {});
    double limit_dt(double time,double candidate) const;
    bool due(std::uint64_t step,double time) const;
    // Accepted host state in [computational node][conservative component] order.
    void write(std::uint64_t step,double time,const std::vector<double>& state);
private:
    struct Zone {
        OutputZone config;
        std::vector<unsigned> columns;
        std::uint64_t next_tick=1, last_step=0;
        double deadline() const;
        bool due(std::uint64_t step,double time) const;
    };
    std::vector<Zone> zones_;
    std::string directory_;
    FieldSnapshot metadata_;
    std::vector<std::uint64_t> node_dof_;
    std::size_t nodes_=0;
    double gamma_;
};
}
#endif
