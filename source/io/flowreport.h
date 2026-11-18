#ifndef AERO_IO_FLOWREPORT_H
#define AERO_IO_FLOWREPORT_H

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace aero {
class ExplicitIntegrator;
struct FlowRange {
    double low=std::numeric_limits<double>::max(),high=std::numeric_limits<double>::lowest();
    long double integral=0;
};
struct FlowSummary {
    FlowRange field[6];
    long double rate_scale=0,scaled_squares=0,volume=0;
    void add_squares(long double scale,long double weight);
    void merge(const FlowSummary& other,unsigned fields);
};
void write_flow_summary(const FlowSummary&,std::uint64_t step,double time,double cfl,double dt,
                        unsigned dimension,std::ostream& out);

// One reporter per run. Volumes correspond to independent computational nodes.
class FlowReport {
public:
    FlowReport(std::uint64_t every, double gamma, unsigned dimension,
               std::size_t nodes, const double* volumes);
    static bool due(std::uint64_t step,std::uint64_t every);
    bool due(std::uint64_t step) const;

    // Use the same upcoming (one-based) step number before and after a
    // successful step. Repeating capture for a retried step is supported.
    void capture(std::uint64_t step, const std::vector<double>& previous);
    FlowSummary measure(std::uint64_t step,const std::vector<double>& current);
    void write(std::uint64_t step, double time, double cfl, double dt, const std::vector<double>& current,
               std::ostream& out=std::cout);
    // CPU and CUDA integrator adapters avoid even inspecting off-schedule steps.
    void capture(std::uint64_t step, const ExplicitIntegrator& solver);
    void write(std::uint64_t step, double time, double cfl, double dt, const ExplicitIntegrator& solver,
               std::ostream& out=std::cout);
private:
    FlowSummary measure_values(std::uint64_t step,const double* u);
    void capture_values(std::uint64_t step, const double* u);
    void write_values(std::uint64_t step, double time, double cfl, double dt, const double* u, std::ostream& out);
    std::uint64_t every_, captured_step_;
    double gamma_;
    unsigned dimension_;
    std::vector<double> volumes_, previous_energy_;
    long double total_volume_;
};
}
#endif
