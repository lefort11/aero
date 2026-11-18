#ifndef AERO_IO_FLOWREPORT_H
#define AERO_IO_FLOWREPORT_H

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace aero {
class ExplicitIntegrator;

// One reporter per run. Volumes correspond to independent computational nodes.
class FlowReport {
public:
    FlowReport(std::uint64_t every, double gamma, unsigned dimension,
               std::size_t nodes, const double* volumes);
    bool due(std::uint64_t step) const;

    // Use the same upcoming (one-based) step number before and after a
    // successful step. Repeating capture for a retried step is supported.
    void capture(std::uint64_t step, const std::vector<double>& previous);
    void write(std::uint64_t step, double time, double cfl, double dt, const std::vector<double>& current,
               std::ostream& out=std::cout);
    // CPU and CUDA integrator adapters avoid even inspecting off-schedule steps.
    void capture(std::uint64_t step, const ExplicitIntegrator& solver);
    void write(std::uint64_t step, double time, double cfl, double dt, const ExplicitIntegrator& solver,
               std::ostream& out=std::cout);
private:
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
