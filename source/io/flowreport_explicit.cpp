#include "flowreport.h"
#include "../explicit/explicit.h"
#include <stdexcept>

namespace aero {
namespace {
template <class Read>
void read_accepted(const ExplicitIntegrator& solver, std::size_t nodes, unsigned width, const Read& read) {
    // Для CPU читаем принятое поле непосредственно, пока действует представление интегратора.
    solver.inspect([&](const ExplicitField& field) {
        if (field.nodes != nodes || field.components != width)
            throw std::invalid_argument("flowreport: integrator and grid have different layouts");
        if (field.backend == ExplicitBackend::Cpu)
            read(field.values);
    });
    if (solver.backend() == ExplicitBackend::Cuda) {
        // Отчёт работает на CPU, поэтому для CUDA нужен снимок принятого состояния.
        const auto host = solver.state();
        read(host.data());
    }
}
}
void FlowReport::capture(std::uint64_t step, const ExplicitIntegrator& solver) {
    if (due(step))
        read_accepted(solver, volumes_.size(), dimension_ + 2,
                      [&](const double* u) { capture_values(step, u); });
}
void FlowReport::write(std::uint64_t step, double time, double cfl, double dt,
                       const ExplicitIntegrator& solver, std::ostream& out) {
    if (due(step))
        read_accepted(solver, volumes_.size(), dimension_ + 2,
                      [&](const double* u) { write_values(step, time, cfl, dt, u, out); });
}
}
