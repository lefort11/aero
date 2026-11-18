#include "lax_friedrichs.h"

namespace aero {
namespace {
unsigned dimension_of(const std::vector<double>& direction) {
    if (direction.empty() || direction.size()>3) throw EulerError(EulerStatus::InvalidDimension);
    return static_cast<unsigned>(direction.size());
}
void check_size(const std::vector<double>& values,unsigned dimension) {
    if (values.size()!=dimension+2) throw std::invalid_argument("Expected dimension+2 Euler components");
}
void check(EulerStatus status) { if (status!=EulerStatus::Success) throw EulerError(status); }
}
EulerFluxResult lax_friedrichs_conservative(const std::vector<double>& left,const std::vector<double>& right,
    const std::vector<double>& direction,double gamma) {
    const unsigned d=dimension_of(direction); check_size(left,d); check_size(right,d);
    EulerFluxResult result; check(lax_friedrichs_conservative(d,left.data(),right.data(),direction.data(),gamma,&result)); return result;
}
EulerFluxResult lax_friedrichs_fluxes(const std::vector<double>& fl,const std::vector<double>& fr,
    const std::vector<double>& ul,const std::vector<double>& ur,const std::vector<double>& direction,double gamma) {
    const unsigned d=dimension_of(direction); check_size(fl,d); check_size(fr,d); check_size(ul,d); check_size(ur,d);
    EulerFluxResult result; check(lax_friedrichs_fluxes(d,fl.data(),fr.data(),ul.data(),ur.data(),direction.data(),gamma,&result)); return result;
}
} // namespace aero
