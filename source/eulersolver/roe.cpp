#include "roe.h"
namespace aero {
namespace {
unsigned dimension_of(const std::vector<double>& direction) {
    if (direction.empty() || direction.size()>3) throw EulerError(EulerStatus::InvalidDimension);
    return static_cast<unsigned>(direction.size());
}
void check_size(const std::vector<double>& values,unsigned d) {
    if (values.size()!=d+2) throw std::invalid_argument("Expected dimension+2 Euler components");
}
void check(EulerStatus status) { if (status!=EulerStatus::Success) throw EulerError(status); }
}
EulerFluxResult roe_conservative(const std::vector<double>& left,const std::vector<double>& right,
    const std::vector<double>& direction,double gamma) {
    const unsigned d=dimension_of(direction); check_size(left,d); check_size(right,d);
    EulerFluxResult result; check(roe_conservative(d,left.data(),right.data(),direction.data(),gamma,&result)); return result;
}
EulerFluxResult roe_fluxes(const std::vector<double>& fl,const std::vector<double>& fr,
    const std::vector<double>& left,const std::vector<double>& right,const std::vector<double>& direction,double gamma) {
    const unsigned d=dimension_of(direction); check_size(fl,d); check_size(fr,d); check_size(left,d); check_size(right,d);
    EulerFluxResult result; check(roe_fluxes(d,fl.data(),fr.data(),left.data(),right.data(),direction.data(),gamma,&result)); return result;
}
} // namespace aero
