#ifndef AERO_BC_TEST_CASE_H
#define AERO_BC_TEST_CASE_H
#include "bc.h"
namespace aero {
struct BcTestCase {
    BoundaryConfig config;
    unsigned dimension;
    bool initial_available;
    double current[5],initial[5],area[3],gamma;
    BoundaryStatus expected_status;
    EulerFluxResult expected;
};
int check_bc_devices(const BcTestCase* cases,size_t count,bool required);
}
#endif
