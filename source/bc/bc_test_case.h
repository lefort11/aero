#ifndef AERO_BC_TEST_CASE_H
#define AERO_BC_TEST_CASE_H
#include "bc.h"
namespace aero {
// Один сценарий связывает состояние узла и внешнюю нормаль с ожидаемым потоком или ошибкой.
struct BcTestCase {
    BoundaryConfig config;
    unsigned dimension;
    bool initial_available;
    double current[5], initial[5], area[3], gamma;
    // При ошибочном входе достаточно сравнить код и убедиться, что результат не перезаписан.
    BoundaryStatus expected_status;
    EulerFluxResult expected;
};
int check_bc_devices(const BcTestCase* cases, size_t count, const FreeBoundaryRules* rules, size_t rule_count,
                     bool required);
}
#endif
