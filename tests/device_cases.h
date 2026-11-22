#ifndef AERO_TESTS_DEVICE_CASES_H
#define AERO_TESTS_DEVICE_CASES_H
#include "../source/parallel/device_memory.h"
#include "../source/bc/bc_test_case.h"
#include "../source/bc/periodic.h"
#include "../source/explicit/explicit_test_problem.h"
#include "../source/cv/cv.h"
namespace aero {
namespace implicit { struct EdgeFluxJacobian; }
namespace device_tests {
AERO_DEVICE_ENTRY const char* periodic_rows(void* stream, ReconstructionStencilView stencil, const PeriodicEdge* edges, const double* field, double* output);
AERO_DEVICE_ENTRY const char* test_flux(void* stream, const BcTestCase* cases, size_t count, const FreeBoundaryRules* rules, int* failed);
AERO_DEVICE_ENTRY const char* inspect(void* stream, aero::CvView v, double* result);
struct Case {
    unsigned mode, dimension;
    double left[5], right[5], aux_left[5], aux_right[5], direction[3], gamma;
};
struct Output {
    double before;
    aero::EulerFluxResult result;
    double after;
    int status;
};
AERO_DEVICE_ENTRY const char* edge_blocks(Case input, double area, double upwind, implicit::EdgeFluxJacobian* output, bool simplified, bool roe);
AERO_DEVICE_ENTRY const char* one_interface(void* stream, const Case* input, Output* output);
AERO_DEVICE_ENTRY const char* evaluate(void* stream, TestProblem problem, double time, aero::ExplicitField field, double* residual);
AERO_DEVICE_ENTRY const char* reconstruct_kernel(void* stream, aero::ReconstructionStencilView s, const double* values, ReconstructionIndex components, double* left, double* right);
AERO_DEVICE_ENTRY const char* one_side_kernel(void* stream, aero::ReconstructionStencilView s, ReconstructionIndex edge, unsigned side, const double* own, ReconstructionIndex components, const double* values, double* output);
AERO_DEVICE_ENTRY const char* speed_kernel(void* stream, const double* u, double* speed);
AERO_DEVICE_ENTRY const char* residual_kernel(void* stream, double* r);
}
}
#endif
