#ifndef AERO_EULERSOLVER_CUDA_TESTS_CUH
#define AERO_EULERSOLVER_CUDA_TESTS_CUH
#include "lax_friedrichs.h"
#include "roe.h"
#include "characteristic.h"
#include "../cuda/runtime.h"

namespace aero {
namespace eulersolver_cuda_tests_kernels {
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
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void one_interface(const Case* input, Output* output) {
    const Case& c = *input;
    aero::EulerStatus status;
    if (c.mode == 0)
        status = aero::lax_friedrichs_conservative(c.dimension, c.left, c.right, c.direction, c.gamma,
                                                   &output->result);
    else if (c.mode == 3)
        status = aero::roe_conservative(c.dimension, c.left, c.right, c.direction, c.gamma, &output->result);
    else if (c.mode == 1 || c.mode == 4) {
        // Перед выбором LF или Roe вызывающий код восстанавливает U из W.
        aero::EulerRoeBasis basis;
        double ul[5], ur[5];
        status = aero::euler_roe_basis(c.dimension, c.aux_left, c.aux_right, c.direction, c.gamma, &basis);
        if (status == aero::EulerStatus::Success)
            status = aero::characteristic_to_conservative(basis, c.left, ul);
        if (status == aero::EulerStatus::Success)
            status = aero::characteristic_to_conservative(basis, c.right, ur);
        if (status == aero::EulerStatus::Success)
            status = c.mode == 1
                         ? aero::lax_friedrichs_conservative(c.dimension, ul, ur, c.direction, c.gamma,
                                                             &output->result)
                         : aero::roe_conservative(c.dimension, ul, ur, c.direction, c.gamma, &output->result);
    } else if (c.mode == 2)
        status = aero::lax_friedrichs_fluxes(c.dimension, c.left, c.right, c.aux_left, c.aux_right,
                                             c.direction, c.gamma, &output->result);
    else
        status = aero::roe_fluxes(c.dimension, c.left, c.right, c.aux_left, c.aux_right, c.direction, c.gamma,
                                  &output->result);
    output->status = static_cast<int>(status);
}
#endif
}
}
#endif
