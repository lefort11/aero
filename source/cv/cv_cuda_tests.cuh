#ifndef AERO_CV_CUDA_TESTS_CUH
#define AERO_CV_CUDA_TESTS_CUH
#include "cv.h"
#include "../cuda/runtime.h"

namespace aero { namespace cv_cuda_tests_kernels {
#ifndef AERO_MINGW_CUDA_HOST
AERO_GPU_KERNEL void inspect(aero::CvView v,double* result) {
    double volume=0, outer=0, edge_projection=0;
    for (size_t i=4;i<12;++i) volume+=v.volumes[i];
    for (size_t i=0;i<v.boundary_count;++i) {
        const aero::CvBoundary b=v.boundary[i];
        if (b.patch==1 && (b.node==5 || b.node==6 || b.node==9 || b.node==10))
            outer+=b.area.x;
    }
    for (size_t e=0;e<v.edge_count;++e) edge_projection+=fabs(v.edge_areas[e].x);
    result[0]=volume;
    result[1]=outer;
    result[2]=edge_projection;
}
#endif
}}
#endif
