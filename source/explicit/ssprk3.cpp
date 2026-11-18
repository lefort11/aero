#include "explicit_stage.h"

namespace aero { namespace explicit_detail {
Stages ssprk3_stages() {
    // Non-autonomous Shu-Osher form: time, weight of U^n, weight of FE(Y).
    static const Stage stages[]={
        {0.0,0.0,1.0},
        {1.0,3.0/4.0,1.0/4.0},
        {0.5,1.0/3.0,2.0/3.0}
    };
    return {stages,3};
}
}}
