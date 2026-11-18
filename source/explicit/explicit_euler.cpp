#include "explicit_stage.h"

namespace aero { namespace explicit_detail {
Stages euler_stages() {
    static const Stage stages[]={{0.0,0.0,1.0}};
    return {stages,1};
}
}}
