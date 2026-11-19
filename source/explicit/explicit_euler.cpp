#include "explicit_stage.h"

namespace aero {
namespace explicit_detail {
Stages euler_stages() {
    // Один вызов даёт принятое состояние U на момент t_n + dt.
    static const Stage stages[] = {first_euler_stage()};
    return {stages, 1};
}
}
}
