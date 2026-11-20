#include "explicit_stage.h"
#include "../parallel/device_work.h"

namespace aero {
namespace explicit_detail {
// Запускает обновление компонент поля по одним и тем же коэффициентам стадии Эйлера или SSPRK3.
void launch_update(FieldStageUpdate update, std::size_t entries, int* invalid, void* stream) {
    device_work::check(device_work::aero_device_stage(update, entries, invalid, stream));
}
}
}
