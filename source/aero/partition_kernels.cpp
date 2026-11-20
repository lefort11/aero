#define AERO_RESIDUAL_KERNEL_ONLY
#define AERO_BC_KERNEL_ONLY
#define AERO_EULERSOLVER_KERNEL_ONLY
#define AERO_RECONSTRUCTION_KERNEL_ONLY
#include "../parallel/device_work.h"
#include <stdexcept>

namespace aero {
namespace partition_detail {
// Запускает обновление только собственных степеней свободы MPI-процесса.
void check_newton_nodes(Grid g) {
    device_work::check(device_work::aero_device_failed_nodes(g));
}
void update(Grid g, explicit_detail::Stage stage, double dt, bool save_original) {
    if (!g.owned_count)
        return;
    device_work::check(device_work::aero_device_owned_stage(g, stage, dt, save_original));
}
// Список индексов определяет порядок значений в сообщении соседнему процессу.
void transfer(RecordField field, const MeshIndex* ids, Index count, const double* input, double* output) {
    if (!count)
        return;
    device_work::check(device_work::aero_device_records(field, ids, count, input, output));
}
}
}
