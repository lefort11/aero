#ifndef AERO_DEVICE_WORK_H
#define AERO_DEVICE_WORK_H
#include "../aero/partition_kernels.h"
#include "../residual/flux_cfl.h"
#include "device_api.h"

// Граница модуля исполнения. Все указатели на поля и геометрию принадлежат текущей GPU.
// Возвращаемое сообщение живёт до следующего вызова на этом потоке CPU. nullptr означает успех.
namespace aero {
namespace device_work {
using residual_detail::Index;
extern "C" {
AERO_DEVICE_ENTRY const char* aero_device_normals(const residual_detail::Geometry&);
AERO_DEVICE_ENTRY const char* aero_device_faces(const residual_detail::FaceBatch&, Index* bad, void* stream,
                                               const residual_detail::FaceBatch* device_faces);
AERO_DEVICE_ENTRY const char* aero_device_gather(const residual_detail::Geometry&, const EulerFluxResult*,
                                                double* residual, Index* bad, void* stream);
AERO_DEVICE_ENTRY const char* aero_device_flux_step(const residual_detail::FluxCfl&, double courant,
                                                   CflStep* answer, void* stream);
AERO_DEVICE_ENTRY const char* aero_device_wave_step(cfl_detail::Grid, WaveSpeeds, double courant,
                                                   CflStep* answer, void* stream);
AERO_DEVICE_ENTRY const char* aero_device_stage(explicit_detail::FieldStageUpdate, size_t entries,
                                               int* invalid, void* stream);
AERO_DEVICE_ENTRY const char* aero_device_failed_nodes(const partition_detail::Grid&);
AERO_DEVICE_ENTRY const char* aero_device_owned_stage(const partition_detail::Grid&, explicit_detail::Stage,
                                                     double dt, bool save_original);
AERO_DEVICE_ENTRY const char* aero_device_records(const partition_detail::RecordField&, const MeshIndex* ids,
                                                 Index count, const double* input, double* output);
}
}
}
#endif
