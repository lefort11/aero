#ifndef AERO_DEVICE_WORK_H
#define AERO_DEVICE_WORK_H
#include "../aero/partition_kernels.h"
#include "../residual/flux_cfl.h"
#include "device_api.h"
#include "../io/diagnostic_fields.h"

// Граница модуля исполнения. Все указатели на поля и геометрию принадлежат текущей GPU.
// Возвращаемое сообщение живёт до следующего вызова на этом потоке CPU. nullptr означает успех.
#ifdef AERO_HAS_TURBULENCE
#include "../turbulence/transport.h"
#include "../turbulence/convection.h"
#endif
namespace aero {
namespace device_work {
using residual_detail::Index;
extern "C" {
AERO_DEVICE_ENTRY const char* aero_device_pressure_rate(const partition_detail::PressureRate&, Index count);
AERO_DEVICE_ENTRY const char* aero_device_observe_pressure(const partition_detail::ObservePressure&, Index count);
AERO_DEVICE_ENTRY const char* aero_device_pressure_caps(const partition_detail::UpdatePressureCaps&, Index count, MeshIndex*);
AERO_DEVICE_ENTRY const char* aero_device_sample(const diagnostic::Sample&,Index count);
#ifdef AERO_HAS_TURBULENCE
AERO_DEVICE_ENTRY const char* aero_device_sa_step(const turbulence::Step&, turbulence::Operation);
AERO_DEVICE_ENTRY const char* aero_device_sa_convection(const turbulence::SaEdgeConvection&, MeshIndex);
AERO_DEVICE_ENTRY const char* aero_device_sa_norm(const turbulence::Step&, turbulence::Norm, bool, double*);
#endif
#ifdef AERO_HAS_VISCOSITY
AERO_DEVICE_ENTRY const char* aero_device_prepare_tetra_volumes(const viscosity::Operator&);
AERO_DEVICE_ENTRY const char* aero_device_viscosity(const viscosity::Operator&, const double* state,
                                                     const EulerState* cached, viscosity::Evaluation phase=viscosity::Evaluation::Full);
#endif
AERO_DEVICE_ENTRY const char* aero_device_normals(const residual_detail::Geometry&);
AERO_DEVICE_ENTRY const char* aero_device_faces(const residual_detail::FaceBatch&, Index* bad, void* stream,
                                               const residual_detail::FaceBatch* device_faces);
AERO_DEVICE_ENTRY const char* aero_device_gather(const residual_detail::Geometry&, const EulerFluxResult*,
                                                double* residual, Index* bad, void* stream);
AERO_DEVICE_ENTRY const char* aero_device_local_steps(const residual_detail::LocalSteadySteps&, residual_detail::StepRange*);
AERO_DEVICE_ENTRY const char* aero_device_limit_local_steps(const residual_detail::LimitLocalSteps&, Index count);
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
