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
// Вычисляет диагностическую производную давления по времени на GPU.
AERO_DEVICE_ENTRY const char* aero_device_pressure_rate(const partition_detail::PressureRate&, Index count);
// Добавляет вклад текущего поля во временные моменты на GPU.
AERO_DEVICE_ENTRY const char* aero_device_sample(const diagnostic::Sample&,Index count);
#ifdef AERO_HAS_TURBULENCE
// Выполняет выбранную операцию шага SA на GPU.
AERO_DEVICE_ENTRY const char* aero_device_sa_step(const turbulence::Step&, turbulence::Operation);
// Собирает конвективный перенос рабочей вязкости SA на GPU.
AERO_DEVICE_ENTRY const char* aero_device_sa_convection(const turbulence::SaEdgeConvection&, MeshIndex);
// Вычисляет максимальную норму выбранной величины SA на GPU.
AERO_DEVICE_ENTRY const char* aero_device_sa_norm(const turbulence::Step&, turbulence::Norm, bool, double*);
#endif
#ifdef AERO_HAS_VISCOSITY
// Подготавливает объёмные множители тетраэдрального разбиения на GPU.
AERO_DEVICE_ENTRY const char* aero_device_prepare_tetra_volumes(const viscosity::Operator&);
// Запускает требуемые фазы вязкого оператора AES на GPU.
AERO_DEVICE_ENTRY const char* aero_device_viscosity(const viscosity::Operator&, const double* state,
                                                     const EulerState* cached, viscosity::Evaluation phase=viscosity::Evaluation::Full);
#endif
// Подготавливает единичные нормали площадок в базисе решателя.
AERO_DEVICE_ENTRY const char* aero_device_normals(const residual_detail::Geometry&);
// Вычисляет численные потоки выбранного набора площадок на GPU.
AERO_DEVICE_ENTRY const char* aero_device_faces(const residual_detail::FaceBatch&, Index* bad, void* stream,
                                               const residual_detail::FaceBatch* device_faces);
// Собирает узловую невязку из ориентированных потоков площадок на GPU.
AERO_DEVICE_ENTRY const char* aero_device_gather(const residual_detail::Geometry&, const EulerFluxResult*,
                                                double* residual, Index* bad, void* stream);
// Собирает ограничения скорости изменения по опорам реконструкции на GPU.
AERO_DEVICE_ENTRY const char* aero_device_stencil_rates(const residual_detail::StencilRates&, Index count);
// Вычисляет локальные стационарные шаги и их диапазон на GPU.
AERO_DEVICE_ENTRY const char* aero_device_local_steps(const residual_detail::LocalSteadySteps&, residual_detail::StepRange*);
// Применяет верхние ограничения локальных шагов на GPU.
AERO_DEVICE_ENTRY const char* aero_device_limit_local_steps(const residual_detail::LimitLocalSteps&, Index count);
// Выполняет проход ограничения отношений шагов соседних узлов на GPU.
AERO_DEVICE_ENTRY const char* aero_device_neighbour_steps(const residual_detail::NeighbourStepSweep&, Index count,
                                                          residual_detail::NeighbourStepRange*);
// Выбирает шаг CFL по спектральным скоростям потоков на GPU.
AERO_DEVICE_ENTRY const char* aero_device_flux_step(const residual_detail::FluxCfl&, double courant,
                                                   CflStep* answer, void* stream);
// Выбирает шаг CFL по узловым скоростям волн на GPU.
AERO_DEVICE_ENTRY const char* aero_device_wave_step(cfl_detail::Grid, WaveSpeeds, double courant,
                                                   CflStep* answer, void* stream);
// Обновляет узловое поле одной явной стадией на GPU.
AERO_DEVICE_ENTRY const char* aero_device_stage(explicit_detail::FieldStageUpdate, size_t entries,
                                               int* invalid, void* stream);
// На итерации Ньютона новое состояние формирует другое ядро. Проверяем его после обмена.
AERO_DEVICE_ENTRY const char* aero_device_failed_nodes(const partition_detail::Grid&);
// Обновляет собственные узлы подобласти одной явной стадией на GPU.
AERO_DEVICE_ENTRY const char* aero_device_owned_stage(const partition_detail::Grid&, explicit_detail::Stage,
                                                     double dt, bool save_original);
// Упаковывает или распаковывает выбранные узловые записи на GPU.
AERO_DEVICE_ENTRY const char* aero_device_records(const partition_detail::RecordField&, const MeshIndex* ids,
                                                 Index count, const double* input, double* output);
}
}
}
#endif
