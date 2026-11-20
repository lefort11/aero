#ifndef AERO_DEVICE_WORK_H
#define AERO_DEVICE_WORK_H
#include "../aero/partition_kernels.h"
#include "../residual/flux_cfl.h"
#include <stdexcept>
#include <cstdlib>

// Граница модуля исполнения. Все указатели на поля и геометрию принадлежат текущей GPU.
// Возвращаемое сообщение живёт до следующего вызова на этом потоке CPU. nullptr означает успех.
#if defined(AERO_BUILD_DEVICE_WORK) && !defined(AERO_STATIC_DEVICE_WORK)
# ifdef _WIN32
#  define AERO_DEVICE_ENTRY __declspec(dllexport)
# else
#  define AERO_DEVICE_ENTRY __attribute__((visibility("default")))
# endif
#else
# define AERO_DEVICE_ENTRY
#endif
namespace aero {
namespace device_work {
using residual_detail::Index;
extern "C" AERO_DEVICE_ENTRY void aero_device_shutdown();
// Переносим сообщение об ошибке через границу модуля в исключение вызывающей стороны.
inline void check(const char* error) {
    // Вызывающая программа завершает Kokkos до выгрузки графического драйвера и DLL.
    static const int registered = std::atexit(aero_device_shutdown);
    if (registered)
        throw std::runtime_error("Unable to register Kokkos finalizer");
    if (error)
        throw std::runtime_error(error);
}
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
AERO_DEVICE_ENTRY const char* aero_device_owned_stage(const partition_detail::Grid&, explicit_detail::Stage,
                                                     double dt, bool save_original);
AERO_DEVICE_ENTRY const char* aero_device_records(const partition_detail::Grid&, const MeshIndex* ids,
                                                 Index count, double* records, bool edges, bool pack);
}
}
}
#endif
