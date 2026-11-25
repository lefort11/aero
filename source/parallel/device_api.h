#ifndef AERO_PARALLEL_DEVICE_API_H
#define AERO_PARALLEL_DEVICE_API_H
#include "../mesh/mesh_index.h"
#include "modules.h"
#include <cstdlib>
#include <stdexcept>

// Интерфейс модуля Kokkos не требует заголовков драйвера. Одинаково применим
// при статической линковке и при подключении отдельной библиотеки устройства.
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
#ifdef AERO_HAS_MULTIGRID
namespace multigrid { struct Transfer; struct VectorUpdate; }
#endif
#ifdef AERO_HAS_VISCOSITY
namespace viscosity { struct Jacobian; }
#endif
namespace implicit { struct CsgsSweep; struct CsgsPacking; struct JacobiBlocks; struct SpatialBlocks; struct KrylovUpdate; struct NewtonUpdate; struct TimePrediction; }
struct SteadyReport;
struct EulerFluxResult;
namespace residual_detail { struct FaceBatch; }
namespace device_work {
extern "C" {
// Завершает среду выполнения вычислительных ядер GPU.
AERO_DEVICE_ENTRY void aero_device_shutdown();
#ifdef AERO_HAS_MULTIGRID
// Выполняет ограничение или продление многосеточного вектора на GPU.
AERO_DEVICE_ENTRY const char* aero_device_multigrid_transfer(int device, MeshIndex count, const multigrid::Transfer&);
// Обновляет рабочие векторы грубосеточной коррекции на GPU.
AERO_DEVICE_ENTRY const char* aero_device_multigrid_update(int device, MeshIndex count, const multigrid::VectorUpdate&);
#endif
// Рабочая память принадлежит исполнителю на указанной карте. При каждом
// вызове он сам выбирает её и восстанавливает прежнее устройство вызывающего потока.
AERO_DEVICE_ENTRY const char* aero_device_jacobi_allocate(int device, implicit::JacobiBlocks&);
// Освобождает память диагональных блоков GPU.
AERO_DEVICE_ENTRY const char* aero_device_jacobi_release(int device, const implicit::JacobiBlocks&);
// Разлагает диагональные блоки параллельно на GPU.
AERO_DEVICE_ENTRY const char* aero_device_jacobi_factor(int device, const implicit::JacobiBlocks&,
                                                       const double* diagonal, const MeshIndex* row_starts, MeshIndex* first_bad);
// Решает независимые диагональные системы на GPU.
AERO_DEVICE_ENTRY const char* aero_device_jacobi_apply(int device, const implicit::JacobiBlocks&,
                                                      const double* residual, double* correction);
// Упаковывает коэффициенты матрицы в порядок проходов CSGS.
AERO_DEVICE_ENTRY const char* aero_device_pack_csgs(int device, const implicit::CsgsPacking&, const double* matrix);
// Вычисляет матричный образ для прохода CSGS на GPU.
AERO_DEVICE_ENTRY const char* aero_device_csgs_image(int device,const implicit::CsgsSweep&,MeshIndex count,double* image);
// Обновляет независимые строки одного цвета CSGS на GPU.
AERO_DEVICE_ENTRY const char* aero_device_csgs_color(int device, const implicit::CsgsSweep&,
                                                    MeshIndex first, MeshIndex count);
#ifdef AERO_HAS_VISCOSITY
// Собирает вязкие производные в блоки пространственной матрицы на GPU.
AERO_DEVICE_ENTRY const char* aero_device_viscous_matrix(int device, const viscosity::Jacobian&, const double* state);
#endif
// Вычисляет производные потоков площадок на GPU.
AERO_DEVICE_ENTRY const char* aero_device_spatial_faces(int device, const implicit::SpatialBlocks&,
                                               const double* state, const double* initial, MeshIndex* bad);
// Собирает строки пространственного якобиана из производных площадок.
AERO_DEVICE_ENTRY const char* aero_device_spatial_rows(int device, const implicit::SpatialBlocks&,
                                               const double* volumes, double dt, MeshIndex* bad);
// Обновляет временные диагональные коэффициенты системы на GPU.
AERO_DEVICE_ENTRY const char* aero_device_time_diagonal(int device, const implicit::SpatialBlocks&,
                                               const double* volumes, double dt, MeshIndex* bad);
// Умножает распределённую пространственную матрицу на вектор на GPU.
AERO_DEVICE_ENTRY const char* aero_device_spatial_apply(int device, const implicit::SpatialBlocks&,
                                               const double* vector, double* product);
// Обновляет рабочие векторы BiCGSTAB на GPU.
AERO_DEVICE_ENTRY const char* aero_device_krylov_update(int device, MeshIndex count, const implicit::KrylovUpdate&);
// Вычисляет невязку неявного шага и её норму на GPU.
AERO_DEVICE_ENTRY const char* aero_device_newton_residual(int device, MeshIndex rows, const implicit::NewtonUpdate&, double* norm);
// Применяет выбранное обновление полей Ньютона на GPU.
AERO_DEVICE_ENTRY const char* aero_device_newton_update(int device, MeshIndex nodes, const implicit::NewtonUpdate&);
// Экстраполирует принятое поле по компактной истории на той же карте.
AERO_DEVICE_ENTRY const char* aero_device_time_prediction(int device, MeshIndex rows, const implicit::TimePrediction&);
// Уточняет пробу на отдельных узлах цельной двумерной сетки.
AERO_DEVICE_ENTRY const char* aero_device_local_gas_prediction(int device, MeshIndex rows, const implicit::NewtonUpdate&,
                                                               EulerFluxResult* flux,const residual_detail::FaceBatch* stored,
                                                               const double* initial,double threshold);
// Вычисляет частичные сводки steady без копирования полей на CPU.
AERO_DEVICE_ENTRY const char* aero_device_steady_report(const SteadyReport&);
// Вычисляет скалярные произведения для стабилизации BiCGSTAB на GPU.
AERO_DEVICE_ENTRY const char* aero_device_krylov_project(int device,MeshIndex count,const double* t,const double* r,double* pair);
// Совмещает обновление вектора Крылова с вычислением нормы на GPU.
AERO_DEVICE_ENTRY const char* aero_device_krylov_update_norm(int device,MeshIndex count,const implicit::KrylovUpdate&,double* norm);
// Вычисляет скалярное произведение или максимальную норму вектора на GPU.
AERO_DEVICE_ENTRY const char* aero_device_krylov_reduce(int device, MeshIndex count, const double* x,
                                                        const double* y, double* scalar);
}
// Текст ошибки принадлежит модулю GPU до следующего обращения на данном потоке.
inline void check(const char* error) {
    static const int registered = std::atexit(aero_device_shutdown);
    if (registered)
        throw std::runtime_error("Unable to register Kokkos finalizer");
    if (error)
        throw std::runtime_error(error);
}
}
}
#endif
