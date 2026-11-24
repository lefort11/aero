#ifndef AERO_TESTS_DEVICE_CASES_H
#define AERO_TESTS_DEVICE_CASES_H
#include "../source/parallel/device_memory.h"
#include "../source/bc/bc_test_case.h"
#include "../source/bc/periodic.h"
#include "../source/explicit/explicit_test_problem.h"
#include "../source/cv/cv.h"
namespace aero {
namespace implicit { struct EdgeFluxJacobian; }
namespace device_tests {
// Запускает проверку сборки строк периодического оператора на GPU.
AERO_DEVICE_ENTRY const char* periodic_rows(void* stream, ReconstructionStencilView stencil, const PeriodicEdge* edges, const double* field, double* output);
// Вычисляет поток тестового газодинамического интерфейса.
AERO_DEVICE_ENTRY const char* test_flux(void* stream, const BcTestCase* cases, size_t count, const FreeBoundaryRules* rules, int* failed);
// Проверяет результат GPU-ядра и целостность контрольных значений.
AERO_DEVICE_ENTRY const char* inspect(void* stream, aero::CvView v, double* result);
// Входные состояния, размерность, нормаль и режим проверки потокового ядра.
struct Case {
    // Тестируемый режим потока и пространственная размерность.
    unsigned mode, dimension;
    // Основные и вспомогательные состояния с двух сторон, направление и показатель адиабаты.
    double left[5], right[5], aux_left[5], aux_right[5], direction[3], gamma;
};
// Результат и статус GPU-потока с контрольными значениями вокруг буфера.
struct Output {
    // Контрольное значение перед выходным массивом.
    double before;
    // Вычисленный поток для сверки с CPU.
    aero::EulerFluxResult result;
    // Контрольное значение после выходного массива.
    double after;
    // Код завершения тестируемого ядра.
    int status;
};
// Входим через ту же функцию, которую будет вызывать сборщик матрицы на GPU.
AERO_DEVICE_ENTRY const char* edge_blocks(Case input, double area, double upwind, implicit::EdgeFluxJacobian* output, bool simplified, bool roe);
// Вычисляет один газодинамический интерфейс на GPU.
AERO_DEVICE_ENTRY const char* one_interface(void* stream, const Case* input, Output* output);
// Вычисляет выбранный набор газодинамических тестов на GPU.
AERO_DEVICE_ENTRY const char* evaluate(void* stream, TestProblem problem, double time, aero::ExplicitField field, double* residual);
// Запускает пакетное тестовое ядро реконструкции.
AERO_DEVICE_ENTRY const char* reconstruct_kernel(void* stream, aero::ReconstructionStencilView s, const double* values, ReconstructionIndex components, double* left, double* right);
// Запускает тестовое ядро одной стороны реконструкции.
AERO_DEVICE_ENTRY const char* one_side_kernel(void* stream, aero::ReconstructionStencilView s, ReconstructionIndex edge, unsigned side, const double* own, ReconstructionIndex components, const double* values, double* output);
// Запускает тестовое ядро оценки скоростей волн.
AERO_DEVICE_ENTRY const char* speed_kernel(void* stream, const double* u, double* speed);
// Запускает тестовое ядро сборки пространственной невязки.
AERO_DEVICE_ENTRY const char* residual_kernel(void* stream, double* r);
}
}
#endif
