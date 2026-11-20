#ifndef AERO_IMPLICIT_BDF1_STEP_H
#define AERO_IMPLICIT_BDF1_STEP_H
#include <functional>
#include <memory>
#include <vector>
#include "partition.h"
#include "implicit_settings.h"

namespace aero {
struct EulerFluxResult;
namespace residual_detail { struct Geometry; }
namespace implicit {
// История принятых шагов в локальной нумерации узлов для контрольной точки.
struct BdfHistory {
    double dt = 0.;                 // Ноль означает первый шаг или старую контрольную точку BDF1.
    std::vector<double> previous;   // U_(n-1) при текущем состоянии U_n.
};
struct NewtonResult {
    unsigned iterations;       // Сколько поправок применили к полю.
    double relative_residual;  // Отношение норм после последней проверки полного уравнения выбранной схемы BDF.
    unsigned jacobian_assemblies; // Сколько раз на шаге заново дифференцировали потоки.
};
// Геометрия и потоки принадлежат пространственному оператору и живут дольше решателя.
// CPU строит структуру матрицы по host, значения блоков вычисляются по active на выбранном устройстве.
class BdfStep {
public:
    BdfStep(const residual_detail::Geometry& host, const residual_detail::Geometry& active, int device,
             const Partition& partition = Partition());
    ~BdfStep();
    // Максимум |R_i|/V_i по собственным узлам всех процессов и всем уравнениям.
    double spatial_rate_norm(const double* volumes, const EulerFluxResult* flux);
    void export_history(BdfHistory& history) const;
    void import_history(const BdfHistory& history);
    // На входе потоки уже вычислены при выборе dt. После каждой поправки refresh_fluxes
    // пересчитывает их тем же пространственным оператором. При ошибке восстанавливаем U_n.
    NewtonResult advance(double* state, const double* initial, const double* volumes,
                         const EulerFluxResult* flux, double dt, const NewtonSettings& settings,
                         const std::function<void()>& refresh_fluxes,
                         const std::function<bool()>& try_prediction_fluxes = {},
                         const std::function<bool()>& confirm_prediction = {});
    // Необязательный обработчик пробного поля пересчитывает те же потоки.
    // false означает физически недопустимую пробу, которую надо откатить.
private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
};
using Bdf1Step = BdfStep;
}
}
#endif
