#ifndef AERO_FLOW_VALUES_H
#define AERO_FLOW_VALUES_H
#include "fields.h"
#include "diagnostic_fields.h"
#include "../eulersolver/linear_euler.h"
#include <array>

namespace aero {
// Локальное число Маха по полным скорости, плотности и статическому давлению.
inline double output_mach(const std::array<double,6>& gas, double gamma, const LinearBackground& background) {
    // В Euler фон нулевой. В Linear столбцы газа содержат пульсации относительно этого фона.
    const double rho = gas[diagnostic::Density] + background.rho;
    const double pressure = gas[diagnostic::Pressure] + background.p;
    if (!(rho > 0. && pressure > 0.))
        throw FieldError("output: mach requires positive total density and pressure");
    const double ux = gas[diagnostic::Ux] + background.velocity[0];
    const double uy = gas[diagnostic::Uy] + background.velocity[1];
    const double uz = gas[diagnostic::Uz] + background.velocity[2];
    const double sound = std::sqrt(gamma * (pressure / rho));
    return std::hypot(std::hypot(ux, uy), uz) / sound;
}

// Первые шесть столбцов — газовые. Следом могут идти рабочая и физическая
// турбулентная вязкость. Один порядок используется и в MPI, и в локальном файле.
inline double field_column(const std::array<double,6>& gas, unsigned column, const diagnostic::Fields& sampling,
                           size_t node, double gamma, const LinearBackground& background) {
    if (column<gas.size()) return gas[column];
    if (column==diagnostic::Mach) return output_mach(gas, gamma, background);
    if((column==diagnostic::WorkingNu || column==diagnostic::EddyNu) && sampling.sa)
        return sampling.sa[2*node+column-diagnostic::WorkingNu];
    if(diagnostic::time_column(column) && sampling.moments) {
        const auto& h=sampling.moments[node];
        const auto c=column-diagnostic::DensityPrime; return gas[c]-h.mean[c];
    }
    if(column==diagnostic::DpDt && sampling.pressure_rate) return sampling.pressure_rate[node];
    if(column==diagnostic::Vorticity && sampling.rotation) return sampling.rotation[node];
    if(column==diagnostic::Schlieren && sampling.schlieren) return sampling.schlieren[node];
    throw FieldError("output: missing data for "+std::string(diagnostic::name(column)));
}
// Собирает столбцы отчёта из плотности, скоростей и давления. Для Linear энергия — пульсация.
inline std::array<double, 6> primitive_output_values(unsigned d, const double* q, double gamma,
                                                      const LinearBackground& background) {
    const double energy = background.active() ? background.energy(d, q, gamma)
        : q[d+1]/(gamma-1.) + .5*q[0]*euler_detail::dot(d, q+1, q+1);
    std::array<double, 6> values;
    diagnostic::store_primary(q[0], d, q+1, q[d+1], energy, values.data());
    return values;
}

// Один порядок колонок для локальных снимков, MPI-вывода и отчёта.
// Для Linear возвращаем пульсации, включая первую вариацию полной энергии.
inline std::array<double, 6> output_values(unsigned dimension, const double* u, double gamma, std::uint64_t node, const LinearBackground& background = {}) {
    std::array<double, 6> values;
    if (!diagnostic::primary_from_state(dimension, u, gamma, background, values.data()))
        throw FieldError("output: invalid Euler state at node " + std::to_string(node));
    if (background.active()) {
        for (double value : values)
            if (!std::isfinite(value)) throw FieldError("output: nonfinite perturbation at node " + std::to_string(node));
    }
    return values;
}

// Один узел декодируется один раз на зону. slot — адрес в поле процесса,
// id — номер узла для диагностики. Получатель сам выбирает раскладку записи.
struct FieldRows {
    // Узловые консервативные переменные для выгрузки.
    const double* state;
    // Размерность задачи, задающая число компонент скорости.
    unsigned dimension;
    // Показатель адиабаты при восстановлении давления.
    double gamma;
    // Фон для вывода полей линеаризованных уравнений.
    const LinearBackground& background;
    // Доступные производные и усреднённые поля.
    const diagnostic::Fields& diagnostics;
    // Номера компонент в порядке выходных столбцов.
    const std::vector<unsigned>& columns;

    // Вычисляет выбранные выходные величины узла и передаёт их функции записи.
    template<class Store> void visit(std::size_t slot, std::uint64_t id, const Store& store) const {
        const auto* u = state + slot * (dimension + 2);
        const auto gas = output_values(dimension, u, gamma, id, background);
        for (std::size_t k = 0; k < columns.size(); ++k)
            store(k, field_column(gas, columns[k], diagnostics, slot, gamma, background));
    }
};
}
#endif
