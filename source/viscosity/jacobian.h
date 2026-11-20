#ifndef AERO_VISCOUS_JACOBIAN_H
#define AERO_VISCOUS_JACOBIAN_H
#include "transport.h"
#include "../implicit/edge_jacobian.h"

namespace aero {
namespace viscosity {
// Узел столбца может входить в несколько элементов и встречаться в одном элементе
// несколькими периодическими копиями. Адреса блоков определяются один раз при старте.
struct ColumnElement {
    MeshIndex cell;
    unsigned source;    // Вершина элемента, консервативное состояние которой варьируем.
    MeshIndex block[8]; // Позиция (строка вершины, столбец источника) в CSR или missing.
};
constexpr MeshIndex missing_block = static_cast<MeshIndex>(-1);

// Производная по одной компоненте U=(rho,rho*u,E). thermal обозначает p/rho.
// Сначала дифференцируем переход к примитивным переменным, затем оба коэффициента
// элемента: среднюю вязкость и среднее произведение вязкости на скорость.
struct GasDerivative {
    Difference primitive;
    Material transport;
    AERO_HOST_DEVICE GasDerivative(unsigned dim, const Law& law, const Gas& gas, unsigned variable)
        : primitive{}, transport{} {
        const double thermal = gas.pressure_over_density;
        const double kinetic = variable == 0 ? .5*euler_detail::dot(dim,gas.velocity,gas.velocity) : 0.;
        const double dp = implicit::edge_detail::pressure_column(variable,dim+1,law.gamma,kinetic,gas);
        primitive.thermal = (dp - (variable == 0 ? thermal : 0.)) / gas.density;
        for (unsigned m = 0; m < dim; ++m)
            primitive.velocity[m] = implicit::edge_detail::velocity_column(variable,m,gas);
        // d ln(mu)/d ln(T) для normal равно 1/2; поправка Сазерленда равна S/(T+S).
        const double theta = law.temperature_scale * thermal;
        const double exponent = .5 + law.sutherland_ratio/(theta+law.sutherland_ratio);
        transport.mu = exponent * gas.mu * (primitive.thermal/thermal);
        for (unsigned m = 0; m < dim; ++m)
            transport.mu_velocity[m] = transport.mu*gas.velocity[m] + gas.mu*primitive.velocity[m];
    }
};

struct Jacobian {
    Operator aes;
    const MeshIndex* offsets;       // Все вхождения каждой неизвестной в элементы.
    const ColumnElement* elements;
    Gradient* gradients;            // Геометрическая часть напряжений и теплопереноса по вершинам элемента.
    double* values;                 // Та же CSR-матрица, в которую добавляется конвективный вклад.
    MeshIndex value_count;

    // Градиенты и коэффициенты нужны всем столбцам элемента. Готовим их один раз
    // при фактической сборке матрицы на том состоянии, которое передал Ньютон.
    AERO_HOST_DEVICE void prepare_cell(MeshIndex index) const {
        aes.average_material(index);
        const auto& cell = aes.cells[index];
        if (aes.dim == 3) aes.stencil.cell_gradients<3>(cell, aes.gas, gradients);
        else if (aes.dim == 2) aes.stencil.cell_gradients<2>(cell, aes.gas, gradients);
        else aes.stencil.cell_gradients<1>(cell, aes.gas, gradients);
    }
    // Обходим столбец: изменение одной компоненты U в одном узле. Потоки CPU
    // и GPU пишут в разные столбцы даже при совпадении строк, атомарные операции не нужны.
    template<unsigned dim>
    AERO_HOST_DEVICE void add_column(MeshIndex node, unsigned variable) const {
        if (offsets[node] == offsets[node+1]) return;
        const unsigned width = dim+2;
        const GasDerivative derivative(dim,aes.law,aes.gas[node],variable);
        for (MeshIndex k = offsets[node]; k < offsets[node+1]; ++k) {
            const auto& entry = elements[k];
            const auto& cell = aes.cells[entry.cell];
            const unsigned j = entry.source;
            const auto diagonal = entry.block[j];
            double balance[5]{};
            for (unsigned i = 0; i < cell.count; ++i) {
                // После периодической склейки поток между копиями одного узла внутренний.
                if (cell.node[i] == node) continue;
                const auto slot = entry.block[i];
                if (slot == missing_block && diagonal == missing_block) continue;

                // Правило произведения. Меняются разности u и p/rho на паре i,j,
                // а также усреднённые по ВСЕМ вершинам коэффициенты перед градиентами.
                Gradient direct{};
                const auto& pair = aes.stencil.metric(cell,i,j);
                add_gradient<dim>(pair,i > j,derivative.primitive,direct);
                double field[5]{}, coefficient[5]{};
                aes.material[entry.cell].add_flux(dim,direct,field);
                derivative.transport.add_flux(dim,gradients[cell.first_vertex+i],coefficient);
                for (unsigned r = 1; r < width; ++r) {
                    const double value = field[r] + coefficient[r]*cell.inverse_count;
                    if (slot != missing_block) values[(slot*width+r)*width+variable] += value;
                    balance[r] -= value;
                }
            }
            // Сумма производных внутренней невязки всех строк по U_j равна нулю.
            // В диагонали учитываем также соседние строки, принадлежащие другому рангу.
            if (diagonal != missing_block)
                for (unsigned r = 1; r < width; ++r)
                    values[(diagonal*width+r)*width+variable] += balance[r];
        }
    }
    AERO_HOST_DEVICE void assemble_column(MeshIndex node, unsigned variable) const {
        if (aes.dim == 3) add_column<3>(node,variable);
        else if (aes.dim == 2) add_column<2>(node,variable);
        else add_column<1>(node,variable);
    }
};
}
}
#endif
