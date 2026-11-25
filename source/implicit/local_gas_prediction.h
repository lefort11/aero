#pragma once
#include "newton_vectors.h"

namespace aero {
namespace implicit {
// Пробная подстановка на цельной двумерной сетке. Балансы выбранных граней
// уже собраны по одному исходному полю. Вязкие строки относятся к принятому слою.
struct LocalGasPrediction {
    NewtonUpdate gas;       // Поля Ньютона, связность граней и временная диагональ.
    double threshold;       // Минимальная исходная невязка узла для дополнительной подстановки.
    const residual_detail::Index* bad; // Номер недопустимой грани в формате ядра потоков.

    // По интегральному балансу начала шага выбирает узлы для уточнения.
    AERO_HOST_DEVICE static bool active(const double* rhs,MeshIndex node,double limit) {
        for (unsigned c=0; c<4; ++c) if (::fabs(rhs[4*node+c])>limit) return true;
        return false;
    }
    // Обновляет узел по проверенным конвективным потокам пробы и прежним
    // вязким напряжениям. Полную нелинейную невязку измеряет вызывающий решатель.
    AERO_HOST_DEVICE void operator()(MeshIndex node) const {
        if (*bad!=~residual_detail::Index(0) || !active(gas.rhs,node,threshold)) return;
        double balance[4]{};
        if (!residual_detail::assemble_node_residual<4>(gas.grid,node,gas.flux,balance)) return;
        const double factor=gas.local_steps.at(node,gas.dt)/gas.volumes[node];
        for (unsigned c=0; c<4; ++c) gas.state[4*node+c]=gas.start[4*node+c]-factor*balance[c];
        residual_detail::decode_nodal_gas(2,gas.state,gas.grid.options.gamma,node,gas.grid.packed.gas);
    }
};
}
}
