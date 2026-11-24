#pragma once
#include "sigma.h"
#include "ddes.h"
#include "../mesh/mesh_index.h"
#include "../mesh/compact_table.h"

namespace aero {
namespace turbulence {
// Собирает узловую величину из вкладов элементов. У Sigma результат идёт
// в газовый поток, у DDES — прямо в источник переноса рабочей вязкости.
struct NodalFilter {
    // Границы строк узлов и номера инцидентных ячеек.
    IndexArray<MeshIndex> offsets,incidence;
    // Объёмы элементов LES и нормы градиентов по ячейкам.
    const double *les_volumes, *cell_gradient_norm;
    // Элементные данные для узловой оценки фильтра.
    const LesSample* samples;
    // Маска узлов стенки с прилипанием.
    const unsigned char* no_slip;
    // Выходная узловая турбулентная кинематическая вязкость.
    double* eddy_nu;
    // Узловые параметры длины DDES.
    DdesPoint* points;
    // Выбранная модель и масштаб фильтра LES.
    LesSettings les;
    // Коэффициент перехода от сеточного масштаба к длине DES.
    double c_des;
    // Длину можно определить без обновления по направлению потока.
    bool fixed_ddes_length;
    // Носитель узловой величины — все элементы, примыкающие к узлу. Перед
    // сборкой потоков владелец передаёт полученную ν_t соседним процессам.
    template<bool ddes>
    AERO_HOST_DEVICE void gather(MeshIndex node) const {
        // На неподвижной стенке разрешённый градиент скорости имеет ранг не выше одного.
        if (!ddes && no_slip[node]) { eddy_nu[node]=0.; return; }
        const double weight=les_volumes[node];
        if (ddes && fixed_ddes_length) {
            // Для min/max/vol узловой размер сетки задан при старте. На стадии
            // меняется только норма градиента, которая защищает пристенный слой.
            double gradient=0.;
            for (MeshIndex k=offsets[node];k<offsets[node+1];++k)
                gradient+=cell_gradient_norm[incidence[k]/8];
            points[node].gradient_norm=weight>0. ? gradient/weight : 0.;
            return;
        }
        double value=0., tilt=0., gradient=0.;
        for (MeshIndex k=offsets[node];k<offsets[node+1];++k) {
            const auto e=incidence[k]/8;
            value+=samples[e].value;
            if (ddes) gradient+=cell_gradient_norm[e];
            tilt+=samples[e].tilt;
        }
        double factor=1.;
        if (weight>0. && les.scale==turbulence::Scale::Sla && les.switch_on_vtm)
            factor=turbulence::kh_factor(tilt/weight,les.f_kh_lim);
        if (ddes) {
            // Окончательный масштаб и норма скоростного градиента нужны только владельцу строки SA.
            // Ореольные узлы с пустой строкой не участвуют в уравнении переноса.
            points[node]=weight>0. ? turbulence::DdesPoint{c_des*::sqrt(value/weight)*factor,gradient/weight}
                                    : turbulence::DdesPoint{};
        } else {
            eddy_nu[node]=weight>0. ? (value/weight)*factor*factor : 0.;
        }
    }
};
}
}
