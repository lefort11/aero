#pragma once
#include "transport.h"
#include <cfloat>

namespace aero {
namespace turbulence {
// Производная узлового источника по рабочей вязкости на зафиксированном газе.
// Значение модели читается из того же выражения, что формирует пространственный баланс.
struct LocalSource {
    const View& sa; // Коэффициенты SA и, при наличии, параметры DDES.
    MeshIndex node; // Локальный номер узла.
    double nu;      // Молекулярная кинематическая вязкость в этом узле.
    double rotation; // Модуль ротора текущего, фиксированного поля скорости.

    // Ограничение длины DDES зависит от турбулентной вязкости и входит в производную.
    AERO_HOST_DEVICE double value(double w) const {
        double inverse=sa.inverse_distance[node];
        const double chi=w/nu;
        const double damp=(inverse!=0. || sa.ddes) ? fv1(chi) : 0.;
        if (sa.ddes) inverse=sa.ddes[node].inverse_length(inverse,nu+w*damp,kappa);
        return inverse<0. ? 0. : w*source_rate_from_chi(w,chi,damp,rotation,inverse);
    }
    // Симметричная разность внутри области w>=0 и односторонняя вблизи её границы.
    AERO_HOST_DEVICE double derivative(double w) const {
        const double h=6.055454452393343e-6*::fmax(nu,::fabs(w));
        const double minus=::fmax(0.,w-h), plus=w+h;
        return (value(plus)-value(minus))/(plus-minus);
    }
};

// Параметры неявной строки скалярного уравнения. Столбцы имеют узловую нумерацию.
struct SaJacobianRow {
    Step step; // Состояние SA, поток массы, молекулярная вязкость и коэффициенты времени.
    GasFaces faces; // Полная геометрия граней для расчёта конвекции и градиента.
    bool simplified; // Фиксирует коэффициент переменной диффузии в упрощённой матрице.

    // Даёт внешнее значение для втекания. Выходная площадка использует значение узла.
    AERO_HOST_DEVICE bool prescribed(MeshIndex face,MeshIndex i,double& inlet) const {
        const auto& t=step.transport;
        inlet=t.sa.ambient;
        if (!t.profiles) return true;
        const auto& profile=t.profiles[face-t.edges];
        if (!profile.present()) return true;
        if (!profile.columns || (profile.columns & NuTilde)) inlet=profile.nutilde;
        else if (profile.fallback==ProfileFallback::Interior) return false;
        else if (profile.fallback==ProfileFallback::Initial && t.sa.inverse_distance[i]<0.) inlet=0.;
        return true;
    }

    // Добавляет dF/dw для конвекции первого порядка, диффузии и источника.
    // Писатель строки складывает повторные вклады одного соседа и реализует
    // выбранный полный либо компактный граф без изменений пространственной невязки.
    template<class Row>
    AERO_HOST_DEVICE void assemble(MeshIndex k,const Row& row) const {
        const auto& t=step.transport;
        const auto& sa=t.sa;
        const auto i=t.node(k);
        const double w=sa.w[i], inv_volume=1./t.volume[i], h=step.local_steps.at(i,step.dt);
        if (sa.zero_working_viscosity(i)) { row.add(i,1.); return; }
        const double inv_density=1./t.gas_nodes[i].density;
        const auto range=faces.row(i);
        double gradient[3]={},diagonal=step.leading;
        for (auto p=range.first;p<range.end;++p) {
            const auto side=faces.incidence[p];
            const auto face=faces.read(i,p,t.edges,t.dim);
            const double incoming=t.incoming_speed(side,inv_density);
            const bool interior=side.face()<t.edges;
            double outside=w;
            if (interior) outside=sa.w[face.neighbour];
            else if (!incoming || !prescribed(side.face(),i,outside)) continue;
            const double adv=h*inv_volume*incoming;
            diagonal+=adv;
            if (interior) row.add(face.neighbour,-adv);
            for (unsigned a=0;a<t.dim;++a) gradient[a]+=face.area[a]*(outside-w)*inv_volume;
        }
        // Нелинейный градиентный член cb2/sigma * grad(w)^2 даёт межузловые связи.
        for (auto p=range.first;p<range.end;++p) {
            const auto side=faces.incidence[p];
            const auto face=faces.read(i,p,t.edges,t.dim);
            const bool interior=side.face()<t.edges;
            double outside;
            if (!interior && (!t.incoming_speed(side,inv_density) || !prescribed(side.face(),i,outside))) continue;
            double grad=0.;
            for (unsigned a=0;a<t.dim;++a) grad+=gradient[a]*face.area[a];
            const double coefficient=h*inv_volume*(2.*cb2/sigma)*grad;
            diagonal+=coefficient;
            if (interior) row.add(face.neighbour,-coefficient);
        }
        for (auto p=t.filter.offsets[i];p<t.filter.offsets[i+1];++p) {
            const auto entry=t.filter.incidence[p],e=entry/8;
            const auto cell=t.cells[e];
            const auto local=static_cast<unsigned>(entry%8);
            const double mean=sa.diffusion.element_mean[e];
            double jump=0.;
            for (unsigned j=0;j<cell.count;++j) {
                const auto n=cell.node[j];
                const double g=diffusion_weight(t.shared_geometry,cell,local,j,t.heat_to_diffusion);
                if (g==0.) continue;
                const double coefficient=h*inv_volume*mean*g;
                diagonal+=coefficient;
                row.add(n,-coefficient);
                if (!simplified) jump+=g*(sa.w[n]-w);
            }
            if (!simplified) {
                const double change=-h*inv_volume*jump*cell.inverse_count;
                for (unsigned j=0;j<cell.count;++j) row.add(cell.node[j],change);
            }
        }
        const LocalSource source{sa,i,sa.nodal_material[i].nu,sa.rotation[i]};
        row.add(i,diagonal-h*source.derivative(w));
    }
};
}
}
