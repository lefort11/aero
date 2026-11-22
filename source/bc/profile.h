#pragma once
#include "bc.h"
#include "../cv/cv.h"
#include <algorithm>
#include <stdexcept>

namespace aero {
// По одну сторону периодического шва исходные координаты могут отличаться.
// Поэтому профиль привязывается к физической площадке до переноса на GPU.
inline void append_boundary_profile(std::vector<ProfileState>& samples, const BoundaryPatches& patches,
                                    const CvBoundary& piece, unsigned dim, const CvVector* axes, double gamma) {
    if (patches.profiles.empty()) return;
    samples.emplace_back(ProfileState{});
    const auto* rules = patches.by_patch[piece.patch].rules(patches.free_rules.data());
    if (!rules) return;
    const auto index = static_cast<std::size_t>(rules-patches.free_rules.data());
    if (index >= patches.profiles.size()) return;
    const auto& profile = patches.profiles[index];
    if (profile.rows.empty() && profile.columns != Radius) return;
    auto& ref = samples.back();
    ref.columns = profile.columns;
    ref.fallback = profile.fallback;
    if (profile.rows.empty()) return;
    Coordinate radial = piece.point-profile.center;
    double along = 0.;
    for (unsigned k=0;k<3;++k) along += radial[k]*profile.axis[k];
    for (unsigned k=0;k<3;++k) radial[k] -= along*profile.axis[k];
    const double radius = coordinate_length(radial);
    // Вне таблицы берём ближайшую строку. На оси радиальная скорость равна нулю.
    auto hi = std::lower_bound(profile.rows.begin(),profile.rows.end(),radius,
        [](const ProfileRow& row,double r){return row.radius<r;});
    if (hi==profile.rows.end()) --hi;
    auto lo = hi;
    if (hi!=profile.rows.begin() && radius<hi->radius) --lo;
    const double t = lo==hi ? 0. : (radius-lo->radius)/(hi->radius-lo->radius);
    const auto interpolate = [&](double ProfileRow::*field){return (*lo).*field + t*((*hi).*field-(*lo).*field);};
    const double ux=interpolate(&ProfileRow::axial), ur=interpolate(&ProfileRow::radial);
    CvVector velocity{};
    Coordinate v{};
    for (unsigned k=0;k<3;++k) v[k]=ux*profile.axis[k]+(radius>0. ? ur*(radial[k]/radius) : 0.);
    velocity={v[0],v[1],v[2]};
    ref.nutilde=interpolate(&ProfileRow::nutilde);
    if (ref.partial_gas()) {
        // Сохраняем заданные компоненты. Недостающие появятся при выборе опорного состояния.
        ref.gas.density = interpolate(&ProfileRow::density);
        ref.gas.pressure = interpolate(&ProfileRow::pressure);
        Coordinate direction{};
        if (profile.columns & Axial) direction = profile.axis;
        else if ((profile.columns & Radial) && radius>0.)
            for (unsigned j=0;j<3;++j) direction[j] = radial[j]/radius;
        const CvVector projected{direction[0], direction[1], direction[2]};
        for (unsigned k=0;k<dim;++k) {
            ref.gas.velocity[k] = velocity.dot(axes[k]);
            ref.direction[k] = projected.dot(axes[k]);
        }
        return;
    }
    // Проецируем мировую скорость в базис, в котором солвер хранит импульс.
    double u[5]={};
    u[0]=interpolate(&ProfileRow::density);
    u[dim+1]=interpolate(&ProfileRow::pressure)/(gamma-1.);
    for (unsigned k=0;k<dim;++k) {
        const double speed=velocity.dot(axes[k]);
        u[k+1]=u[0]*speed;
        u[dim+1]+=.5*u[k+1]*speed;
    }
    if (euler_state(dim,u,gamma,&ref.gas)!=EulerStatus::Success)
        throw std::invalid_argument("Invalid interpolated boundary profile");
}
}
