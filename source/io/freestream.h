#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace aero {
// Начальный однородный поток и reference_state на границе используют одну систему углов и одну нормировку.
struct FreestreamGas {
    double gamma, mach, phi, theta;
    // Возвращает примитивную величину набегающего газа по выбранной компоненте.
    std::vector<double> primitive(unsigned d) const {
        if (!(mach > 0. && std::isfinite(mach)))
            throw std::invalid_argument("freestream requires gas.txt:mach > 0");
        if (std::abs(theta) > 90. || (d < 3 && theta != 0.))
            throw std::invalid_argument("freestream theta must be in [-90,90] and zero below 3D");
        const double radians = std::acos(-1.) / 180.;
        const double azimuth = std::remainder(phi, 360.) * radians, elevation = theta*radians;
        const double velocity[] = {std::cos(elevation)*std::cos(azimuth), std::cos(elevation)*std::sin(azimuth), std::sin(elevation)};
        if (d == 1 && std::abs(velocity[1]) > 1.e-14)
            throw std::invalid_argument("freestream phi must be a multiple of 180 degrees in 1D");
        std::vector<double> q(d+2, 1.);
        std::copy_n(velocity, d, q.begin()+1);
        q.back() = 1. / mach / gamma / mach;
        if (!(q.back() > 0.) || !std::isfinite(q.back())) throw std::invalid_argument("freestream pressure out of range");
        return q;
    }
};
}
