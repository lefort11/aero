#pragma once

namespace aero {
enum class ResidualLimit { Off, Node, Stencil, Work };
inline const char* residual_limit_name(ResidualLimit limit) {
    switch (limit) {
    case ResidualLimit::Off: return "off";
    case ResidualLimit::Node: return "on";
    case ResidualLimit::Stencil: return "stencil";
    case ResidualLimit::Work: return "work";
    }
    return "";
}
enum class LocalStepMode { Off, On, Auto };

inline const char* local_step_name(LocalStepMode mode) {
    switch (mode) {
    case LocalStepMode::Off: return "off";
    case LocalStepMode::On: return "on";
    case LocalStepMode::Auto: return "auto";
    }
    return "";
}
}
