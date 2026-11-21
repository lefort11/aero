#pragma once

namespace aero {
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
