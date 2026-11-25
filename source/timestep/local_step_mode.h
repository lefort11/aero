#pragma once

namespace aero {
// Настройки изменения узлового псевдошага по двум принятым слоям газа и SA.
struct AdaptiveStepSettings {
    double density = .01;  // Целевое изменение плотности относительно прежней плотности.
    double pressure = .01; // Целевое изменение давления относительно прежнего давления.
    double velocity = .01; // Целевое изменение вектора скорости в долях скорости звука.
    double sa = .05;       // Целевое изменение рабочей вязкости в долях nu + |nu_tilde|.
    double min_factor = .5;   // Наименьший множитель адаптации за один принятый шаг.
    double max_factor = 1.25; // Наибольший множитель роста за один принятый шаг.
};

enum class ResidualLimit { Off, Node, Stencil, Work, Cfl };
// Возвращает текстовое имя режима ограничения по невязке.
inline const char* residual_limit_name(ResidualLimit limit) {
    switch (limit) {
    case ResidualLimit::Off: return "off";
    case ResidualLimit::Node: return "on";
    case ResidualLimit::Stencil: return "stencil";
    case ResidualLimit::Work: return "work";
    case ResidualLimit::Cfl: return "cfl";
    }
    return "";
}
enum class LocalStepMode { Off, On, Auto };

// Возвращает текстовое имя режима локального шага.
inline const char* local_step_name(LocalStepMode mode) {
    switch (mode) {
    case LocalStepMode::Off: return "off";
    case LocalStepMode::On: return "on";
    case LocalStepMode::Auto: return "auto";
    }
    return "";
}
}
