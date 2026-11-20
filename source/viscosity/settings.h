#pragma once
#include <string>
namespace aero {
// Данные для температурного закона динамической вязкости идеального газа.
struct ViscositySettings {
    std::string law = "off";        // Выбранный в main.txt закон: off, normal или sutherland.
    double reynolds = 0.;           // Число Рейнольдса по опорным плотности, скорости, длине и вязкости при t_amb.
    double prandtl = 0.;            // Молекулярное число Прандтля для теплопроводности газа.
    double t_amb = 0.;              // Абсолютная опорная температура газа, К.
    double sutherland_s = 0.;       // Постоянная Сазерленда в тех же единицах температуры.
};

}
