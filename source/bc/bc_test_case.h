#ifndef AERO_BC_TEST_CASE_H
#define AERO_BC_TEST_CASE_H
#include "bc.h"
namespace aero {
// Один сценарий связывает состояние узла и внешнюю нормаль с ожидаемым потоком или ошибкой.
struct BcTestCase {
    // Граничное условие, воспроизводимое тестом.
    BoundaryConfig config;
    // Размерность тестовой задачи.
    unsigned dimension;
    // Можно ли использовать начальное поле для этой границы.
    bool initial_available;
    // Текущее и начальное состояния, вектор площади и показатель адиабаты.
    double current[5], initial[5], area[3], gamma;
    // Табличные граничные данные тестового примера.
    ProfileState profile;
    // Создаёт таблицу тестового граничного профиля.
    AERO_HOST_DEVICE const ProfileState* profile_data() const {
        return profile.present() ? &profile : nullptr;
    }
    // При ошибочном входе достаточно сравнить код и убедиться, что результат не перезаписан.
    BoundaryStatus expected_status;
    // Эталон потока для проверки результата ядра.
    EulerFluxResult expected;
};
// Сравнивает граничные потоки на CPU и доступных GPU.
int check_bc_devices(const BcTestCase* cases, size_t count, const FreeBoundaryRules* rules, size_t rule_count,
                     bool required);
}
#endif
