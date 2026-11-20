#include "../source/turbulence/sa.h"
#include "test_checks.h"
#include <iostream>
using namespace aero::turbulence;
int main() {
    try {
        // Половина демпфирования при chi=cv1 даёт независимую проверку обращения.
        aero_test::require(std::abs(ambient_chi(cv1 / 2.) - cv1) < 2.e-14, "SA midpoint viscosity");
        for (double ratio : {0., 1.e-150, .001, .21, 1., 100., 1.e15}) {
            const double chi = ambient_chi(ratio);
            aero_test::require(std::abs(chi * fv1(chi) - ratio) <= 2.e-14 * ratio,
                               "physical eddy viscosity ratio");
        }
        // При нулевой завихренности и бесконечном расстоянии до стенки источник равен нулю.
        aero_test::require(source_rate(.03, .01, 0., 0.) == 0., "uniform freestream");
        const double growth = cb1 * (1. - 1.2 * std::exp(-4.5)) * 2.;
        aero_test::require(std::abs(source_rate(.03, .01, 2., 0.) - growth) < 1.e-14, "free shear production");
        aero_test::require(source_rate(.03, .01, 0., 10.) < 0., "wall destruction");
        // Независимая формула с pow охватывает обе ветви и весь допустимый интервал r.
        // Густая сетка около g=2 проверяет область наибольшей ошибки начальной оценки.
        for (unsigned i = 0; i <= 12000; ++i) {
            const double r = 10. * i / 12000., g = .7 * r + .3 * std::pow(r, 6);
            const double exact = g * std::pow(65. / (64. + std::pow(g, 6)), 1. / 6.);
            aero_test::require(std::abs(wall_function(g) - exact) <= 2.e-15 * exact, "SA wall function");
        }
        std::cout << "SA freestream, damping, production and wall destruction passed.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
