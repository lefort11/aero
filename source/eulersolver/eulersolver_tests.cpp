#include "lax_friedrichs.h"
#include "../../tests/test_checks.h"
#include "roe.h"
#include "linear_euler.h"
#include "characteristic.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#ifdef AERO_WITH_KOKKOS_GPU
// Возвращает число устройств, доступных GPU-проверкам потоков.
extern "C" int euler_test_device_count(int*, char*, unsigned);
// Запускает выбранное потоковое ядро на GPU и возвращает результат проверки.
extern "C" int euler_test_device_flux(int, unsigned, unsigned, const double*, const double*, const double*,
                                    const double*, const double*, double, aero::EulerFluxResult*, int*, char*,
                                    unsigned);
#endif

namespace {
using namespace aero;
typedef std::vector<double> Values;
std::vector<std::pair<std::string, std::function<void()>>> tests;
using aero_test::require;
// Проверяет совпадение чисел в пределах заданного допуска.
void near(double actual, double expected, double tolerance = 3.e-11) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)),
            "Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
// Проверяет условие теста и сообщает понятную причину нарушения.
void ok(EulerStatus status) {
    require(status == EulerStatus::Success, euler_status_message(status));
}
// Добавляет именованную проверку в список тестов модуля.
void add(const std::string& name, std::function<void()> test) {
    tests.push_back(std::make_pair(name, test));
}
// Сравнивает результаты двух реализаций или представлений с заданным допуском.
void same(const EulerFluxResult& a, const EulerFluxResult& b) {
    require(a.components == b.components, "Component count mismatch");
    near(a.max_wave_speed, b.max_wave_speed);
    for (unsigned k = 0; k < 5; ++k)
        near(a.flux[k], b.flux[k]);
}
// Создаёт консервативное газовое состояние из тестовых примитивных величин.
Values state(double rho, const Values& velocity, double pressure, double gamma = 1.4) {
    Values u(velocity.size() + 2);
    u[0] = rho;
    double kinetic = 0;
    for (std::size_t k = 0; k < velocity.size(); ++k) {
        u[k + 1] = rho * velocity[k];
        kinetic += 0.5 * rho * velocity[k] * velocity[k];
    }
    u.back() = pressure / (gamma - 1) + kinetic;
    return u;
}
// Задаёт направление нормали для проверки многомерного потока.
Values direction(unsigned d) {
    if (d == 1)
        return Values{1};
    if (d == 2)
        return Values{3, 4};
    return Values{2, -3, 6};
}
// Формирует тестовый вектор скорости активной размерности.
Values velocity(unsigned d, double scale) {
    Values v(d);
    for (unsigned k = 0; k < d; ++k)
        v[k] = scale * (1 + 0.3 * k) * (k % 2 ? -1 : 1);
    return v;
}
// Вычисляет физический поток тестового состояния и проверяет успешность газодинамического ядра.
Values physical(const Values& u, const Values& n, double gamma = 1.4) {
    Values f(u.size());
    double speed = 0;
    ok(euler_flux(static_cast<unsigned>(n.size()), u.data(), n.data(), gamma, f.data(), &speed));
    return f;
}
// Независимый эталон проекции для теста: решаем R·W = U методом исключения с выбором ведущего элемента.
Values characteristic(const EulerRoeBasis& basis, const Values& u) {
    const unsigned count = basis.dimension + 2;
    aero::FixedMatrix<double,5,6> a = {};
    for (unsigned r = 0; r < count; ++r) {
        for (unsigned c = 0; c < count; ++c)
            a(r, c) = basis.right[5 * r + c];
        a(r, count) = u[r];
    }
    for (unsigned c = 0; c < count; ++c) {
        unsigned pivot = c;
        for (unsigned r = c + 1; r < count; ++r)
            if (std::abs(a(r, c)) > std::abs(a(pivot, c)))
                pivot = r;
        require(std::abs(a(pivot, c)) > 1.e-13, "Singular test Roe matrix");
        for (unsigned j = 0; j <= count; ++j)
            std::swap(a(c, j), a(pivot, j));
        const double divisor = a(c, c);
        for (unsigned j = c; j <= count; ++j)
            a(c, j) /= divisor;
        for (unsigned r = 0; r < count; ++r)
            if (r != c) {
                const double factor = a(r, c);
                for (unsigned j = c; j <= count; ++j)
                    a(r, j) -= factor * a(c, j);
            }
    }
    Values w(count);
    for (unsigned k = 0; k < count; ++k)
        w[k] = a(k, count);
    return w;
}
// В тесте отдельно соединяем обратное преобразование и вычисление потока выбранной схемой.
EulerStatus test_characteristic_lf(unsigned d, const double* wl, const double* wr, const double* ui,
                                   const double* uj, const double* n, double gamma, EulerFluxResult* result) {
    EulerRoeBasis basis;
    EulerStatus status = euler_roe_basis(d, ui, uj, n, gamma, &basis);
    if (status != EulerStatus::Success)
        return status;
    double ul[5], ur[5];
    status = characteristic_to_conservative(basis, wl, ul);
    if (status != EulerStatus::Success)
        return status;
    status = characteristic_to_conservative(basis, wr, ur);
    if (status != EulerStatus::Success)
        return status;
    return lax_friedrichs_conservative(d, ul, ur, n, gamma, result);
}
// В тесте отдельно соединяем обратное преобразование и вычисление потока выбранной схемой.
EulerFluxResult test_characteristic_lf(const Values& wl, const Values& wr, const Values& ui, const Values& uj,
                                       const Values& n, double gamma) {
    EulerFluxResult result;
    ok(test_characteristic_lf(static_cast<unsigned>(n.size()), wl.data(), wr.data(), ui.data(), uj.data(),
                              n.data(), gamma, &result));
    return result;
}
// Вычисляет независимый эталон потока Роу для сравнения с ядром.
EulerFluxResult reference_roe(const Values& ul, const Values& ur, const Values& n, double gamma) {
    const unsigned d = static_cast<unsigned>(n.size());
    EulerRoeBasis basis;
    ok(euler_roe_basis(d, ul.data(), ur.data(), n.data(), gamma, &basis));
    Values jump(ul.size());
    for (std::size_t k = 0; k < jump.size(); ++k)
        jump[k] = ur[k] - ul[k];
    const Values amplitudes = characteristic(basis, jump), fl = physical(ul, n, gamma),
                 fr = physical(ur, n, gamma);
    EulerFluxResult result = lax_friedrichs_conservative(ul, ur, n, gamma);
    for (unsigned r = 0; r < d + 2; ++r) {
        double dissipation = 0;
        for (unsigned c = 0; c < d + 2; ++c)
            dissipation += basis.right[5 * r + c] * std::abs(basis.eigenvalues[c]) * amplitudes[c];
        result.flux[r] = 0.5 * (fl[r] + fr[r] - dissipation);
        result.max_wave_speed = (std::max)(result.max_wave_speed, std::abs(basis.eigenvalues[r]));
    }
    return result;
}
// Заполняет результат контрольными значениями до вызова проверяемого ядра.
EulerFluxResult sentinel() {
    EulerFluxResult r = {};
    r.components = 99;
    r.max_wave_speed = 77;
    for (unsigned k = 0; k < 5; ++k)
        r.flux[k] = 100 + k;
    return r;
}
// Проверяет сохранность контрольных значений после неуспешного вызова.
void unchanged(const EulerFluxResult& r) {
    require(r.components == 99 && r.max_wave_speed == 77, "Failed call changed metadata");
    for (unsigned k = 0; k < 5; ++k)
        require(r.flux[k] == 100 + k, "Failed call changed output");
}

// Проверяет действие диссипации Роу на характеристических направлениях.
void roe_eigenvector_test(unsigned d) {
    const double gamma = 1.67;
    const Values ui = state(2, velocity(d, 0.7), 1.3, gamma), uj = state(0.5, velocity(d, -0.4), 0.8, gamma),
                 n = direction(d);
    EulerRoeBasis basis;
    ok(euler_roe_basis(d, ui.data(), uj.data(), n.data(), gamma, &basis));
    const double a = std::sqrt(ui[0]) / (std::sqrt(ui[0]) + std::sqrt(uj[0])), b = 1 - a;
    double kin_i = 0, kin_j = 0, kin = 0, norm = 0;
    for (unsigned k = 0; k < d; ++k) {
        const double vi = ui[k + 1] / ui[0], vj = uj[k + 1] / uj[0];
        near(basis.velocity[k], a * vi + b * vj);
        kin_i += 0.5 * vi * vi;
        kin_j += 0.5 * vj * vj;
        kin += 0.5 * basis.velocity[k] * basis.velocity[k];
        norm += n[k] * n[k];
    }
    const double h =
        a * (gamma * 1.3 / ((gamma - 1) * ui[0]) + kin_i) + b * (gamma * 0.8 / ((gamma - 1) * uj[0]) + kin_j);
    near(basis.enthalpy, h);
    near(basis.sound_speed, std::sqrt((gamma - 1) * (h - kin)));
    Values unit = n;
    for (double& x : unit)
        x /= std::sqrt(norm);
    const Values q = state(1, Values(basis.velocity, basis.velocity + d),
                           basis.sound_speed * basis.sound_speed / gamma, gamma);
    const double pressure = basis.sound_speed * basis.sound_speed / gamma;
    double un = 0;
    for (unsigned k = 0; k < d; ++k)
        un += basis.velocity[k] * unit[k];
    for (unsigned column = 0; column < d + 2; ++column) {
        // Аналитическая производная потока Эйлера вдоль заданного направления для независимой проверки базиса.
        const double dr = basis.right[column], de = basis.right[5 * (d + 1) + column];
        double dot_m = 0, velocity_dot_m = 0;
        for (unsigned k = 0; k < d; ++k) {
            const double dm = basis.right[5 * (k + 1) + column];
            dot_m += dm * unit[k];
            velocity_dot_m += basis.velocity[k] * dm;
        }
        const double dp = (gamma - 1) * (de - velocity_dot_m + kin * dr), du = dot_m - un * dr;
        near(dot_m, basis.eigenvalues[column] * dr);
        for (unsigned k = 0; k < d; ++k)
            near(basis.right[5 * (k + 1) + column] * un + q[k + 1] * du + dp * unit[k],
                 basis.eigenvalues[column] * basis.right[5 * (k + 1) + column]);
        near((de + dp) * un + (q.back() + pressure) * du, basis.eigenvalues[column] * de);
    }
}

#ifdef AERO_WITH_KOKKOS_GPU
// Сравнивает CPU- и GPU-результаты одного газодинамического сценария.
void gpu_case(int device, unsigned mode, unsigned d, const Values& left, const Values& right,
              const Values& aux_left, const Values& aux_right, const Values& n, double gamma,
              EulerStatus expected_status, const EulerFluxResult* expected) {
    EulerFluxResult r = sentinel();
    int status = -1;
    char error[512] = {};
    const int failed = euler_test_device_flux(
        device, mode, d, left.data(), right.data(), aux_left.empty() ? nullptr : aux_left.data(),
        aux_right.empty() ? nullptr : aux_right.data(), n.data(), gamma, &r, &status, error, sizeof(error));
    require(!failed, error);
    require(static_cast<EulerStatus>(status) == expected_status, "CUDA solver status mismatch");
    if (expected)
        same(r, *expected);
    else
        unchanged(r);
}
#endif
}

// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main(int argc, char** argv) {
    const bool require_cuda = argc == 2 && aero_test::require_gpu_argument(argv[1]);
    if (argc > 2 || (argc == 2 && !require_cuda)) {
        std::cerr << "Usage: eulersolver_tests [--require-gpu]\n";
        return 2;
    }
    add("linear Euler agrees with directional derivative of nonlinear LF and Roe", [] {
        const double gamma = 1.4, h = 1.e-5;
        for (unsigned d=1; d<=3; ++d) {
            LinearBackground gas;
            gas.rho = 1.3;
            gas.p = .9;
            gas.sound = std::sqrt(gamma*gas.p/gas.rho);
            Values n(d), velocity(d);
            for (unsigned j=0; j<d; ++j) {
                velocity[j] = gas.velocity[j] = .12*(j+1);
                n[j] = (j%2 ? -1. : 1.)/std::sqrt(double(d));
            }
            const LinearFace face(gas,d,n.data());
            Values left(d+2,.13), right(d+2,-.08), fl(d+2), fr(d+2);
            left[0] = -.4;
            right[d+1] = .22;
            face.physical(left.data(),fl.data());
            face.physical(right.data(),fr.data());
            // Создаёт малое возмущение состояния для независимой проверки характеристик.
            const auto perturbed = [&](const Values& q, double step) {
                Values u = velocity;
                for (unsigned j=0; j<d; ++j) u[j] += step*q[j+1];
                return state(gas.rho+step*q[0],u,gas.p+step*q[d+1],gamma);
            };
            for (bool roe : {false,true}) {
                EulerFluxResult expected[2], actual{};
                for (unsigned side=0; side<2; ++side) {
                    const double step = side ? h : -h;
                    const auto l = perturbed(left,step), r = perturbed(right,step);
                    expected[side] = roe ? roe_conservative(l,r,n,gamma) : lax_friedrichs_conservative(l,r,n,gamma);
                }
                require(face.numerical(roe,1.,left.data(),right.data(),fl.data(),fr.data(),actual),"linear flux");
                gas.to_conservative(d,actual.flux,gamma);
                for (unsigned k=0; k<d+2; ++k)
                    near(actual.flux[k],(expected[1].flux[k]-expected[0].flux[k])/(2*h),3.e-6);
            }
        }
    });
    add("external characteristic inverse transforms states and increments", [] {
        for (unsigned d = 1; d <= 3; ++d) {
            const Values ui = state(1, velocity(d, 0.3), 1), uj = state(2, velocity(d, -0.2), 0.7),
                         n = direction(d);
            EulerRoeBasis basis;
            ok(euler_roe_basis(d, ui.data(), uj.data(), n.data(), 1.4, &basis));
            const Values target = state(0.7, velocity(d, 2), 0.6), w = characteristic(basis, target);
            const Values converted = characteristic_to_conservative(basis, w);
            for (unsigned k = 0; k < d + 2; ++k)
                near(converted[k], target[k]);
            Values jump(d + 2);
            for (unsigned k = 0; k < d + 2; ++k)
                jump[k] = ui[k] - uj[k];
            const auto restored = characteristic_to_conservative(basis, characteristic(basis, jump));
            for (unsigned k = 0; k < d + 2; ++k)
                near(restored[k], jump[k]);
            double out[5] = {77, 77, 77, 77, 77};
            Values invalid = w;
            invalid[0] = (std::numeric_limits<double>::quiet_NaN)();
            require(characteristic_to_conservative(basis, invalid.data(), out) == EulerStatus::NonFinite,
                    "Inverse finite check");
            for (double x : out)
                require(x == 77, "Failed inverse wrote output");
        }
    });
    add("Roe Sod flux has independent known values", [] {
        const auto r = roe_conservative(Values{1, 0, 2.5}, Values{0.125, 0, 0.25}, Values{1}, 1.4);
        near(r.flux[0], 0.39066048578596296);
        near(r.flux[1], 0.55);
        near(r.flux[2], 1.295882277373112);
    });
    add("Roe resolves stationary contact and shear without LF smearing", [] {
        for (unsigned d = 1; d <= 3; ++d) {
            Values n(d, 0), vl(d, 0), vr(d, 0);
            n[0] = 1;
            if (d > 1) {
                vl[1] = 1.5;
                vr[1] = -0.8;
            }
            const Values ul = state(1, vl, 1), ur = state(2, vr, 1);
            const auto r = roe_conservative(ul, ur, n, 1.4), lf = lax_friedrichs_conservative(ul, ur, n, 1.4);
            for (unsigned k = 0; k < d + 2; ++k)
                near(r.flux[k], k == 1 ? 1 : 0);
            require(std::abs(lf.flux[0]) > 0.1, "Contact fixture must distinguish LF from Roe");
        }
    });
    add("Roe resolves a stationary normal shock", [] {
        const double gamma = 1.4, ul = 2 * std::sqrt(gamma), rho_r = 8.0 / 3;
        const Values left = state(1, Values{ul}, 1), right = state(rho_r, Values{ul / rho_r}, 4.5), n = {1};
        const auto r = roe_conservative(left, right, n, gamma);
        const auto fl = physical(left, n), fr = physical(right, n);
        for (unsigned k = 0; k < 3; ++k) {
            near(fl[k], fr[k]);
            near(r.flux[k], fl[k]);
        }
    });
    for (unsigned d = 1; d <= 3; ++d) {
        add(std::to_string(d) + "D Roe independent matrix dissipation, reversal and input modes", [d] {
            const Values ul = state(0.8, velocity(d, 1.2), 0.6), ur = state(1.7, velocity(d, -0.7), 2.3),
                         n = direction(d);
            const auto r = roe_conservative(ul, ur, n, 1.4);
            same(r, reference_roe(ul, ur, n, 1.4));
            same(r, roe_fluxes(physical(ul, n), physical(ur, n), ul, ur, n, 1.4));
            Values reverse = n, scaled = n;
            for (double& v : reverse)
                v = -v;
            for (double& v : scaled)
                v *= 7;
            const auto back = roe_conservative(ur, ul, reverse, 1.4);
            near(back.max_wave_speed, r.max_wave_speed);
            for (unsigned k = 0; k < d + 2; ++k)
                near(back.flux[k], -r.flux[k]);
            same(r, roe_conservative(ul, ur, scaled, 1.4));
            // При реконструкции базис строится по узлам, а в потоке Роу — по восстановленным состояниям на грани.
            const Values ui = state(2, velocity(d, 0.1), 1), uj = state(1, velocity(d, -0.2), 0.9);
            EulerRoeBasis frozen;
            ok(euler_roe_basis(d, ui.data(), uj.data(), n.data(), 1.4, &frozen));
            same(r, roe_conservative(characteristic_to_conservative(frozen, characteristic(frozen, ul)),
                                     characteristic_to_conservative(frozen, characteristic(frozen, ur)), n,
                                     1.4));
            Values fl = physical(ul, n), fr = physical(ur, n);
            fl[0] += 8;
            fr.back() -= 6;
            const auto changed = roe_fluxes(fl, fr, ul, ur, n, 1.4);
            near(changed.flux[0], r.flux[0] + 4);
            near(changed.flux[d + 1], r.flux[d + 1] - 3);
        });
        add(std::to_string(d) + "D Roe equal states and supersonic upwinding", [d] {
            Values n(d, 0);
            n[0] = 1;
            for (double sign : Values{-1, 1}) {
                Values vl = velocity(d, 0.3), vr = velocity(d, -0.2);
                vl[0] = 5 * sign;
                vr[0] = 4 * sign;
                const Values ul = state(1, vl, 1), ur = state(0.8, vr, 0.7);
                const auto r = roe_conservative(ul, ur, n, 1.4);
                const Values f = physical(sign > 0 ? ul : ur, n);
                for (unsigned k = 0; k < d + 2; ++k)
                    near(r.flux[k], f[k]);
                same(roe_conservative(ul, ul, n, 1.4), lax_friedrichs_conservative(ul, ul, n, 1.4));
            }
        });
    }
    add("Roe speed includes averaged acoustic waves and invalid calls preserve output", [] {
        const Values ul = state(1, Values{0, 10}, 1), ur = state(1, Values{0, -10}, 1), n = {1, 0};
        const auto r = roe_conservative(ul, ur, n, 1.4);
        near(r.max_wave_speed, std::sqrt(0.4 * 53.5));
        EulerFluxResult output = sentinel();
        const double good[3] = {1, 0, 2.5}, bad[3] = {1, 3, 1}, one[1] = {1};
        require(roe_conservative(1, good, bad, one, 1.4, &output) == EulerStatus::InvalidPressure,
                "Roe pressure check");
        unchanged(output);
        require(roe_conservative(0, good, good, one, 1.4, &output) == EulerStatus::InvalidDimension,
                "Roe dimension check");
        unchanged(output);
        require(roe_conservative(1, nullptr, good, one, 1.4, &output) == EulerStatus::NullPointer,
                "Roe pointer check");
        unchanged(output);
        const double invalid[3] = {1, 0, (std::numeric_limits<double>::infinity)()};
        require(roe_fluxes(1, invalid, good, good, good, one, 1.4, &output) == EulerStatus::NonFinite,
                "Roe flux check");
        unchanged(output);
    });
    add("1D physical flux and equal-state consistency", [] {
        const Values u = {2, 6, 19}, n = {1};
        const auto r = lax_friedrichs_conservative(u, u, n, 1.4);
        require(r.components == 3, "1D component count");
        near(r.flux[0], 6);
        near(r.flux[1], 22);
        near(r.flux[2], 69);
        near(r.max_wave_speed, 3 + std::sqrt(2.8));
        require(r.flux[3] == 0 && r.flux[4] == 0, "Inactive entries");
    });
    add("Sod interface has independent analytic LF flux", [] {
        const auto r = lax_friedrichs_conservative(Values{1, 0, 2.5}, Values{0.125, 0, 0.25}, Values{1}, 1.4);
        near(r.flux[0], 0.4375 * std::sqrt(1.4));
        near(r.flux[1], 0.55);
        near(r.flux[2], 1.125 * std::sqrt(1.4));
        near(r.max_wave_speed, std::sqrt(1.4));
    });
    add("3D oblique equal-state flux includes tangential momentum", [] {
        const Values u = state(2, Values{3, -1, 2}, 4), n = {0, 3, 4};
        const auto r = lax_friedrichs_conservative(u, u, n, 1.4);
        // Единичная нормаль n = (0, 0.6, 0.8), нормальная скорость равна 1, энергия E = 24.
        near(r.flux[0], 2);
        near(r.flux[1], 6);
        near(r.flux[2], 0.4);
        near(r.flux[3], 7.2);
        near(r.flux[4], 28);
        near(r.max_wave_speed, 1 + std::sqrt(2.8));
    });
    for (unsigned d = 1; d <= 3; ++d) {
        add(std::to_string(d) + "D Roe averages and A*R=R*Lambda", [d] { roe_eigenvector_test(d); });
        add(std::to_string(d) + "D three input forms agree on reconstructed states", [d] {
            const Values n = direction(d), ui = state(2, velocity(d, 0.3), 1),
                         uj = state(0.5, velocity(d, -0.2), 0.7);
            const Values ul = state(0.8, velocity(d, 4), 0.2), ur = state(1.3, velocity(d, -2), 3);
            EulerRoeBasis basis;
            ok(euler_roe_basis(d, ui.data(), uj.data(), n.data(), 1.4, &basis));
            const auto expected = lax_friedrichs_conservative(ul, ur, n, 1.4);
            same(test_characteristic_lf(characteristic(basis, ul), characteristic(basis, ur), ui, uj, n, 1.4),
                 expected);
            same(lax_friedrichs_fluxes(physical(ul, n), physical(ur, n), ul, ur, n, 1.4), expected);
            require(expected.max_wave_speed > basis.sound_speed, "Speed must come from reconstructed states");
        });
        add(std::to_string(d) + "D flux reversal and direction scaling", [d] {
            const Values ul = state(2, velocity(d, 0.7), 1), ur = state(0.5, velocity(d, -0.3), 0.7),
                         n = direction(d);
            const auto a = lax_friedrichs_conservative(ul, ur, n, 1.4);
            Values reverse = n, large = n, small = n;
            for (unsigned k = 0; k < d; ++k) {
                reverse[k] *= -1;
                large[k] *= 1.e250;
                small[k] *= 1.e-250;
            }
            const auto b = lax_friedrichs_conservative(ur, ul, reverse, 1.4);
            near(a.max_wave_speed, b.max_wave_speed);
            for (unsigned k = 0; k < d + 2; ++k)
                near(a.flux[k], -b.flux[k]);
            same(a, lax_friedrichs_conservative(ul, ur, large, 1.4));
            same(a, lax_friedrichs_conservative(ul, ur, small, 1.4));
        });
    }
    add("1D Roe matrix has documented acoustic and entropy normalization", [] {
        const Values ui = state(1, Values{2}, 1), n = {1};
        EulerRoeBasis basis;
        ok(euler_roe_basis(1, ui.data(), ui.data(), n.data(), 1.4, &basis));
        const double c = std::sqrt(1.4), h = 5.5;
        const aero::FixedMatrix<double,3,3> expected = {{1, 1, 1, 2 - c, 2, 2 + c, h - 2 * c, 2, h + 2 * c}};
        for (unsigned r = 0; r < 3; ++r)
            for (unsigned k = 0; k < 3; ++k)
                near(basis.right[5 * r + k], expected(r, k));
    });
    add("supplied fluxes are used rather than recomputed", [] {
        const Values ul = state(1, Values{1}, 1), ur = state(2, Values{-0.3}, 0.7), n = {1};
        Values fl = physical(ul, n), fr = physical(ur, n);
        const auto expected = lax_friedrichs_conservative(ul, ur, n, 1.4);
        fl[0] += 8;
        fr[2] -= 6;
        const auto r = lax_friedrichs_fluxes(fl, fr, ul, ur, n, 1.4);
        near(r.flux[0], expected.flux[0] + 4);
        near(r.flux[1], expected.flux[1]);
        near(r.flux[2], expected.flux[2] - 3);
        near(r.max_wave_speed, expected.max_wave_speed);
    });
    add("2D rotation covariance", [] {
        const Values ul = state(1, Values{2, -0.5}, 1), ur = state(2, Values{-1, 0.3}, 0.7), n = {2, 1};
        Values vl = ul, vr = ur, nn = n;
        vl[1] = 0.6 * ul[1] - 0.8 * ul[2];
        vl[2] = 0.8 * ul[1] + 0.6 * ul[2];
        vr[1] = 0.6 * ur[1] - 0.8 * ur[2];
        vr[2] = 0.8 * ur[1] + 0.6 * ur[2];
        nn[0] = 0.6 * n[0] - 0.8 * n[1];
        nn[1] = 0.8 * n[0] + 0.6 * n[1];
        const auto a = lax_friedrichs_conservative(ul, ur, n, 1.4),
                   b = lax_friedrichs_conservative(vl, vr, nn, 1.4);
        near(a.flux[0], b.flux[0]);
        near(a.flux[3], b.flux[3]);
        near(a.max_wave_speed, b.max_wave_speed);
        near(b.flux[1], 0.6 * a.flux[1] - 0.8 * a.flux[2]);
        near(b.flux[2], 0.8 * a.flux[1] + 0.6 * a.flux[2]);
    });
    add("invalid inputs preserve outputs without density or pressure floors", [] {
        const double good[3] = {1, 0, 2.5}, normal[1] = {1}, zero[1] = {0};
        EulerFluxResult r = sentinel();
        require(lax_friedrichs_conservative(0, good, good, normal, 1.4, &r) == EulerStatus::InvalidDimension,
                "Dimension check");
        unchanged(r);
        require(lax_friedrichs_conservative(4, good, good, normal, 1.4, &r) == EulerStatus::InvalidDimension,
                "Dimension check");
        unchanged(r);
        require(lax_friedrichs_conservative(1, nullptr, good, normal, 1.4, &r) == EulerStatus::NullPointer,
                "Pointer check");
        unchanged(r);
        require(lax_friedrichs_conservative(1, good, good, zero, 1.4, &r) == EulerStatus::InvalidDirection,
                "Direction check");
        unchanged(r);
        for (double gamma : Values{1, 0, (std::numeric_limits<double>::quiet_NaN)()}) {
            require(lax_friedrichs_conservative(1, good, good, normal, gamma, &r) ==
                        EulerStatus::InvalidGamma,
                    "Gamma check");
            unchanged(r);
        }
        for (double rho : Values{0, -1}) {
            const double u[3] = {rho, 0, 2.5};
            require(lax_friedrichs_conservative(1, u, good, normal, 1.4, &r) == EulerStatus::InvalidDensity,
                    "Density check");
            unchanged(r);
        }
        const double pressure_bad[3] = {1, 3, 1},
                     infinite[3] = {1, 0, (std::numeric_limits<double>::infinity)()};
        require(lax_friedrichs_conservative(1, good, pressure_bad, normal, 1.4, &r) ==
                    EulerStatus::InvalidPressure,
                "Pressure check");
        unchanged(r);
        require(lax_friedrichs_conservative(1, infinite, good, normal, 1.4, &r) == EulerStatus::NonFinite,
                "Finite check");
        unchanged(r);
        require(lax_friedrichs_fluxes(1, infinite, good, good, good, normal, 1.4, &r) ==
                    EulerStatus::NonFinite,
                "Flux finite check");
        unchanged(r);
        require(test_characteristic_lf(1, infinite, good, good, good, normal, 1.4, &r) ==
                    EulerStatus::NonFinite,
                "Characteristic finite check");
        unchanged(r);
        const double negative_w[3] = {0, -1, 0};
        require(test_characteristic_lf(1, negative_w, good, good, good, normal, 1.4, &r) ==
                    EulerStatus::InvalidDensity,
                "Back-transform physical check");
        unchanged(r);
    });
    add("checked CPU adapters reject vector size and invalid state", [] {
        bool rejected = false;
        try {
            lax_friedrichs_conservative(Values{1, 0}, Values{1, 0, 2.5}, Values{1}, 1.4);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "Size check");
        rejected = false;
        try {
            lax_friedrichs_conservative(Values{0, 0, 2.5}, Values{1, 0, 2.5}, Values{1}, 1.4);
        } catch (const EulerError& e) {
            rejected = e.status() == EulerStatus::InvalidDensity;
        }
        require(rejected, "Physical status check");
    });
    add("single-call output is bounded and shared functions are thread safe", [] {
        const Values u = state(1, Values{0.7, -0.3}, 1), v = state(2, Values{-0.1, 0.5}, 0.8), n = {3, 4};
        const auto expected = lax_friedrichs_conservative(u, v, n, 1.4);
        // Контрольные значения вокруг результата для выявления записи за границы.
        struct Guard {
            double before;
            EulerFluxResult result;
            double after;
        } guarded = {123, {}, 456};
        ok(lax_friedrichs_conservative(2, u.data(), v.data(), n.data(), 1.4, &guarded.result));
        require(guarded.before == 123 && guarded.after == 456, "Output bounds");
        same(guarded.result, expected);
        std::vector<EulerFluxResult> outputs(1024);
        std::vector<EulerStatus> statuses(1024);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int k = 0; k < 1024; ++k)
            statuses[static_cast<std::size_t>(k)] = lax_friedrichs_conservative(
                2, u.data(), v.data(), n.data(), 1.4, &outputs[static_cast<std::size_t>(k)]);
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            ok(statuses[k]);
            same(outputs[k], expected);
        }
    });
    try {
        int devices = 0;
#ifdef AERO_WITH_KOKKOS_GPU
        char error[512] = {};
        require(!euler_test_device_count(&devices, error, sizeof(error)), error);
        for (int device = 0; device < devices; ++device) {
            add("GPU " + std::to_string(device) + " all dimensions and input forms", [device] {
                for (unsigned d = 1; d <= 3; ++d)
                    for (double gamma : Values{1.4, 1.67}) {
                        std::vector<Values> directions = {direction(d)};
                        Values reverse = direction(d);
                        for (double& x : reverse)
                            x = -x;
                        directions.push_back(reverse);
                        if (d == 3) {
                            directions.push_back(Values{1, 1, 1});
                            directions.push_back(Values{1, std::nextafter(1.0, 0.0), 1});
                        }
                        for (const Values& n : directions) {
                            const Values ui = state(2, velocity(d, 0.3), 1, gamma),
                                         uj = state(0.5, velocity(d, -0.2), 0.7, gamma);
                            const Values ul = state(0.8, velocity(d, 3), 0.2, gamma),
                                         ur = state(1.3, velocity(d, -1), 3, gamma);
                            const auto expected = lax_friedrichs_conservative(ul, ur, n, gamma);
                            gpu_case(device, 0, d, ul, ur, {}, {}, n, gamma, EulerStatus::Success, &expected);
                            EulerRoeBasis basis;
                            ok(euler_roe_basis(d, ui.data(), uj.data(), n.data(), gamma, &basis));
                            gpu_case(device, 1, d, characteristic(basis, ul), characteristic(basis, ur), ui,
                                     uj, n, gamma, EulerStatus::Success, &expected);
                            Values fl = physical(ul, n, gamma), fr = physical(ur, n, gamma);
                            fl[0] += 0.7;
                            const auto expected_flux = lax_friedrichs_fluxes(fl, fr, ul, ur, n, gamma);
                            gpu_case(device, 2, d, fl, fr, ul, ur, n, gamma, EulerStatus::Success,
                                     &expected_flux);
                            const auto expected_roe = roe_conservative(ul, ur, n, gamma);
                            gpu_case(device, 3, d, ul, ur, {}, {}, n, gamma, EulerStatus::Success,
                                     &expected_roe);
                            gpu_case(device, 4, d, characteristic(basis, ul), characteristic(basis, ur), ui,
                                     uj, n, gamma, EulerStatus::Success, &expected_roe);
                            const auto expected_roe_flux = roe_fluxes(fl, fr, ul, ur, n, gamma);
                            gpu_case(device, 5, d, fl, fr, ul, ur, n, gamma, EulerStatus::Success,
                                     &expected_roe_flux);
                        }
                    }
            });
            add("GPU " + std::to_string(device) + " rejects invalid states without writing output",
                [device] {
                    const Values good = {1, 0, 2.5}, n = {1};
                    gpu_case(device, 0, 1, Values{0, 0, 1}, good, {}, {}, n, 1.4, EulerStatus::InvalidDensity,
                             nullptr);
                    gpu_case(device, 0, 1, good, good, {}, {}, Values{0}, 1.4, EulerStatus::InvalidDirection,
                             nullptr);
                    gpu_case(device, 1, 1, Values{0, -1, 0}, good, good, good, n, 1.4,
                             EulerStatus::InvalidDensity, nullptr);
                    gpu_case(device, 2, 1, good, Values{0, 0, (std::numeric_limits<double>::quiet_NaN)()},
                             good, good, n, 1.4, EulerStatus::NonFinite, nullptr);
                    gpu_case(device, 3, 1, Values{0, 0, 1}, good, {}, {}, n, 1.4, EulerStatus::InvalidDensity,
                             nullptr);
                    gpu_case(device, 4, 1, Values{0, -1, 0}, good, good, good, n, 1.4,
                             EulerStatus::InvalidDensity, nullptr);
                    gpu_case(device, 5, 1, good, Values{0, 0, (std::numeric_limits<double>::quiet_NaN)()},
                             good, good, n, 1.4, EulerStatus::NonFinite, nullptr);
                });
        }
#endif
        require(!require_cuda || devices > 0, "GPU verification requested but no GPU device available");
        unsigned passed = 0;
        for (const auto& test : tests) {
            try {
                test.second();
                ++passed;
                std::cout << "PASS " << test.first << '\n';
            } catch (const std::exception& e) {
                std::cerr << "FAIL " << test.first << ": " << e.what() << '\n';
                return 1;
            }
        }
        std::cout << passed << " eulersolver tests passed; GPU devices tested: " << devices << '\n';
        if (!devices)
            std::cout << "CUDA runtime tests skipped; CPU remains available.\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
