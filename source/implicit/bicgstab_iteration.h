#pragma once
#include "bdf1_linear_system.h"
#include "bicgstab_vectors.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aero {
namespace implicit {
// Общая правая предобусловленная итерация для блока произвольной ширины.
// Исполнитель предоставляет рабочие векторы, умножение матрицы, применение
// предобусловливателя и глобальные редукции. Критерий и контроль b−Ax едины.
template<class System>
LinearResult solve_bicgstab(System& s,const double* b,double* x,const LinearTolerance& tol,
                            double rhs_norm,double* shifted_rhs = nullptr) {
    const auto& w = s.work;
    if (!s.ready) throw std::logic_error("BDF: assemble before solving");
    s.partition().local([&] {
        if (s.count && (!b || !x || b == x))
            throw std::invalid_argument("BiCGSTAB: separate input and output required");
    });
    if (!(tol.relative >= 0. && tol.relative < 1. && tol.absolute >= 0.) || !std::isfinite(tol.absolute))
        throw std::invalid_argument("BiCGSTAB: invalid tolerance");
    if (rhs_norm < 0.) rhs_norm = s.norm(b);
    if (!std::isfinite(rhs_norm)) throw std::domain_error("BiCGSTAB: nonfinite right-hand side");
    // Абсолютный допуск защищает почти нулевую правую часть, относительный масштабируется её нормой.
    const double threshold = std::max(tol.absolute, tol.relative * rhs_norm);
    // В итерациях масштабируем невязку, чтобы при малой правой части скалярные
    // произведения не обнулялись. Допуск и итоговый отчёт остаются в исходных единицах.
    const double scale = rhs_norm > 0. ? rhs_norm : 1.;
    // Применяет выбранное обновление рабочих полей текущего этапа решателя.
    const auto update = [&](KrylovStep step, double a, double omega, const double* image) {
        s.partition().local([&] { s.update({w, b, x, image, scale, a, omega, step}); });
    };
    if (shifted_rhs) {
        // Фильтр требует x₀+δ. Начинаем с x₀, решаем A·x=b+A·x₀.
        // Исходная невязка этого уравнения равна b. Сначала сохраняем её
        // в обычных векторах Крылова, затем добавляем образ сдвига к правой
        // части для дальнейших проверок истинной невязки и перезапусков.
        s.apply(x,w.t);
        update(KrylovStep::BeginShifted,0.,0.,nullptr);
        s.partition().local([&] {
            s.update({w,nullptr,shifted_rhs,w.t,scale,0.,0.,KrylovStep::AddImage});
        });
    } else update(KrylovStep::Begin, 0., 0., nullptr);
    if (rhs_norm <= threshold) return {LinearStatus::Converged, 0, rhs_norm};
    // Независимо вычисляет b−A*x, чтобы проверить рекуррентную оценку BiCGSTAB.
    const auto actual_residual = [&]() {
        s.apply(x, w.t);
        // Элемент b−A*x уже находится в регистре. В ту же редукцию включаем
        // его модуль, сохраняя саму невязку для возможного перезапуска BiCGSTAB.
        return s.update_norm({w, b, x, w.t, scale, 0., 0., KrylovStep::TrueResidual});
    };
    unsigned iteration = 0;
    LinearStatus reason = LinearStatus::IterationLimit;
    bool first = true, use_sweep_image = true;
    double old_rho = 1., alpha = 1., omega = 1.;
    while (iteration < tol.max_iterations) {
        ++iteration;
        reason = LinearStatus::Breakdown; // До завершения итерации выход возможен при вырождении.
        const double rho = s.dot(w.shadow, w.r);
        if (!std::isfinite(rho) || rho == 0.) break;
        if (!first) {
            const double beta = (rho/old_rho) * (alpha/omega);
            if (!std::isfinite(beta)) break;
            update(KrylovStep::Direction, beta, omega, nullptr);
        }
        // Первая поправка идёт вдоль M^{-1}p. После неё r хранит промежуточную
        // невязку s из формулы BiCGSTAB. При точном попадании второй шаг не нужен.
        s.apply_preconditioned(w.p, w.v, use_sweep_image);
        const double projection = s.dot(w.shadow, w.v);
        if (!std::isfinite(projection) || projection == 0.) break;
        alpha = rho/projection;
        if (!std::isfinite(alpha)) break;
        double estimate = s.update_norm({w,b,x,w.v,scale,alpha,0.,KrylovStep::Advance});
        if (!std::isfinite(estimate)) break;
        if (estimate > threshold/scale) {
            // Вторая поправка выбирает omega, минимизирующую евклидову норму r−omega*t.
            s.apply_preconditioned(w.r, w.t, use_sweep_image);
            omega = s.omega_projection();
            if (!std::isfinite(omega) || omega == 0.) break;
            estimate = s.update_norm({w,b,x,w.t,scale,omega,0.,KrylovStep::Advance});
            if (!std::isfinite(estimate)) break;
        }
        old_rho = rho;
        first = false;
        if (estimate <= threshold/scale) {
            // Округление накапливается в рекуррентной невязке. Принимаем решение
            // только после независимого умножения A*x. Иначе продолжаем с b−A*x.
            const double actual = actual_residual();
            if (actual <= threshold) return {LinearStatus::Converged, iteration, actual};
            if (!std::isfinite(actual)) break;
            // Если рекуррентная оценка оказалась оптимистичной, следующий цикл
            // использует прямое умножение CSR и для промежуточных направлений.
            use_sweep_image = false;
            update(KrylovStep::Restart, 0., 0., nullptr);
            first = true;
        }
        reason = LinearStatus::IterationLimit;
    }
    const double actual = actual_residual();
    return {actual <= threshold ? LinearStatus::Converged : reason, iteration, actual};
}
}
}
