#include "../../tests/test_mesh.h"
#include "bdf1_linear_system.h"
#include "spatial_jacobian.h"
#include "block_layout.h"
#include "../residual/residual.h"
#include "../../tests/test_field.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace aero;
using namespace aero::implicit;
using aero_test::TestField;
using aero_test::require;
using Vector = std::vector<double>;

// Берём известную поправку x*, отдельно строим J*x* и временной вклад V*x*/dt.
// Это обнаруживает как потерю временного блока, так и повторное прибавление при пересборке.
void check_mesh(unsigned d, int device) {
    const Mesh mesh = aero_test::test_mesh(d);
    MedianDual dual(mesh);
    const auto cv = dual.view();
    const unsigned width = d + 2;
    const auto size = cv.node_count * width;
    ReconstructionOptions reconstruction;
    reconstruction.scheme = ReconstructionScheme::FirstOrder;
    BoundaryPatches patches;
    patches.by_patch.resize(mesh.data().patch_names.size());
    SpatialResidual op(mesh.view(), cv, prepare_reconstruction(mesh, reconstruction), patches,
                       residual_options("characteristic","lf",1.4), device);
    BdfLinearSystem system(op);
    SpatialJacobian spatial(op);
    TestField state(device,size), volumes(device,cv.node_count), x(device,size), b(device,size), ax(device,size);
    Vector u(size), exact(size);
    Vector volume(cv.volumes,cv.volumes+cv.node_count);
    for (MeshIndex i = 0; i < cv.node_count; ++i) {
        u[i*width] = 1. + .1 * std::sin(.2 * static_cast<double>(i));
        for (unsigned k = 1; k <= d; ++k) u[i*width+k] = .18*k;
        u[i*width+d+1] = 3.;
        for (unsigned k = 0; k < width; ++k)
            exact[i*width+k] = std::cos(.19 * static_cast<double>(i*width+k));
    }
    state.set(u);
    volumes.set(volume);
    x.set(exact);
    spatial.assemble(state.data());
    spatial.apply(x.data(),ax.data());
    Vector jx = ax.read();
    if (mixed_linear) {
        // Независимый эталон для хранения блоков: из матрицы в double извлекаем
        // по одному столбцу и округляем только коэффициенты соседних узлов.
        std::fill(jx.begin(),jx.end(),0.);
        Vector unit(size,0.);
        for (MeshIndex col = 0; col < size; ++col) {
            unit[col]=1.;
            x.set(unit);
            spatial.apply(x.data(),ax.data());
            const auto column=ax.read();
            for (MeshIndex row=0; row<size; ++row) {
                const double a=row/width==col/width ? column[row] : static_cast<LinearCoefficient>(column[row]);
                jx[row]+=a*exact[col];
            }
            unit[col]=0.;
        }
    }
    LinearTolerance tol;
    tol.relative = 1.e-11;
    for (double dt : {.005,.017}) {
        system.assemble(state.data(),nullptr,volumes.data(),dt);
        x.set(exact);
        system.apply(x.data(),ax.data());
        const Vector applied = ax.read();
        Vector rhs(size);
        double rhs_norm = 0.;
        for (MeshIndex i = 0; i < size; ++i) {
            rhs[i] = jx[i] + volume[i/width]/dt * exact[i];
            require(std::abs(applied[i]-rhs[i]) <= 5.e-14*(1.+std::abs(rhs[i])), "incorrect BDF1 time diagonal");
            rhs_norm = std::max(rhs_norm,std::abs(rhs[i]));
        }
        for (double scale : {1.,1.e-165}) {
            Vector input = rhs;
            for (double& value : input) value *= scale;
            b.set(input);
            const auto report = system.solve(b.data(),x.data(),tol);
            require(report.status == LinearStatus::Converged && report.iterations > 0, "BiCGSTAB failed to converge");
            system.apply(x.data(),ax.data());
            const auto& computed = ax.read();
            const auto& solution = x.read();
            double real_norm = 0.;
            for (MeshIndex i = 0; i < size; ++i) {
                real_norm = std::max(real_norm,std::abs(input[i]-computed[i]));
                require(std::abs(solution[i]/scale-exact[i]) < 1.e-8, "BiCGSTAB returned incorrect solution");
            }
            require(real_norm <= tol.relative*(scale*rhs_norm), "true linear residual exceeds tolerance");
            require(std::abs(report.residual_norm-real_norm) <= 1.e-14*(scale*rhs_norm), "incorrect reported residual");
            if (d == 2 && dt == .017 && scale == 1.)
                std::cout << "2D BDF1 BiCGSTAB: " << report.iterations
                          << " iterations, true relative residual " << real_norm/rhs_norm << '\n';
        }
        // Проверяем выход по лимиту и повторное решение с теми же рабочими векторами.
        b.set(rhs);
        LinearTolerance limited = tol;
        limited.max_iterations = 1;
        const auto unfinished = system.solve(b.data(),x.data(),limited);
        require(unfinished.status == LinearStatus::IterationLimit && unfinished.iterations == 1,
                "unexpected first-iteration convergence or breakdown");
        system.apply(x.data(),ax.data());
        const auto& image = ax.read();
        double remaining = 0.;
        for (MeshIndex i = 0; i < size; ++i)
            remaining = std::max(remaining, std::abs(rhs[i]-image[i]));
        require(std::abs(unfinished.residual_norm-remaining) <= 1.e-14*rhs_norm, "limit returned wrong residual");
        b.set(Vector(size,0.));
        const auto zero = system.solve(b.data(),x.data(),tol);
        require(zero.status == LinearStatus::Converged && zero.iterations == 0 && zero.residual_norm == 0. &&
                x.read() == Vector(size,0.), "wrong solution for zero RHS");
        // Ненулевая правая часть ниже абсолютного допуска допускает нулевую
        // поправку. В отчёте должна сохраниться её ненулевая фактическая невязка.
        b.set(rhs);
        LinearTolerance absolute = tol;
        absolute.absolute = 2.*rhs_norm;
        const auto small = system.solve(b.data(),x.data(),absolute);
        require(small.status == LinearStatus::Converged && small.iterations == 0 &&
                small.residual_norm == rhs_norm, "absolute tolerance was ignored");
    }
    // При малом dt матрица почти диагональна по узлам. Уже первая поправка
    // блочного Якоби даёт сходимость без стабилизирующей поправки omega.
    system.assemble(state.data(),nullptr,volumes.data(),1.e-18);
    b.set(exact);
    LinearTolerance one = tol;
    one.max_iterations = 1;
    const auto early = system.solve(b.data(),x.data(),one);
    require(early.status == LinearStatus::Converged && early.iterations == 1, "missed first BiCGSTAB correction");
    // Неудачная сборка не должна оставлять доступным старое разложение.
    volume.back() = -1.;
    volumes.set(volume);
    bool rejected = false;
    try { system.assemble(state.data(),nullptr,volumes.data(),.005); }
    catch (const std::domain_error&) { rejected = true; }
    require(rejected,"BDF1 accepted negative control volume");
    rejected = false;
    try { system.solve(b.data(),x.data()); } catch (const std::logic_error&) { rejected = true; }
    require(rejected,"BDF1 reused failed assembly");
    volumes.set(Vector(cv.volumes,cv.volumes+cv.node_count));
    rejected = false;
    try { system.assemble(state.data(),nullptr,volumes.data(),0.); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected,"BDF1 accepted zero time step");
    system.assemble(state.data(),nullptr,volumes.data(),.005);
    Vector invalid(size,0.);
    invalid.back() = std::numeric_limits<double>::quiet_NaN();
    b.set(invalid);
    rejected = false;
    try { system.solve(b.data(),x.data()); } catch (const std::domain_error&) { rejected = true; }
    require(rejected,"BiCGSTAB accepted nonfinite RHS");
}
}

void test_bdf1_linear_system(int device) {
    for (unsigned dim = 1; dim <= 3; ++dim) check_mesh(dim,device);
    std::cout << "BDF1 + LF, block Jacobi BiCGSTAB passed on "
              << (device < 0 ? "CPU" : "GPU " + std::to_string(device)) << '\n';
}
