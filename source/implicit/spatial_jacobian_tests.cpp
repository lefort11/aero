#include "../../tests/test_mesh.h"
#include "spatial_jacobian.h"
#include "../residual/residual.h"
#include "../residual/residual_detail.h"
#include "block_layout.h"
#include "../../tests/test_field.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
using namespace aero;
using aero_test::require;
using aero_test::TestField;
using Vector = std::vector<double>;

// Два разных ребра между узлами должны занимать один соседний блок в каждой
// строке. Сверяем число байтов для независимого BJ и общей с CSGS матрицы.
void check_matrix_budget() {
    using namespace aero::implicit;
    using namespace aero::residual_detail;
    Surface faces[2]{};
    for (auto& f : faces) f.end=1;
    const Index offsets[]={0,2,4};
    const OrientedFace incidence[]={OrientedFace::outward(0),OrientedFace::outward(1),
                                    OrientedFace::inward(0),OrientedFace::inward(1)};
    Geometry g{};
    g.dim=1;
    g.nodes=g.edges=g.faces=2;
    g.offsets=offsets;
    g.incidence=incidence;
    g.surfaces=faces;
    Partition part;
    for (bool csgs : {false,true}) {
        part.shared_csgs=csgs;
        const auto plan=gpu_matrix_plan(g,g.nodes,part);
        // CSR: 3 смещения, 4 столбца и 4 привязки рёбер. Два плотных блока
        // диагонали хранятся в double, два соседних — в точности сборки.
        MeshIndex expected=3*8+4*4+4*4+2*9*8+2*9*sizeof(LinearCoefficient);
        if (csgs || mixed_linear) expected+=4*4; // Адрес каждого блока.
        if (csgs) expected+=2*4+2*16+2*4+6*8;  // Порядок, два прохода, соседи и прямое решение.
        require(gpu_matrix_bytes(*plan)==expected,"wrong exact GPU matrix budget");
    }
}

// Гладкое физическое поле с различающимися скоростями LF в соседних узлах.
Vector states(unsigned d, MeshIndex nodes, double speed) {
    Vector u(nodes*(d+2));
    for (MeshIndex i = 0; i < nodes; ++i) {
        const double phase = static_cast<double>(i);
        double* node = u.data()+i*(d+2);
        node[0] = 1. + .03*std::sin(.2*phase);
        node[d+1] = (1. + .025*std::cos(.3*phase))/.4;
        for (unsigned k = 0; k < d; ++k) {
            const double v = speed/(k+1) + .02*std::sin(phase+k);
            node[k+1] = node[0]*v;
            node[d+1] += .5*node[k+1]*v;
        }
    }
    return u;
}

// Сравниваем J*v с центральной разностью ДВУХ полных сборок SpatialResidual.
// Начальные данные при возмущении текущего поля остаются фиксированными.
void check_product(SpatialResidual& op, unsigned d, MeshIndex nodes, double speed, int device,
                   bool periodic = false, bool linear = false) {
    const Vector u = states(d,nodes,speed);
    Vector initial = u, direction(u.size()), plus = u, minus = u;
    const double h = 2.e-6;
    for (size_t k = 0; k < u.size(); ++k) {
        initial[k] *= 1.001;
        direction[k] = .4*std::sin(.8*static_cast<double>(k)+.3);
        plus[k] += h*direction[k];
        minus[k] -= h*direction[k];
    }
    TestField state(device,u.size()), u0(device,u.size()), vector(device,u.size()), result(device,u.size());
    implicit::SpatialJacobian matrix(op);
    u0.set(initial);
    auto residual = [&](const Vector& field) -> Vector {
        state.set(field);
        op.assemble_rhs(state.data(), u0.data(), result.data());
        return result.read();
    };
    const Vector rp = residual(plus), rm = residual(minus);
    state.set(u);
    matrix.assemble(state.data(), u0.data());
    vector.set(direction);
    matrix.apply(vector.data(),result.data());
    const Vector product = result.read();
    for (size_t k = 0; k < u.size(); ++k) {
        const double expected = (rp[k]-rm[k])/(2*h);
        require(std::abs(product[k]-expected) <= 2.e-8*(1.+std::abs(expected)),
                "spatial J*v mismatch at " + std::to_string(k) + ": " +
                std::to_string(product[k]) + " vs " + std::to_string(expected));
    }
    if (periodic) {
        // Единичный вектор выделяет столбец матрицы. Сумма по всем строкам
        // узлов должна исчезать отдельно для каждого уравнения Эйлера.
        for (size_t col = 0; col < u.size(); ++col) {
            std::fill(direction.begin(),direction.end(),0.);
            direction[col] = 1.;
            vector.set(direction);
            matrix.apply(vector.data(),result.data());
            const auto& column = result.read();
            for (unsigned r = 0; r < d+2; ++r) {
                double sum = 0., scale = 0.;
                for (size_t k = r; k < column.size(); k += d+2) {
                    sum += column[k];
                    scale += std::abs(column[k]);
                }
                require(std::abs(sum) <= 2.e-15*(1.+scale), "nonconservative spatial Jacobian column");
            }
        }
    }
    // Повторная сборка должна заменить все блоки. После ошибки нельзя
    // использовать частично обновлённую матрицу, а следующий расчёт должен восстановиться.
    if (linear) return; // Знак пульсации свободен, а матрица не зависит от её величины.
    Vector invalid = u;
    invalid[0] = -1.;
    state.set(invalid);
    bool rejected = false;
    try { matrix.assemble(state.data(),u0.data()); } catch (const std::domain_error&) { rejected = true; }
    require(rejected, "matrix accepted negative density");
    rejected = false;
    try { matrix.apply(vector.data(),result.data()); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "matrix reused a failed assembly");
    state.set(u);
    matrix.assemble(state.data(),u0.data());
    if (!periodic) {
        matrix.apply(vector.data(),result.data());
        require(result.read() == product, "reassembly accumulated previous coefficients");
    }
}
}

void test_spatial_jacobian(int device) {
    using namespace aero;
    if (device<0) check_matrix_budget();
    ReconstructionOptions reconstruction;
    reconstruction.scheme = ReconstructionScheme::FirstOrder;
    auto options = residual_options("characteristic","lf",1.4);
    options.upw_coeff = .65;
    for (unsigned d = 1; d <= 3; ++d) {
        const Mesh mesh = aero_test::test_mesh(d);
        MedianDual dual(mesh);
        auto run = [&](const FreeBoundaryRules* rules, double speed) {
            BoundaryPatches patches;
            const BoundaryConfig config = rules ? patches.add_free(*rules) : BoundaryConfig{};
            patches.by_patch.assign(mesh.data().patch_names.size(), config);
            SpatialResidual op(mesh.view(),dual.view(),prepare_reconstruction(mesh,reconstruction),patches,options,device);
            check_product(op,d,mesh.view().node_count,speed,device);
            // На свободных границах режим определяется фоновым потоком. У стенки
            // берём покоящийся фон, так что проверка охватывает и наклонные стенки в 2D/3D.
            auto linear_options = options;
            linear_options.background.rho = 2.;
            linear_options.background.p = 2./options.gamma;
            linear_options.background.sound = 1.;
            linear_options.background.velocity[0] = rules ? speed : 0.;
            SpatialResidual linear_op(mesh.view(),dual.view(),prepare_reconstruction(mesh,reconstruction),
                                      patches,linear_options,device);
            check_product(linear_op,d,mesh.view().node_count,speed,device,false,true);
        };
        run(nullptr,.35);
        FreeBoundaryRules rules{};
        run(&rules,.35); // На дозвуковом входе и выходе задано текущее состояние.
        if (d != 1) continue;
        for (auto mode : {BoundaryMode::Pressure,BoundaryMode::Density,BoundaryMode::NormalVelocity}) {
            for (auto src : {BoundarySource::Initial,BoundarySource::Constant}) {
                rules.inflow = {mode,src,mode == BoundaryMode::NormalVelocity ? -.35 : 1.};
                rules.outflow = {mode,src,mode == BoundaryMode::NormalVelocity ? .35 : 1.};
                run(&rules,.35);
            }
        }
        rules.superin.use_initial = true;
        run(&rules,2.5); // Фиксированный сверхзвуковой вход и свободный сверхзвуковой выход.
        rules.superin.use_initial = false;
        rules.superin.mask = 1;
        rules.superin.primitive[0] = 1.;
        run(&rules,2.5); // Часть компонент входного состояния по-прежнему зависит от U.
        PeriodicDomain ring(mesh, {{"xmin","xmax",{1.,0.,0.}}});
        SpatialResidual op(d,ring.view(),ring.prepare(reconstruction),{},options,device);
        check_product(op,d,ring.view().node_count,.35,device,true);
    }
    std::cout << "Conservative spatial LF Jacobian passed on "
              << (device < 0 ? "CPU" : "GPU "+std::to_string(device)) << '\n';
}
