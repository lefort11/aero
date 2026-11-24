#include "coarse_step.h"
#include "scalar_smoother.h"
#include "../implicit/bdf1_linear_system.h"
#include "../../tests/test_field.h"
#include <cmath>

namespace {
using namespace aero;
using aero_test::require;

// Две собственные моды с разными скоростями затухания. Полиномиальная
// итерация должна уменьшить настоящий дефект обеих при неизменной матрице.
void check_scalar_smoother() {
    multigrid::ScalarSmoother polynomial;
    const double eigenvalues[] = {.06,1.8};
    double error[] = {1.,.001}, last_change[2] = {};
    const double initial_norm = .06;
    double norm = initial_norm;
    // Сначала выполняем четыре обычные точечные итерации, как при решении SA.
    // По измеренному уменьшению дефекта сглаживатель оценивает нижнюю границу спектра.
    for (unsigned iteration = 0; iteration < 4; ++iteration) {
        norm = 0.;
        for (unsigned k = 0; k < 2; ++k) {
            error[k] -= eigenvalues[k]*error[k];
            norm = std::max(norm,std::abs(eigenvalues[k]*error[k]));
        }
    }
    for (unsigned iteration = 0; iteration < 65 && norm > 1.e-6; ++iteration) {
        polynomial.next(initial_norm,norm,4);
        norm = 0.;
        for (unsigned k = 0; k < 2; ++k) {
            const double defect = eigenvalues[k]*error[k];
            last_change[k] = -polynomial.current*defect + polynomial.previous*last_change[k];
            error[k] += last_change[k];
            norm = std::max(norm,std::abs(eigenvalues[k]*error[k]));
        }
    }
    require(norm <= 1.e-6,"slow and fast modes of the scalar smoother");
    require(!polynomial.accepts(HUGE_VAL),"nonfinite scalar relaxation must be rejected");
    polynomial.restart();
    polynomial.next(.06,.05,4);
    require(polynomial.previous == 0.,"restart used a rejected scalar correction");
}

// Сравниваем PᵀJP с ручным умножением маленькой несимметричной матрицы.
// Псевдошаги и объёмы разные: оставшаяся в результате масса выявляет ошибку в LTS.
void check_transfer() {
    const MeshIndex starts[] = {0,2,4}, members[] = {0,1,2,3}, owned[] = {2,0,3,1};
    const unsigned char wall[] = {1,0,0,0};
    const double volumes[] = {3.,5.,7.,11.};
    double steps[] = {.5,.25,1.,2.};
    double coarse[] = {2.,2.,2.,3.,3.,3.}, fine[12], image[12];
    multigrid::Transfer transfer{};
    transfer.offsets = starts; transfer.members = members; transfer.owned = owned;
    transfer.width = 3; transfer.wall = wall; transfer.volumes = volumes;
    transfer.steps.values = steps; transfer.dt = .5;
    transfer.coarse = coarse; transfer.fine = fine; transfer.product = image;
    transfer.operation = multigrid::Transfer::Prolong;
    for (unsigned i = 0; i < 6; ++i) transfer(i);
    require(fine[4] == 0., "coarse prolongation changed fixed momentum");
    // J имеет диагональ 4 и смещённую на один узел периодическую диагональ -1.
    for (unsigned row = 0; row < 4; ++row) {
        const auto node = owned[row];
        const double mass = volumes[node]/(.5*steps[node]);
        for (unsigned k = 0; k < 3; ++k)
            image[3*row+k] = (4.+mass)*fine[3*row+k] - fine[3*((row+1)%4)+k];
    }
    transfer.operation = multigrid::Transfer::RestrictProduct;
    for (unsigned i = 0; i < 6; ++i) transfer(i);
    const double expected[] = {11.,8.,11.,19.,19.,19.};
    for (unsigned i = 0; i < 6; ++i) require(coarse[i] == expected[i],"Galerkin coarse operator");
    // Прилипание исключает массу закреплённого узла только из компонент импульса.
    double inverse_mass[4]{}, smoother[12]{};
    transfer.inverse_mass=inverse_mass; transfer.smoother=smoother;
    transfer.operation=multigrid::Transfer::RestrictResidual;
    for (unsigned i=0;i<6;++i) transfer(i);
    const double masses[]={1./26.,1./14.,1./51.,1./51.};
    for (unsigned i=0;i<4;++i) require(inverse_mass[i]==masses[i],"scalar and wall momentum aggregate masses");
    const double ones[]={1.,1.,1.,1.,1.,1.};
    multigrid::VectorUpdate scale{multigrid::VectorUpdate::ScaleResidual,coarse,ones,inverse_mass,0.,3};
    for (unsigned i=0;i<6;++i) scale(i);
    require(coarse[0]==masses[0] && coarse[1]==masses[1] && coarse[2]==masses[0] &&
            coarse[3]==masses[2] && coarse[4]==masses[3] && coarse[5]==masses[2],"aggregate mass component mapping");
}
}

// Проверяет ограничение, продление и принятие грубосеточной коррекции.
void test_steady_multigrid(int device) {
    using namespace aero;
    using namespace residual_detail;
    using aero_test::require;
    check_transfer();
    check_scalar_smoother();
    // Четыре периодических объёма, два агрегата. При постоянном газе матрица LF
    // имеет ненулевую диффузию неоднородной плотности даже при нулевой скорости.
    std::vector<Surface> edges;
    for (MeshIndex i = 0; i < 4; ++i) edges.push_back({i,(i+1)%4,0,{1.,0.,0.},1.,0});
    std::vector<Index> offsets;
    std::vector<OrientedFace> incidence;
    build_incidence(4,4,edges,offsets,incidence);
    Geometry host{};
    host.dim = 1; host.nodes = host.edges = host.faces = 4;
    host.surfaces = edges.data(); host.offsets = offsets.data(); host.incidence = incidence.data();
    Geometry active = host;
#ifdef AERO_WITH_KOKKOS_GPU
    gpu::DeviceArrays storage;
    if (device >= 0) {
        storage.device = device;
        active.surfaces = storage.upload(edges); active.offsets = storage.upload(offsets);
        active.incidence = storage.upload(incidence);
    }
#endif
    const std::vector<double> uniform{1.,0.,2.5, 1.,0.,2.5, 1.,0.,2.5, 1.,0.,2.5};
    const std::vector<double> forcing{.01,0.,0., .01,0.,0., -.01,0.,0., -.01,0.,0.};
    aero_test::TestField state(device,12), rhs(device,12), product(device,12), volumes(device,4);
    state.set(uniform); volumes.set({1.,1.,1.,1.});
    implicit::Partition partition;
    aero_test::TestField scratch(device,72);
    partition.coarse_graph=std::make_shared<multigrid::Aggregates>(host,4,nullptr);
    if (device>=0) {
        partition.krylov_scratch.borrow(scratch.data(),72*sizeof(double),12);
        partition.coarse_product_persistent=true;
        const auto separate=multigrid::CoarseStep::storage_bytes(*partition.coarse_graph,4,3,{});
        const auto shared=multigrid::CoarseStep::storage_bytes(*partition.coarse_graph,4,3,partition.krylov_scratch);
        require(shared<separate,"multigrid reuses Krylov storage");
        const auto shared_product=multigrid::CoarseStep::storage_bytes(*partition.coarse_graph,4,3,partition.krylov_scratch,true);
        require(shared-shared_product==12*sizeof(double),"multigrid reuses caller product for persistent smoother");
    }
    implicit::BdfLinearSystem matrix(host,active,device,partition);
    matrix.assemble(state.data(),nullptr,volumes.data(),.1);
    multigrid::CoarseStep coarse(host,active,4,device,partition);
    unsigned trials = 0, restores = 0;
    // Проверяет одну попытку грубосеточной коррекции и критерий её принятия.
    const auto attempt = [&](double trial_norm) {
        return coarse.correct(matrix,state.data(),volumes.data(),.1,rhs.data(),product.data(),1.,
                              [&] {
                                  ++trials;
                                  // Пересчёт потоков уничтожает заимствованные направления.
                                  // Они уже не нужны для повторной пробы с меньшим множителем.
                                  if (device>=0) scratch.set(std::vector<double>(72,12345.));
                                  return trial_norm;
                              },[&] { ++restores; });
    };
    rhs.set(std::vector<double>(12,0.));
    require(attempt(0.) == 0. && trials == 0 && state.read() == uniform,"zero coarse forcing");
    rhs.set(forcing);
    require(attempt(2.) == 0. && trials > 0 && restores == 1 && state.read() == uniform,
            "rejected coarse correction must restore accepted gas exactly");
    rhs.set(forcing);
    require(attempt(.5) == 1. && state.read()[0] > 1. && state.read()[6] < 1.,
            "accepted coarse correction has the wrong sign");
    // Гладкое направление строится из двух принятых полей. Проверяем его
    // знак и то, что решатель не использовал историю LTS как рабочий буфер.
    aero_test::TestField start(device,12);
    auto previous = uniform;
    for (unsigned i = 0; i < previous.size(); ++i) previous[i] -= forcing[i];
    start.set(previous); state.set(uniform);
    rhs.set(forcing);
    const double scale = coarse.correct(matrix,state.data(),volumes.data(),.1,rhs.data(),product.data(),
                                        1.,[] { return .5; },[] {},start.data());
    require(scale > 0. && coarse.used_history() && start.read() == previous,"smooth coarse direction and LTS history");
    const auto result = state.read();
    for (unsigned i = 0; i < result.size(); ++i)
        require(std::abs(result[i]-(uniform[i]+scale*forcing[i])) < 2.e-15,"smooth coarse prolongation");
}
