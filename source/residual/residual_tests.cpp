#include "residual_detail.h"
#include "../../qa/test_checks.h"
#include "../eulersolver/roe.h"
#include "../eulersolver/lax_friedrichs.h"
#ifndef AERO_RESIDUAL_NO_TIME_TEST
#include "../timestep/timestep.h"
#include "../explicit/explicit.h"
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef AERO_WITH_CUDA
#include "../../qa/cuda_test_buffer.h"
#include "../cuda/runtime.h"
#endif

using namespace aero;
namespace {
using Values = std::vector<double>;
using aero_test::require;
void near(double a, double b) {
    if (!(std::abs(a - b) <= 3e-11 * (1 + std::abs(b))))
        throw std::runtime_error("numerical mismatch: " + std::to_string(a) + " versus " + std::to_string(b));
}
struct Result {
    Values r, edge, bc;
    Result(size_t n, size_t e, size_t b) : r(n), edge(e), bc(b) {}
};
#ifdef AERO_WITH_CUDA
using aero::cuda_detail::checked;
#endif
void evaluate(int device, SpatialResidual& op, const Values& u, Result& output) {
    if (device < 0) {
        op.assemble(u.data(), u.data(), output.r.data(), output.edge.data(), output.bc.data());
        return;
    }
#ifdef AERO_WITH_CUDA
    checked(cudaSetDevice(device));
    aero_test::DeviceBuffer<double> state(u.size()), r(u.size()), edge(output.edge.size()),
        bc(output.bc.size());
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        checked(
            cudaMemcpyAsync(state.data, u.data(), u.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        op.assemble(state.data, state.data, r.data, edge.data, bc.data, stream);
        r.download(output.r.data(), output.r.size());
        edge.download(output.edge.data(), output.edge.size());
        bc.download(output.bc.data(), output.bc.size());
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    checked(cudaStreamDestroy(stream));
#endif
}
void compare(const Values& a, const Values& b) {
    require(a.size() == b.size(), "array extent");
    for (size_t i = 0; i < a.size(); ++i)
        near(a[i], b[i]);
}
void compare(const Result& a, const Result& b) {
    compare(a.r, b.r);
    compare(a.edge, b.edge);
    compare(a.bc, b.bc);
}
BoundaryPatches free_patches(const Mesh& mesh) {
    BoundaryPatches patches;
    const auto open = patches.add_free(FreeBoundaryRules{});
    patches.by_patch.assign(mesh.data().patch_names.size(), open);
    return patches;
}
constexpr double gamma_gas = 1.57;
// Проверяем путь уже вычисленного потока в обе строки невязки, независимо от решателя Римана.
void signed_surface_balance() {
    using namespace residual_detail;
    std::vector<Surface> surfaces(3);
    surfaces[0].begin = 0;
    surfaces[0].end = 1; // Ребро между двумя контрольными объёмами.
    surfaces[1].begin = 1;
    surfaces[1].end = 1;   // Периодическая петля возвращается в тот же узел.
    surfaces[2].begin = 0; // Одна наружная площадка первого узла.
    std::vector<Index> offsets;
    std::vector<OrientedFace> entries;
    build_incidence(2, 2, surfaces, offsets, entries);
    require(offsets == std::vector<Index>{0, 2, 5}, "edge, loop and boundary incidence counts");
    Geometry grid{};
    grid.dim = 1;
    grid.offsets = offsets.data();
    grid.incidence = entries.data();
    EulerFluxResult flux[3] = {};
    for (unsigned c = 0; c < 3; ++c) {
        flux[0].flux[c] = 3 * (c + 1);
        flux[1].flux[c] = 7 * (c + 1);
        flux[2].flux[c] = -2.0 * (c + 1);
    }
    Values residual(6);
    for (Index node = 0; node < 2; ++node)
        require(assemble_node_residual(grid, node, flux, grid.node_row(residual.data(), node)),
                "node balance");
    // На узле 0: +3−2=1. На узле 1: −3+7−7=−3. Сумма по сетке равна внешнему потоку −2.
    compare(residual, {1, 2, 3, -3, -6, -9});
}
void trace_limiter_test() {
    // При γ = 2 в узле ρ = 2, u = (1,0), p = 4. Кандидат имеет ρ = 4, u = (3,2), p = 12.
    // Давление ограничивает коэффициент до 1/4, тогда как плотность допускает 1/2.
    const double node[] = {2, 2, 0, 5}, normal[] = {1, 0};
    auto options = residual_options("conservative", "roe", 2, .5, 1.5);
    double state[] = {4, 12, 8, 38};
    double flux[] = {6, 10, 8, 17};
    require(residual_detail::limit_trace(2, options, node, state, normal, flux), "limited trace");
    compare(Values(state, state + 4), {2.5, 3.75, 1.25, 9.125});
    compare(Values(flux, flux + 4), {3, 7, 2, 11});

    double low[] = {.5, .5, 0, 4.25};
    require(residual_detail::limit_trace(2, options, node, low, normal, nullptr), "low density trace");
    compare(Values(low, low + 4), {1, 1, 0, 4.5});
    for (const auto& nonphysical : {Values{2, 4, 0, 1}, Values{-1, 1, 0, 5}}) {
        Values trace = nonphysical;
        require(residual_detail::limit_trace(2, options, node, trace.data(), normal, flux),
                "zero-order fallback");
        compare(trace, {2, 2, 0, 5});
        compare(Values(flux, flux + 4), {2, 6, 0, 9});
    }
    double nan_trace[] = {2, 2, NAN, 5};
    require(!residual_detail::limit_trace(2, options, node, nan_trace, normal, nullptr),
            "nonfinite trace accepted");
}
void basis_test() {
    const double a[] = {1, .21, .32, .43, 4}, b[] = {1.5, .2, .3, .4, 5}, dir[] = {-.4, .8, .32};
    for (unsigned d = 1; d <= 3; ++d) {
        double left[5] = {1, .21, .32, .43, 4}, right[5] = {1.5, .2, .3, .4, 5};
        left[d + 1] = a[4];
        right[d + 1] = b[4];
        EulerRoeBasis basis;
        require(euler_roe_basis(d, left, right, dir, gamma_gas, &basis) == EulerStatus::Success, "basis");
        for (unsigned c = 0; c < d + 2; ++c) {
            double u[5], w[5];
            for (unsigned k = 0; k < d + 2; ++k)
                u[k] = basis.right[5 * k + c];
            require(conservative_to_characteristic(basis, dir, gamma_gas, u, w) == EulerStatus::Success,
                    "projection");
            for (unsigned k = 0; k < d + 2; ++k)
                near(w[k], c == k ? 1 : 0);
        }
    }
}

// Независимый эталон для малых 1D-тестов разносит потоки по узлам. Для каждого ребра
// преобразуются полные поля на всех узлах его шаблона.
Result reference(const Mesh& mesh, CvView cv, const ReconstructionStencil& stencil, ResidualOptions options,
                 const Values& u, const BoundaryPatches& bc) {
    const auto m = mesh.view();
    Result out(u.size(), cv.edge_count, cv.boundary_count);
    std::vector<bool> open_node(m.node_count, false);
    for (MeshIndex k = 0; k < cv.boundary_count; ++k) {
        const auto& b = cv.boundary[k];
        if (bc.by_patch[b.patch].type() != BoundaryType::Wall)
            open_node[b.node] = true;
    }
    for (MeshIndex e = 0; e < m.edge_count; ++e) {
        const auto i = m.edge_begin[e], j = m.edge_end[e];
        // Базис хранит размерность во время выполнения. Передаём нормаль в том же трёхкомпонентном буфере, что и солвер.
        const double normal[3] = {cv.edge_areas[e].x, 0, 0};
        EulerRoeBasis basis;
        require(euler_roe_basis(1, &u[i * 3], &u[j * 3], normal, options.gamma, &basis) == EulerStatus::Success,
                "reference basis");
        Values q = u;
        for (MeshIndex k = 0; k < m.node_count; ++k) {
            double speed;
            if (options.variables == ResidualVariables::Characteristic)
                require(conservative_to_characteristic(basis, normal, options.gamma, &u[3 * k], &q[3 * k]) ==
                            EulerStatus::Success,
                        "project");
            if (options.variables == ResidualVariables::Flux)
                require(euler_flux(1, &u[3 * k], normal, options.gamma, &q[3 * k], &speed) ==
                            EulerStatus::Success,
                        "project flux");
        }
        auto pair = [&](const Values& field, double* left, double* right) {
            if (open_node[i] || open_node[j]) {
                std::copy_n(field.data() + 3 * i, 3, left);
                std::copy_n(field.data() + 3 * j, 3, right);
                return;
            }
            const ReconstructionFieldView read = {field.data(), 3, 1};
            double dl[3], dr[3];
            reconstruct_increment2(stencil.view(), e, ReconstructionSide::Left, &field[3*i], 3, read, dl);
            reconstruct_increment2(stencil.view(), e, ReconstructionSide::Right, &field[3*j], 3, read, dr);
            for (unsigned k = 0; k < 3; ++k) {
                const double a = field[3*i+k], b = field[3*j+k];
                // Независимый эталон смешивает именно приращения, до построения следов.
                const double w = options.upw_coeff;
                if (stencil.data().scheme == ReconstructionScheme::Ebr3 || stencil.data().scheme == ReconstructionScheme::Ebr5) {
                    const double lr = 0.5 * (dl[k] - dr[k]);
                    dl[k] = w * dl[k] + (1 - w) * ((b - a) + lr);
                    dr[k] = w * dr[k] + (1 - w) * ((b - a) - lr);
                }
                left[k] = a + .5 * dl[k];
                right[k] = b - .5 * dr[k];
            }
        };
        double l[3], r[3], ul[3], ur[3];
        pair(q, l, r);
        EulerFluxResult f{};
        const bool roe = options.solver == ResidualSolver::Roe;
        if (options.variables == ResidualVariables::Flux) {
            pair(u, ul, ur);
            require((roe ? roe_fluxes(1, l, r, ul, ur, normal, options.gamma, &f)
                         : lax_friedrichs_fluxes(1, l, r, ul, ur, normal, options.gamma, &f)) ==
                        EulerStatus::Success,
                    "reference fluxes");
        } else {
            if (options.variables == ResidualVariables::Characteristic) {
                characteristic_to_conservative(basis, l, l);
                characteristic_to_conservative(basis, r, r);
            }
            require((roe ? roe_conservative(1, l, r, normal, options.gamma, &f)
                         : lax_friedrichs_conservative(1, l, r, normal, options.gamma, &f)) ==
                        EulerStatus::Success,
                    "reference states");
            double speed;
            require(euler_flux(1, l, normal, options.gamma, ul, &speed) == EulerStatus::Success &&
                    euler_flux(1, r, normal, options.gamma, ur, &speed) == EulerStatus::Success, "central flux");
        }
        for (unsigned k = 0; k < 3; ++k) {
            // Публичный решатель вернул полный LF/Roe. Масштабируем только отличие от среднего потока.
            const double center = options.variables == ResidualVariables::Flux ? .5*(l[k]+r[k]) : .5*(ul[k]+ur[k]);
            f.flux[k] = center + options.upw_coeff * (f.flux[k] - center);
            out.r[3 * i + k] += std::abs(normal[0]) * f.flux[k];
            out.r[3 * j + k] -= std::abs(normal[0]) * f.flux[k];
        }
        out.edge[e] = f.max_wave_speed;
    }
    for (MeshIndex k = 0; k < cv.boundary_count; ++k) {
        const auto& piece = cv.boundary[k];
        EulerFluxResult f{};
        const double* state = &u[3 * piece.node];
        require(boundary_flux(1, bc.by_patch[piece.patch], bc.free_rules.data(), state, state, &piece.area.x,
                              options.gamma, &f) == BoundaryStatus::Success,
                "reference BC");
        for (unsigned c = 0; c < 3; ++c)
            out.r[3 * piece.node + c] += f.flux[c];
        out.bc[k] = f.max_wave_speed;
    }
    return out;
}
void edge_modes(int device) {
    const Mesh mesh = Mesh::read("examples/tecplot/demo_1d.mesh");
    const auto m = mesh.view();
    MedianDual dual(mesh);
    const auto cv = dual.view();
    auto bc = free_patches(mesh);
    bc.by_patch[0] = BoundaryConfig{};
    Values u(3 * m.node_count);
    for (MeshIndex k = 0; k < m.node_count; ++k) {
        u[k * 3] = 1 + .15 * std::sin(9 * m.x[k]);
        u[k * 3 + 1] = .15 * std::cos(18 * m.x[k]);
        u[k * 3 + 2] = 2.8 + .26 * std::sin(15 * m.x[k]);
    }
    for (auto scheme :
         {ReconstructionScheme::FirstOrder, ReconstructionScheme::Ebr3, ReconstructionScheme::Ebr5,
          ReconstructionScheme::Weno3, ReconstructionScheme::Weno5, ReconstructionScheme::Mp5, ReconstructionScheme::MPWENO5}) {
        ReconstructionOptions opts;
        opts.scheme = scheme;
        auto stencil = prepare_reconstruction(mesh, opts);
        for (const char* vars : {"conservative", "characteristic", "flux"})
            for (const char* solver : {"lf", "roe"}) {
                auto o = residual_options(vars, solver, gamma_gas);
                for (double weight : {0., .4, 1.}) {
                    o.upw_coeff = weight;
                    SpatialResidual op(m, cv, stencil, bc, o, device);
                    Result result(u.size(), cv.edge_count, cv.boundary_count);
                    evaluate(device, op, u, result);
                    compare(result, reference(mesh, cv, stencil, o, u, bc));
                    if (weight != 1. || (scheme != ReconstructionScheme::Mp5 &&
                                         scheme != ReconstructionScheme::MPWENO5))
                        continue;

                    // Проверяем, что параметр действительно доходит от невязки до полной пары
                    // следов. На непостоянном поле включение центральной части меняет невязку.
                    o.mp6_hybrid = 1.;
                    SpatialResidual hybrid(m, cv, stencil, bc, o, device);
                    Result corrected(u.size(), cv.edge_count, cv.boundary_count);
                    evaluate(device, hybrid, u, corrected);
                    require(corrected.r != result.r, "MP6 correction did not change residual");
                    if (device >= 0) {
                        SpatialResidual host(m, cv, stencil, bc, o);
                        evaluate(-1, host, u, result);
                        compare(corrected, result);
                    }
                }
            }
    }
}
// Первая компонента полностью запрещает общую поправку. Вектор импульса
// должен получить ровно свою проекцию на касательную плоскость.
void oblique_mp_limit() {
    for (unsigned d : {2u, 3u}) {
        const double normal[3] = {d == 2 ? .6 : 2./3., d == 2 ? .8 : 1./3., d == 2 ? 0. : 2./3.};
        const double momentum[] = {1., 2., -1.};
        const residual_detail::FaceMomentumAxes axes(d, normal, normal);
        double slopes[5][5] = {};
        slopes[residual_detail::CentralSlope][0] = 1.;
        for (auto& row : slopes) {
            for (unsigned j = 0; j < d; ++j)
                row[1+j] = momentum[j];
            row[d+1] = 1.;
        }
        for (bool weno : {false, true}) {
            double left[5] = {}, right[5] = {};
            const Mp5Settings settings = {!weno, weno, true, 1.};
            if (weno) residual_detail::correct_cartesian_pair<true>(axes, slopes, settings, 1., left, right);
            else residual_detail::correct_cartesian_pair<false>(axes, slopes, settings, 1., left, right);
            const double normal_part = euler_detail::dot(d, momentum, normal);
            for (const double* side : {left, right}) {
                near(side[0], 0.);
                near(side[d+1], 0.);
                for (unsigned j = 0; j < d; ++j)
                    near(side[1+j], momentum[j] - normal[j]*normal_part);
            }
        }
    }
}

// Поворачиваем сетку, чтобы ни один из плоских координатных импульсов
// нельзя было целиком принять за нормальный или касательный к ребру.
void cartesian_hybrid_limit(unsigned d, int device) {
    MeshData data = Mesh::read("examples/tecplot/demo_" + std::to_string(d) + "d.mesh").data();
    for (size_t i = 0; i < data.x.size(); ++i) {
        const double x = data.x[i], y = data.y[i];
        data.x[i] = .8*x - .6*y;
        data.y[i] = .6*x + .8*y;
    }
    const Mesh mesh(std::move(data));
    const auto m = mesh.view();
    MedianDual dual(mesh);
    const auto cv = dual.view();
    const auto bc = free_patches(mesh);
    Values u((d+2)*m.node_count);
    for (MeshIndex i = 0; i < m.node_count; ++i) {
        double* node = &u[(d+2)*i];
        node[0] = 1. + .1*std::sin(3*m.x[i] + 4*m.y[i]);
        for (unsigned j = 0; j < d; ++j)
            node[1+j] = .12*std::cos((j+1)*m.x[i] - 5*m.y[i] + 2*m.z[i]);
        node[d+1] = 3. + .2*std::sin(7*m.x[i] + m.y[i]);
    }
    for (auto scheme : {ReconstructionScheme::Mp5, ReconstructionScheme::MPWENO5}) {
        ReconstructionOptions st;
        st.scheme = scheme;
        const auto stencil = prepare_reconstruction(mesh, st);
        for (const char* variables : {"conservative", "flux"}) {
            auto options = residual_options(variables, "roe", gamma_gas);
            const auto run = [&](double h, int backend) {
                options.mp6_hybrid = h;
                SpatialResidual op(m, cv, stencil, bc, options, backend);
                Result result(u.size(), cv.edge_count, cv.boundary_count);
                evaluate(backend, op, u, result);
                return result;
            };
            const Result base = run(0., device);
            // Малый положительный h проходит весь гибридный путь. Невязка и
            // скорости распространения должны сходиться к обычной схеме.
            compare(run(1.e-14, device), base);
            const Result full = run(1., device);
            require(full.r != base.r, "No active MP6 correction on the rotated mesh");
            if (device >= 0)
                compare(full, run(1., -1));
        }
    }
}

void constant_test(unsigned d, int device) {
    const Mesh mesh = Mesh::read("examples/tecplot/demo_" + std::to_string(d) + "d.mesh");
    const auto m = mesh.view();
    MedianDual dual(mesh);
    const auto cv = dual.view();
    auto bc = free_patches(mesh);
    ReconstructionOptions opts;
    opts.scheme = ReconstructionScheme::Ebr3;
    auto stencil = prepare_reconstruction(mesh, opts);
    Values u((d + 2) * m.node_count);
    for (MeshIndex n = 0; n < m.node_count; ++n) {
        u[(d + 2) * n] = 1;
        u[(d + 2) * n + d + 1] = 3;
        for (unsigned k = 0; k < d; ++k)
            u[(d + 2) * n + 1 + k] = .1 * (k + 1);
    }
    for (const char* vars : {"conservative", "characteristic", "flux"}) {
        SpatialResidual op(m, cv, stencil, bc, residual_options(vars, "roe", gamma_gas), device);
        Result out(u.size(), cv.edge_count, cv.boundary_count);
        evaluate(device, op, u, out);
        for (double v : out.r)
            near(v, 0);
        require(*std::min_element(out.edge.begin(), out.edge.end()) > 0, "edge speeds");
        require(*std::min_element(out.bc.begin(), out.bc.end()) > 0, "boundary speeds");
        Values varying = u;
        for (size_t i = 0; i < m.node_count; ++i) {
            varying[(d + 2) * i] += 0.15 * std::cos(4 * m.x[i] + 3 * m.y[i] + m.z[i]);
            varying[(d + 2) * i + 1] += 0.2 * std::sin(7 * m.y[i] - 3 * m.x[i]);
        }
        evaluate(device, op, varying, out);
        if (device >= 0) {
            SpatialResidual host(m, cv, stencil, bc, residual_options(vars, "roe", gamma_gas));
            Result expected = out;
            evaluate(-1, host, varying, expected);
            compare(out, expected);
        }
#ifdef _OPENMP
        if (device < 0) {
            const int threads = omp_get_max_threads();
            omp_set_num_threads(1);
            Result serial = out;
            evaluate(-1, op, varying, serial);
            omp_set_num_threads(threads);
            require(out.r == serial.r && out.edge == serial.edge && out.bc == serial.bc,
                    "OpenMP summation order");
        }
#endif
        Values invalid(varying.size(), 0);
        bool rejected = false;
        try {
            evaluate(device, op, invalid, out);
        } catch (const std::domain_error&) {
            rejected = true;
        }
        require(rejected, "invalid state accepted");
        evaluate(device, op, u, out);
        for (double v : out.r)
            near(v, 0);
    }
}
void inlet_outlet_edges(int device) {
    const Mesh mesh = Mesh::read("examples/tecplot/demo_2d.mesh");
    const auto m = mesh.view();
    MedianDual dual(mesh);
    const auto cv = dual.view();
    auto bc = free_patches(mesh);
    ReconstructionOptions opts;
    opts.scheme = ReconstructionScheme::Weno5;
    auto stencil = prepare_reconstruction(mesh, opts);
    for (size_t patch = 0; patch < bc.by_patch.size(); ++patch) {
        const auto& name = mesh.data().patch_names[patch];
        // Эта граница принимает сверхзвуковой поток из начального состояния.
        if (name == "xmin") {
            FreeBoundaryRules rules{};
            rules.superin.use_initial = true;
            bc.by_patch[patch] = bc.add_free(rules);
        }
    }
    Values u(4 * m.node_count);
    for (MeshIndex n = 0; n < m.node_count; ++n) {
        u[4 * n] = 1 + .1 * m.x[n];
        u[4 * n + 1] = u[4 * n] * (3 + .1 * m.y[n]);
        u[4 * n + 2] = .1 * m.x[n];
        u[4 * n + 3] = 2.5 + .3 * m.x[n] + .2 * m.y[n] + .5 * u[4 * n + 1] * u[4 * n + 1] / u[4 * n];
    }
    SpatialResidual op(m, cv, stencil, bc, residual_options("characteristic", "roe", gamma_gas), device);
    Result out(u.size(), cv.edge_count, cv.boundary_count);
    evaluate(device, op, u, out);
    std::vector<bool> boundary_node(m.node_count, false);
    for (MeshIndex i = 0; i < cv.boundary_count; ++i)
        boundary_node[cv.boundary[i].node] = true;
    size_t tangent_edges = 0, interior_to_boundary_edges = 0;
    for (MeshIndex e = 0; e < m.edge_count; ++e) {
        const auto i = m.edge_begin[e], j = m.edge_end[e];
        if (!boundary_node[i] && !boundary_node[j])
            continue;
        if (!stencil.data().weno_order[2 * e] && !stencil.data().weno_order[2 * e + 1])
            continue;
        const double area[] = {cv.edge_areas[e].x, cv.edge_areas[e].y};
        EulerFluxResult expected{};
        require(roe_conservative(2, &u[4 * i], &u[4 * j], area, gamma_gas, &expected) == EulerStatus::Success,
                "inlet/outlet reference flux");
        near(out.edge[e], expected.max_wave_speed);
        if (boundary_node[i] && boundary_node[j])
            ++tangent_edges;
        else
            ++interior_to_boundary_edges;
    }
    require(tangent_edges && interior_to_boundary_edges, "missing inlet/outlet edge coverage");
}
void ring_test(int device) {
    const Mesh mesh = Mesh::read("examples/tecplot/demo_1d.mesh");
    PeriodicDomain ring(mesh, {{"inlet", "outlet", {1, 0, 0}}});
    const auto g = ring.view();
    ReconstructionOptions opts;
    opts.scheme = ReconstructionScheme::Weno3;
    SpatialResidual op(1, g, ring.prepare(opts), {}, residual_options("characteristic", "roe", gamma_gas),
                       device);
    Values u(3 * g.node_count);
    for (size_t n = 0; n < g.node_count; ++n) {
        u[3 * n] = 1 + .1 * std::sin(6.28 * mesh.view().x[n]);
        u[3 * n + 1] = .4;
        u[3 * n + 2] = 3;
    }
    Result out(u.size(), g.edge_count, 0);
    evaluate(device, op, u, out);
    for (unsigned k = 0; k < 3; ++k) {
        double total = 0;
        for (size_t n = 0; n < g.node_count; ++n)
            total += out.r[3 * n + k];
        near(total, 0);
    }
}
#ifndef AERO_RESIDUAL_NO_TIME_TEST
void sod_step(int device) {
    Mesh mesh = Mesh::read("examples/tecplot/demo_1d.mesh");
    MedianDual dual(mesh);
    const auto cv = dual.view();
    const auto m = mesh.view();
    ReconstructionOptions opts;
    opts.scheme = ReconstructionScheme::FirstOrder;
    SpatialResidual op(m, cv, prepare_reconstruction(mesh, opts), free_patches(mesh),
                       residual_options("conservative", "lf", 1.4), device);
    Values u(3 * m.node_count);
    for (size_t n = 0; n < m.node_count; ++n) {
        u[3 * n] = m.x[n] < .5 ? 1 : .125;
        u[3 * n + 2] = m.x[n] < .5 ? 2.5 : .25;
    }
    Result out(u.size(), cv.edge_count, cv.boundary_count);
    evaluate(device, op, u, out);
    CflTimestep cfl(m, cv);
    const auto dt = cfl.compute(.4, {out.edge.data(), out.edge.size(), out.bc.data(), out.bc.size()});
    ExplicitIntegrator step(m.node_count, 3, cv.volumes, u.data(),
                            device < 0 ? ExplicitBackend::Cpu : ExplicitBackend::Cuda,
                            device < 0 ? 0 : device);
    step.step(ExplicitScheme::Euler, 0, dt.dt, [&](double, const ExplicitField& f, double* r) {
        const double actual = op.assemble_cfl(f.values, f.values, r, f.volumes, .4, f.stream);
        require(actual == dt.dt, "Sod CFL differs from reduction of the exported speeds");
    });
    const auto next = step.state();
    require(next != u, "Sod jump must evolve");
    for (size_t i = 0; i < next.size(); ++i)
        near(next[i], u[i] - dt.dt * out.r[i] / cv.volumes[i / 3]);
}
#endif
}
int main(int argc, char** argv) {
    try {
        signed_surface_balance();
        basis_test();
        trace_limiter_test();
        oblique_mp_limit();
        const int devices = ReconstructionContext::cuda_device_count();
        if (argc > 1 && std::string(argv[1]) == "--require-cuda")
            require(devices > 0, "CUDA device required");
        for (int dev = -1; dev < devices; ++dev) {
#ifndef AERO_RESIDUAL_NO_TIME_TEST
            sod_step(dev);
#endif
            edge_modes(dev);
            ring_test(dev);
            inlet_outlet_edges(dev);
            constant_test(2, dev);
            constant_test(3, dev);
            cartesian_hybrid_limit(2, dev);
            cartesian_hybrid_limit(3, dev);
            std::cout << "Residual tests passed: " << (dev < 0 ? "CPU" : "GPU " + std::to_string(dev))
                      << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
