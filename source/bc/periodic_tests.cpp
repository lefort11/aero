#include "periodic.h"
#include "../../tests/test_checks.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace aero;
#ifdef AERO_WITH_KOKKOS_GPU
void periodic_gpu(int, PeriodicView, ReconstructionStencilView, const double*, const double*);
#endif
namespace {
using Id = MeshIndex;
using aero_test::require;
void near(double a, double b) {
    require(std::abs(a - b) < 1e-11, "periodic geometry or reconstruction mismatch");
}
PeriodicPair parse(const std::string& spec) {
    PeriodicPair pair;
    std::istringstream input("bc_type periodic\n" + spec);
    BoundaryPatches patches;
    require(read_boundary(input, 3, patches, pair).type() == BoundaryType::Periodic &&
                patches.free_rules.empty(),
            "periodic parser");
    return pair;
}
void append(MeshCsr& csr, const std::vector<Id>& list) {
    csr.indices.insert(csr.indices.end(), list.begin(), list.end());
    csr.offsets.push_back(csr.indices.size());
}
Mesh line(unsigned repeats) {
    MeshData d;
    d.dimension = 1;
    d.patch_names = {"left", "right"};
    const unsigned N = 6 * repeats;
    for (unsigned i = 0; i <= N; ++i) {
        d.x.push_back(i + .125 * (i % 2));
        d.y.push_back(0);
        d.z.push_back(0);
        append(d.face_nodes, {i});
        if (i < N) {
            append(d.cell_faces, {i, i + 1});
            d.cell_face_orientation.insert(d.cell_face_orientation.end(), {-1, 1});
        }
    }
    d.boundary_faces = {0, N};
    d.boundary_patch = {0, 1};
    return Mesh(std::move(d));
}
double increment(ReconstructionStencilView st, Id e, unsigned side, const std::vector<double>& q) {
    double v = 0;
    const Id owner = st.row_owner[2 * e + side];
    const ReconstructionFieldView field = {q.data(), 1, 1};
    require(reconstruct_increment2(st, e, side ? ReconstructionSide::Right : ReconstructionSide::Left,
                                   &q[owner], 1, field, &v) == ReconstructionStatus::Success,
            "periodic row access");
    return v;
}
void line_test(bool require_gpu) {
    Mesh original = line(1), extended = line(3);
    PeriodicDomain domain(original, {parse("patch left\npair right\ntranslation 6 0 0")});
    const auto view = domain.view();
    require(view.node_count == 6 && view.edge_count == 6 && !view.boundary_count, "periodic line shape");
    // На шве складываются половины двух соседних отрезков длиной 1.125 и 0.875.
    near(view.volumes[0], 0.5 * (1.125 + 0.875));
    std::vector<double> field = {.3, -.8, .7, 1.1, .9, -.2}, long_field(19);
    for (size_t i = 0; i < long_field.size(); ++i)
        long_field[i] = field[i % 6];
    for (auto scheme :
         {ReconstructionScheme::FirstOrder, ReconstructionScheme::Ebr3, ReconstructionScheme::Ebr5,
          ReconstructionScheme::Weno3, ReconstructionScheme::Weno5}) {
        ReconstructionOptions opts;
        opts.scheme = scheme;
        auto ref = prepare_reconstruction(extended, opts);
        opts.require_complete_stencil = true;
        auto st = domain.prepare(opts);
        std::vector<double> expected(12), residual(6), ref_flux(6);
        for (Id e = 0; e < 6; ++e) {
            const auto& edge = view.edges[e];
            for (unsigned s = 0; s < 2; ++s) {
                expected[2 * e + s] = increment(ref.view(), edge.original + 6, s, long_field);
                near(increment(st.view(), e, s, field), expected[2 * e + s]);
            }
            // Эталонная невязка противопоточного переноса с единичной скоростью на склеенной сетке.
            const double flux = edge.area.x * (field[edge.begin] + .5 * increment(st.view(), e, 0, field));
            residual[edge.begin] += flux;
            residual[edge.end] -= flux;
            ref_flux[e] = long_field[6 + e] + .5 * expected[2 * e];
        }
        for (Id n = 0; n < 6; ++n)
            near(residual[n], ref_flux[n] - ref_flux[(n + 5) % 6]);
#ifdef AERO_WITH_KOKKOS_GPU
        const int count = ReconstructionContext::cuda_device_count();
        require(!require_gpu || count > 0, "periodic test requires GPU");
        for (int dev = 0; dev < count; ++dev) {
            ReconstructionContext ctx(st, ReconstructionBackend::Cuda, dev);
            periodic_gpu(dev, view, ctx.device_view(), field.data(), expected.data());
        }
#else
        (void)require_gpu;
#endif
    }
}
Mesh box(unsigned d) {
    MeshData m;
    m.dimension = d;
    m.patch_names = {"xm", "xp", "ym", "yp", "zm", "zp"};
    const std::vector<std::vector<Id>> faces =
        d == 2 ? std::vector<std::vector<Id>>{{2, 0}, {1, 3}, {0, 1}, {3, 2}}
               : std::vector<std::vector<Id>>{{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4},
                                              {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    for (unsigned i = 0; i < (1u << d); ++i) {
        m.x.push_back(i & 1);
        m.y.push_back((i >> 1) & 1);
        m.z.push_back((i >> 2) & 1);
    }
    std::vector<Id> ids;
    for (Id i = 0; i < faces.size(); ++i) {
        append(m.face_nodes, faces[i]);
        m.boundary_faces.push_back(i);
        m.boundary_patch.push_back(i);
        m.cell_face_orientation.push_back(1);
        ids.push_back(i);
    }
    append(m.cell_faces, ids);
    return Mesh(std::move(m));
}
}
void test_periodic(bool require_gpu) {
    line_test(require_gpu);
    const PeriodicPair x = parse("patch xm\npair xp\ntranslation 1 0 0");
    const PeriodicPair y = parse("patch ym\npair yp\ntranslation 0 1 0");
    const PeriodicPair z = parse("patch zm\npair zp\ntranslation 0 0 1");
    // Справа добавлена вершина. Одного совпадения концов после сдвига недостаточно для совпадения граней.
    MeshData split = box(2).data();
    split.x.push_back(1);
    split.y.push_back(.5);
    split.z.push_back(0);
    split.face_nodes.indices[3] = 4;
    append(split.face_nodes, {4, 3});
    split.cell_faces.indices.push_back(4);
    ++split.cell_faces.offsets.back();
    split.cell_face_orientation.push_back(1);
    split.boundary_faces.push_back(4);
    split.boundary_patch.push_back(1);
    Mesh nonmatching(split);
    bool rejected = false;
    try {
        PeriodicDomain invalid(nonmatching, {x});
    } catch (const MeshError&) {
        rejected = true;
    }
    require(rejected, "nonconforming subdivisions accepted");
    for (unsigned d : {2u, 3u}) {
        Mesh mesh = box(d);
        PeriodicDomain partial(mesh, {x});
        require(partial.view().boundary_count == (d == 2 ? 4u : 16u), "physical pieces were lost");
        const std::vector<PeriodicPair> pairs =
            d == 2 ? std::vector<PeriodicPair>{x, y} : std::vector<PeriodicPair>{x, y, z};
        PeriodicDomain full(mesh, pairs);
        auto v = full.view();
        require(v.node_count == 1 && !v.boundary_count && v.edge_count == d,
                "corner equivalence or duplicate seam edge");
        near(v.volumes[0], 1);
        for (Id e = 0; e < v.edge_count; ++e) {
            const CvVector area = v.edges[e].area;
            near(std::sqrt(area.x * area.x + area.y * area.y + area.z * area.z), 1);
        }
        ReconstructionOptions opt;
        opt.require_complete_stencil = true;
        for (auto scheme : {ReconstructionScheme::Ebr5, ReconstructionScheme::Weno5}) {
            opt.scheme = scheme;
            auto st = full.prepare(opt);
            for (Id e = 0; e < v.edge_count; ++e)
                near(increment(st.view(), e, 0, {3.25}), 0);
        }
        for (auto bad : {parse("patch xm\npair xp\ntranslation 2 0 0"),
                         parse("patch xm\npair missing\ntranslation 1 0 0")}) {
            try {
                PeriodicDomain invalid(mesh, {bad});
            } catch (const MeshError&) {
                continue;
            }
            throw std::runtime_error("unmatched periodic patch accepted");
        }
    }
    std::cout << "Periodic matching, compact CV and reconstruction tests passed.\n";
}
