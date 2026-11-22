#include "cv.h"
#include "../../qa/test_checks.h"
#include "../mesh/mesh.h"
#include "../eulersolver/lax_friedrichs.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <string>
#ifdef AERO_WITH_KOKKOS_GPU
int check_cv_devices(aero::CvView, bool);
#endif
namespace {
using namespace aero;
void append_face(MeshData& d, const std::vector<MeshIndex>& vertices) {
    d.face_nodes.indices.insert(d.face_nodes.indices.end(), vertices.begin(), vertices.end());
    d.face_nodes.offsets.push_back(d.face_nodes.indices.size());
}
void mark_all_boundary(MeshData& d) {
    const std::size_t count = d.face_nodes.offsets.size() - 1;
    d.boundary_faces.resize(count);
    std::iota(d.boundary_faces.begin(), d.boundary_faces.end(), MeshIndex(0));
    d.boundary_patch.assign(count, 0);
}
void make_single_cell(MeshData& d) {
    const std::size_t count = d.face_nodes.offsets.size() - 1;
    d.cell_faces.indices.resize(count);
    std::iota(d.cell_faces.indices.begin(), d.cell_faces.indices.end(), MeshIndex(0));
    d.cell_faces.offsets = {0, static_cast<MeshIndex>(count)};
    d.cell_face_orientation.assign(count, 1);
    mark_all_boundary(d);
}

MeshData tetrahedron() {
    MeshData d;
    d.x = {0, 1, 0, 0};
    d.y = {0, 0, 1, 0};
    d.z = {0, 0, 0, 1};
    d.patch_names = {"wing"};
    append_face(d, {0, 2, 1});
    append_face(d, {0, 1, 3});
    append_face(d, {1, 2, 3});
    append_face(d, {2, 0, 3});
    make_single_cell(d);
    return d;
}

MeshData mixed() {
    MeshData d = tetrahedron();
    d.patch_names.push_back("farfield");
    d.x.insert(d.x.end(), {3, 4, 4, 3, 3, 4, 4, 3, 6, 7, 6, 6, 7, 6});
    d.y.insert(d.y.end(), {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1});
    d.z.insert(d.z.end(), {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1});
    append_face(d, {4, 7, 6, 5});
    append_face(d, {8, 9, 10, 11});
    append_face(d, {4, 5, 9, 8});
    append_face(d, {5, 6, 10, 9});
    append_face(d, {6, 7, 11, 10});
    append_face(d, {7, 4, 8, 11});
    append_face(d, {12, 14, 13});
    append_face(d, {15, 16, 17});
    append_face(d, {12, 13, 16, 15});
    append_face(d, {13, 14, 17, 16});
    append_face(d, {14, 12, 15, 17});
    d.cell_faces.offsets = {0, 4, 10, 15};
    d.cell_faces.indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    d.cell_face_orientation.assign(15, 1);
    mark_all_boundary(d);
    for (std::size_t i = 4; i < 10; ++i)
        d.boundary_patch[i] = 1;
    return d;
}

MeshData adjacent_tetrahedra() {
    MeshData d = tetrahedron();
    d.x.push_back(0);
    d.y.push_back(0);
    d.z.push_back(-1);
    append_face(d, {0, 2, 4});
    append_face(d, {2, 1, 4});
    append_face(d, {1, 0, 4});
    d.cell_faces = MeshCsr({0, 4, 8}, {0, 1, 2, 3, 0, 4, 5, 6});
    d.cell_face_orientation = {1, 1, 1, 1, -1, 1, 1, 1};
    d.boundary_faces = {1, 2, 3, 4, 5, 6};
    d.boundary_patch.assign(6, 0);
    return d;
}

MeshData polygonal_prism(std::size_t n) {
    MeshData d;
    d.patch_names = {"surface"};
    const double pi = std::acos(-1.0);
    for (unsigned layer = 0; layer < 2; ++layer)
        for (std::size_t i = 0; i < n; ++i) {
            const double angle = 2 * pi * static_cast<double>(i) / static_cast<double>(n);
            d.x.push_back(std::cos(angle));
            d.y.push_back(std::sin(angle));
            d.z.push_back(layer);
        }
    std::vector<MeshIndex> bottom, top;
    for (std::size_t i = 0; i < n; ++i) {
        bottom.push_back(n - 1 - i);
        top.push_back(n + i);
    }
    append_face(d, bottom);
    append_face(d, top);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = (i + 1) % n;
        append_face(d, {i, j, n + j, n + i});
    }
    make_single_cell(d);
    return d;
}

MeshData line_mesh() {
    MeshData d;
    d.dimension = 1;
    d.x = {-2, -1.75, 0, 4};
    d.y.assign(4, 0);
    d.z.assign(4, 0);
    for (MeshIndex i = 0; i < 4; ++i)
        append_face(d, {i});
    d.cell_faces = MeshCsr({0, 2, 4, 6}, {0, 1, 1, 2, 2, 3});
    d.cell_face_orientation = {-1, 1, -1, 1, -1, 1};
    d.boundary_faces = {0, 3};
    d.boundary_patch = {0, 1};
    d.patch_names = {"left", "right"};
    return d;
}

MeshData polygon(std::size_t n) {
    MeshData d;
    d.dimension = 2;
    d.patch_names = {"wall"};
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double angle = 2 * pi * static_cast<double>(i) / static_cast<double>(n);
        d.x.push_back(std::cos(angle));
        d.y.push_back(std::sin(angle));
        d.z.push_back(0);
        append_face(d, {i, (i + 1) % n});
    }
    make_single_cell(d);
    return d;
}

MeshData adjacent_polygons() {
    MeshData d;
    d.dimension = 2;
    d.patch_names = {"wall"};
    d.x = {0, 1, 1, 0, 2};
    d.y = {0, 0, 1, 1, 0.5};
    d.z.assign(5, 0);
    // У четырёхугольника и треугольника общая грань с номером 0.
    append_face(d, {1, 2});
    append_face(d, {0, 1});
    append_face(d, {2, 3});
    append_face(d, {3, 0});
    append_face(d, {1, 4});
    append_face(d, {4, 2});
    d.cell_faces = MeshCsr({0, 4, 7}, {0, 1, 2, 3, 0, 4, 5});
    d.cell_face_orientation = {1, 1, 1, 1, -1, 1, 1};
    d.boundary_faces = {1, 2, 3, 4, 5};
    d.boundary_patch.assign(5, 0);
    return d;
}

using aero_test::require;
void near(double value, double expected) {
    require(std::fabs(value - expected) < 2e-12 * (1 + std::fabs(expected)), "numerical mismatch");
}

void verify(const MeshData& data, double total, const std::vector<double>& nodal = {}) {
    const Mesh mesh(data);
    const MeshView m = mesh.view();
    const MedianDual cv(mesh);
    const CvView v = cv.view();
    require(v.dimension == data.dimension && v.node_count == m.node_count && v.edge_count == m.edge_count,
            "view shape");
    near(std::accumulate(v.volumes, v.volumes + v.node_count, 0.0), total);
    for (MeshIndex i = 0; i < v.node_count; ++i) {
        require(v.volumes[i] > 0, "positive nodal measure");
        if (!nodal.empty())
            near(v.volumes[i], nodal.at(i));
    }
    // Постоянное физическое состояние с ненулевыми давлением и скоростью во всех направлениях.
    const unsigned d = v.dimension;
    double state[5] = {1, .3, -.4, .5, 0};
    state[d + 1] = 3;
    std::vector<double> residual(v.node_count * (d + 2));
    const auto add_flux = [&](MeshIndex node, CvVector s, double sign) {
        const double n[] = {s.x, s.y, s.z};
        const double area = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
        EulerFluxResult result;
        require(lax_friedrichs_conservative(d, state, state, n, 1.4, &result) == EulerStatus::Success,
                "Euler flux");
        for (unsigned k = 0; k < d + 2; ++k)
            residual[node * (d + 2) + k] += sign * area * result.flux[k];
    };
    for (MeshIndex e = 0; e < v.edge_count; ++e) {
        const MeshIndex a = m.edge_begin[e], b = m.edge_end[e];
        const CvVector s = v.edge_areas[e];
        require((m.x[b] - m.x[a]) * s.x + (m.y[b] - m.y[a]) * s.y + (m.z[b] - m.z[a]) * s.z > 0,
                "edge area orientation");
        add_flux(a, s, 1);
        add_flux(b, s, -1);
    }
    MeshIndex piece = 0;
    for (MeshIndex j = 0; j < m.boundary_face_count; ++j) {
        for (MeshIndex node : m.face_nodes(m.boundary_faces[j])) {
            require(piece < v.boundary_count, "boundary count");
            const CvBoundary p = v.boundary[piece++];
            require(p.node == node && p.patch == m.boundary_patch[j] && p.face == m.boundary_faces[j],
                    "boundary identity");
            add_flux(p.node, p.area, 1);
        }
    }
    require(piece == v.boundary_count, "unexpected boundary piece");
    for (double r : residual)
        near(r, 0);
}
void reject(MeshData data, const std::string& where) {
    const Mesh mesh(data);
    try {
        MedianDual bad(mesh);
    } catch (const MeshError& e) {
        require(std::string(e.what()).find(where) != std::string::npos, "missing geometry diagnostic");
        return;
    }
    throw std::runtime_error("invalid geometry accepted");
}
}
int main(int argc, char** argv) {
    try {
        const bool required = argc == 2 && std::string(argv[1]) == "--require-cuda";
        verify(line_mesh(), 6, {.125, 1, 2.875, 2});
        const MedianDual line{Mesh(line_mesh())};
        const double widths[] = {.25, .25, 1.75, 4.};
        for (unsigned i = 0; i < 4; ++i) near(line.view().min_heights[i], widths[i]);
        // Для ребра (0,1), идущего вдоль X, вектор площади двойственной грани равен (1/3, 1/3).
        MeshData triangle = polygon(3);
        triangle.x = {10, 12, 10};
        triangle.y = {20, 20, 21};
        verify(triangle, 1, {1. / 3, 1. / 3, 1. / 3});
        const MedianDual skew{Mesh(triangle)};
        near(skew.view().min_heights[0], 2. / std::sqrt(5.));
        near(skew.view().edge_areas[0].x, 1. / 3);
        near(skew.view().edge_areas[0].y, 1. / 3);
        MeshData cells = adjacent_polygons();
        cells.boundary_patch[0] = 1;
        cells.patch_names.push_back("wing");
        verify(cells, 1.5, {.25, 5. / 12, 5. / 12, .25, 1. / 6});
        MeshData pentagon = polygon(5);
        verify(pentagon, 2.5 * std::sin(2 * std::acos(-1.) / 5));
        verify(adjacent_tetrahedra(), 1. / 3, {1. / 12, 1. / 12, 1. / 12, 1. / 24, 1. / 24});
        verify(mixed(), 5. / 3);
        const MedianDual solids{Mesh(mixed())};
        near(solids.view().min_heights[0], 1. / std::sqrt(3.));   // Тетраэдр.
        near(solids.view().min_heights[4], 1.);                   // Куб.
        near(solids.view().min_heights[12], 1. / std::sqrt(2.)); // Треугольная призма.
        // Поднимаем один угол единичного гекса. Средняя высота верхней
        // грани растёт на 1/8, настолько же увеличивается объём под ней.
        MeshData warped = mixed();
        warped.z[10] += .5;
        verify(warped, 5. / 3 + .125);
        verify(polygonal_prism(9), 4.5 * std::sin(2 * std::acos(-1.) / 9));
        MeshData concave = polygon(5);
        concave.x = {0, 2, 2, 1, 0};
        concave.y = {0, 0, 2, 1.5, 2};
        verify(concave, 3.5);
        concave.y[3] = .1;
        reject(concave, "cell 0");
        MeshData flat = tetrahedron();
        flat.z[3] = 0;
        reject(flat, "cell 0");
        flat.z[3] = -1;
        reject(flat, "cell 0");
        cells.z[1] = .1;
        reject(cells, "cell 0");
        std::cout << "CPU median dual tests passed (1D, 2D, 3D).\n";
#ifdef AERO_WITH_KOKKOS_GPU
        const MedianDual cv{Mesh(mixed())};
        return check_cv_devices(cv.view(), required);
#else
        return required ? 1 : 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
