#include "cv.h"
#include "../../tests/test_checks.h"
#include "../../tests/mesh_fixture.h"
#include "../mesh/mesh.h"
#include "../eulersolver/lax_friedrichs.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <string>
#ifdef AERO_WITH_KOKKOS_GPU
// Сравнивает геометрию контрольных объёмов на CPU и доступных GPU.
int check_cv_devices(aero::CvView, bool);
#endif
namespace {
using namespace aero;
using namespace aero_test::mesh_fixture;
using aero_test::require;
// Проверяет совпадение чисел в пределах заданного допуска.
void near(double value, double expected) {
    require(std::fabs(value - expected) < 2e-12 * (1 + std::fabs(expected)), "numerical mismatch");
}

// Проверяет геометрические инварианты и ожидаемые контрольные объёмы.
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
    // Добавляет ориентированный вклад площадки в тестовый баланс.
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
// Проверяет отклонение некорректной геометрии сетки.
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
// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main(int argc, char** argv) {
    try {
        const bool required = argc == 2 && aero_test::require_gpu_argument(argv[1]);
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
