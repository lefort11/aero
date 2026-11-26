#pragma once
#include "../source/mesh/mesh.h"
#include <cmath>
#include <numeric>
#include <initializer_list>

namespace aero_test {
namespace mesh_fixture {
using aero::MeshCsr;
using aero::MeshData;
using aero::MeshIndex;

// Дописывает строку связности CSR и отмечает её конец. Для литерала используется
// initializer_list, для готового вектора передаётся ссылка на исходный массив.
template<class Nodes = std::initializer_list<MeshIndex>>
inline void append_row(MeshCsr& csr, const Nodes& vertices) {
    csr.indices.insert(csr.indices.end(), vertices.begin(), vertices.end());
    csr.offsets.push_back(csr.indices.size());
}
// Добавляет грань к сетке, сохраняя переданный порядок вершин.
template<class Nodes = std::initializer_list<MeshIndex>>
inline void append_face(MeshData& d, const Nodes& vertices) {
    append_row(d.face_nodes, vertices);
}

// Задаёт две последовательные ячейки одномерной сетки с общей средней вершиной.
// Имена и номера граничных участков задаёт владелец конкретного теста.
inline void make_two_intervals(MeshData& d, double last_node) {
    d.dimension = 1;
    d.x = {0, .25, last_node};
    d.y = d.z = {0, 0, 0};
    d.face_nodes = MeshCsr({0, 1, 2, 3}, {0, 1, 2});
    d.cell_faces = MeshCsr({0, 2, 4}, {0, 1, 1, 2});
    d.cell_face_orientation = {-1, 1, -1, 1};
    d.boundary_faces = {0, 2};
}
// Помечает все внешние грани тестовой сетки одним участком.
inline void mark_all_boundary(MeshData& d) {
    const std::size_t count = d.face_nodes.offsets.size() - 1;
    d.boundary_faces.resize(count);
    std::iota(d.boundary_faces.begin(), d.boundary_faces.end(), MeshIndex(0));
    d.boundary_patch.assign(count, 0);
}
// Создаёт сетку из одной ячейки с заданной оболочкой.
inline void make_single_cell(MeshData& d) {
    const std::size_t count = d.face_nodes.offsets.size() - 1;
    d.cell_faces.indices.resize(count);
    std::iota(d.cell_faces.indices.begin(), d.cell_faces.indices.end(), MeshIndex(0));
    d.cell_faces.offsets = {0, static_cast<MeshIndex>(count)};
    d.cell_face_orientation.assign(count, 1);
    mark_all_boundary(d);
}

// Создаёт тестовую тетраэдральную ячейку.
inline MeshData tetrahedron() {
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

// Создаёт тестовую сетку со смешанными типами ячеек.
inline MeshData mixed() {
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

// Создаёт соседние тетраэдры с общей гранью.
inline MeshData adjacent_tetrahedra() {
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

// Добавляет горизонтальный слой из равномерно расположенных вершин единичной окружности.
inline void append_ring(MeshData& d, std::size_t n, double z) {
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double angle = 2 * pi * static_cast<double>(i) / static_cast<double>(n);
        d.x.push_back(std::cos(angle));
        d.y.push_back(std::sin(angle));
        d.z.push_back(z);
    }
}

// Создаёт призматическую сетку для проверки геометрии и связности.
inline MeshData polygonal_prism(std::size_t n) {
    MeshData d;
    d.patch_names = {"surface"};
    for (unsigned layer = 0; layer < 2; ++layer)
        append_ring(d, n, layer);
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

// Создаёт одномерную сетку для проверки алгоритма.
inline MeshData line_mesh() {
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

// Создаёт многоугольную ячейку тестовой сетки.
inline MeshData polygon(std::size_t n) {
    MeshData d;
    d.dimension = 2;
    d.patch_names = {"wall"};
    append_ring(d, n, 0);
    for (std::size_t i = 0; i < n; ++i)
        append_face(d, {i, (i + 1) % n});
    make_single_cell(d);
    return d;
}

// Создаёт соседние многоугольники с общим ребром.
inline MeshData adjacent_polygons() {
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
}
}
