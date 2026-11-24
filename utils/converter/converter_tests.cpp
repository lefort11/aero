#include "converter.h"
#include "../../tests/test_checks.h"
#include "cli.h"
#include "detail.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#ifdef AERO_WITH_CGNS
#include <cgnslib.h>
#endif
namespace {
using namespace aero;
using namespace aero::converter;
typedef std::pair<std::string, std::function<void()>> Test;
std::vector<Test> tests;
using aero_test::require;
// Проверяет, что некорректный ввод завершается ожидаемой ошибкой.
template <class F> void rejects(F f) {
    try {
        f();
    } catch (const MeshError&) {
        return;
    }
    throw std::runtime_error("Expected MeshError");
}
// Добавляет именованную проверку в список тестов модуля.
void add(const std::string& s, std::function<void()> f) {
    tests.push_back(std::make_pair(s, f));
}
// Создаёт текстовую заготовку Gmsh для проверки конвертера.
Mesh gmsh(const std::string& s) {
    std::istringstream in(s, std::ios::in | std::ios::binary);
    return read_gmsh(in);
}
// Читает двоичное значение исходного типа с учётом порядка байтов.
template <class T> void raw(std::ostream& out, T value, bool reverse = false) {
    char b[sizeof(T)];
    std::memcpy(b, &value, sizeof(T));
    if (reverse)
        std::reverse(b, b + sizeof(T));
    out.write(b, sizeof(T));
}
// Одна и та же геометрия проверяет версии Gmsh, порядок байтов и текстовую запись.
std::string triangle(bool modern, bool binary, bool reverse = false) {
    std::ostringstream o(std::ios::out | std::ios::binary);
    o << "$MeshFormat\n" << (modern ? "4.1" : "2.2") << " " << (binary ? 1 : 0) << " 8\n";
    if (binary) {
        raw<std::int32_t>(o, 1, reverse);
        o << "\n";
    }
    o << "$EndMeshFormat\n$PhysicalNames\n1\n1 7 \"wing\"\n$EndPhysicalNames\n";
    // Записывает целое значение нужной разрядности в тестовый файл Gmsh.
    const auto integer = [&](int n) {
        if (binary)
            raw<std::int32_t>(o, n, reverse);
        else
            o << n << " ";
    };
    // Записывает целое значение нужной разрядности в тестовый файл Gmsh.
    const auto size = [&](std::uint64_t n) {
        if (binary)
            raw<std::uint64_t>(o, n, reverse);
        else
            o << n << " ";
    };
    // Читает или записывает вещественный параметр в формате текущей операции.
    const auto real = [&](double n) {
        if (binary)
            raw<double>(o, n, reverse);
        else
            o << n << " ";
    };
    if (modern) {
        o << "$Entities\n";
        size(0);
        size(1);
        size(1);
        size(0);
        integer(1);
        for (int i = 0; i < 6; ++i)
            real(i > 2 ? 1 : 0);
        size(1);
        integer(7);
        size(0);
        integer(1);
        for (int i = 0; i < 6; ++i)
            real(i > 2 ? 1 : 0);
        size(0);
        size(1);
        integer(1);
        o << "\n$EndEntities\n";
    }
    o << "$Nodes\n";
    const int tags[] = {30, 10, 90};
    const double xy[] = {0, 0, 1, 0, 0, 1};
    if (modern) {
        size(1);
        size(3);
        size(10);
        size(90);
        integer(2);
        integer(1);
        integer(0);
        size(3);
        for (int n : tags)
            size(static_cast<std::uint64_t>(n));
    } else
        o << "3\n";
    for (int i = 0; i < 3; ++i) {
        if (!modern) {
            if (binary)
                integer(tags[i]);
            else
                o << tags[i] << " ";
        }
        real(xy[2*i]);
        real(xy[2*i+1]);
        real(0);
    }
    o << "\n$EndNodes\n$Elements\n";
    if (modern) {
        size(2);
        size(4);
        size(1);
        size(4);
        integer(1);
        integer(1);
        integer(1);
        size(3);
    } else {
        o << "4\n";
        if (binary) {
            integer(1);
            integer(3);
            integer(2);
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (modern) {
            size(static_cast<std::uint64_t>(i + 1));
            size(static_cast<std::uint64_t>(tags[i]));
            size(static_cast<std::uint64_t>(tags[(i + 1) % 3]));
        } else {
            integer(i + 1);
            if (!binary) {
                integer(1);
                integer(2);
            }
            integer(7);
            integer(1);
            integer(tags[i]);
            integer(tags[(i + 1) % 3]);
        }
    }
    if (modern) {
        integer(2);
        integer(1);
        integer(2);
        size(1);
        size(4);
        for (int n : tags)
            size(static_cast<std::uint64_t>(n));
    } else {
        if (binary) {
            integer(2);
            integer(1);
            integer(2);
        }
        integer(4);
        if (!binary) {
            integer(2);
            integer(2);
        }
        integer(0);
        integer(1);
        for (int n : tags)
            integer(n);
    }
    o << "\n$EndElements\n";
    return o.str();
}
// Создаёт минимальную тестовую сетку внешнего формата.
std::string simple(int kind, const std::vector<double>& x, const std::vector<double>& y,
                   const std::vector<double>& z) {
    std::ostringstream o;
    o << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n" << x.size() << "\n";
    for (std::size_t i = 0; i < x.size(); ++i)
        o << i + 1 << " " << x[i] << " " << y[i] << " " << z[i] << "\n";
    o << "$EndNodes\n$Elements\n1\n1 " << kind << " 0";
    for (std::size_t i = 0; i < x.size(); ++i)
        o << " " << i + 1;
    o << "\n$EndElements\n";
    return o.str();
}
// Меняет указанный фрагмент заготовки для проверки повреждённого ввода.
std::string replace(std::string s, const std::string& before, const std::string& after) {
    const std::size_t p = s.find(before);
    require(p != std::string::npos, "Missing test token");
    s.replace(p, before.size(), after);
    return s;
}
// Владеет временным файлом проверки и удаляет его при выходе из области видимости.
struct Temp {
    std::string path;
    // Создаёт уникальное имя временного файла теста.
    Temp() {
        static unsigned n = 0;
        path = "build/converter-test-" +
               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + "-" +
               std::to_string(n++) + ".cgns";
    }
    // Удаляет временные файлы и каталоги, созданные объектом.
    ~Temp() { std::remove(path.c_str()); }
};
// Временный набор файлов формата NTTE для проверок конвертера.
struct AnotherFixture {
    std::string path;
    std::vector<std::string> files;
    // Создаёт временный набор файлов для проверки формата NTTE.
    AnotherFixture() {
        Temp name;
        path = name.path + "-ntte";
#ifdef _WIN32
        require(_mkdir(path.c_str()) == 0, "Create test directory");
#else
        require(mkdir(path.c_str(), 0700) == 0, "Create test directory");
#endif
    }
    // Записывает один файл многокомпонентной тестовой сетки.
    void put(const std::string& name, const std::string& data) {
        const std::string file = path + "/" + name;
        if (std::find(files.begin(), files.end(), file) == files.end())
            files.push_back(file);
        std::ofstream out(file.c_str(), std::ios::binary | std::ios::trunc);
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        require(static_cast<bool>(out), "Write test fixture");
    }
    // Удаляет временные файлы и каталоги, созданные объектом.
    ~AnotherFixture() {
        for (const std::string& file : files)
            std::remove(file.c_str());
#ifdef _WIN32
        _rmdir(path.c_str());
#else
        rmdir(path.c_str());
#endif
    }
    // Создаёт квадратную тестовую область внешнего формата.
    void square(bool triangles = true) {
        put("mesh.txt", std::string("Name test\nNumCoords 2\nNum_Node 4\nNum_Elem ") +
                            (triangles ? "2" : "1") +
                            "\nNum_BndFace 4\nMeshFormat TEXT\nMeshType HYBRID\nMaxNBLevel -1\n");
        put("coordinate.msh", "0 0\n1 0\n0 1\n1 1\n");
        put("topo.msh", triangles ? "3 0 1 3\n3 0 3 2\n" : "4 0 1 2 3\n");
        put("bctopo.msh", "2 0 1 1\n2 1 3 2\n2 3 2 3\n2 2 0 4\n");
        put("label.txt", "# Boundary names\n1 bottom\n2 right\n3 top\n4 left\n");
    }
    // Создаёт гексаэдральную тестовую область внешнего формата.
    void hex() {
        put("mesh.txt", "Name cube\nNumCoords 3\nNn 8\nNt 1\nNFaceBC 6\nMeshFormat 0\n"
                        "BufferZoneType no\nNnBase -1\nNtBase -1\n");
        put("label.txt", "1\n# aliases are not boundary-condition types\n1 wall WALL # comment\n");
        put("coordinate.msh", "0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n");
        put("topo.msh", "0 1 2 3 4 5 6 7\n");
        put("bctopo.msh", "0 2 3 1 1\n4 5 7 6 1\n0 1 5 4 1\n1 3 7 5 1\n3 2 6 7 1\n2 0 4 6 1\n");
    }
};
// Проверяем, что импортированная сетка переживает сохранение и повторную загрузку.
void roundtrip(const Mesh& m) {
    for (MeshFormat f : {MeshFormat::Text, MeshFormat::Binary}) {
        std::ostringstream out(std::ios::out | std::ios::binary);
        m.write(out, f);
        std::istringstream in(out.str(), std::ios::in | std::ios::binary);
        Mesh n = Mesh::read(in);
        require(n.data().face_nodes.indices == m.data().face_nodes.indices &&
                    n.data().cell_faces.indices == m.data().cell_faces.indices &&
                    n.data().cell_face_orientation == m.data().cell_face_orientation &&
                    n.data().patch_names == m.data().patch_names,
                "Native round trip");
    }
}
#ifdef AERO_WITH_CGNS
// Проверяет код возврата CGNS и преобразует ошибку библиотеки в исключение.
void cgcheck(int code) {
    if (code != CG_OK)
        throw std::runtime_error(cg_get_error());
}
// Владеет дескриптором тестового файла CGNS и закрывает его после проверки.
struct CgFile {
    int fn;
    // Создаёт тестовый файл CGNS с выбранным форматом хранения.
    CgFile(const std::string& path, int backend) {
        cgcheck(cg_set_file_type(backend));
        cgcheck(cg_open(path.c_str(), CG_MODE_WRITE, &fn));
    }
    // Закрывает тестовый файл CGNS с проверкой статуса.
    void close() {
        cgcheck(cg_close(fn));
        fn = -1;
    }
    // Закрывает принадлежащий объекту файл CGNS.
    ~CgFile() {
        if (fn != -1)
            cg_close(fn);
    }
};
// Записывает координатные массивы тестовой зоны CGNS.
void coordinates(int fn, int b, int z, const std::vector<double>& x, const std::vector<double>& y,
                 const std::vector<double>& zz, int dim = 3) {
    int c;
    cgcheck(cg_coord_write(fn, b, z, CGNS_ENUMV(RealDouble), "CoordinateX", x.data(), &c));
    if (dim >= 2)
        cgcheck(cg_coord_write(fn, b, z, CGNS_ENUMV(RealDouble), "CoordinateY", y.data(), &c));
    if (dim >= 3)
        cgcheck(cg_coord_write(fn, b, z, CGNS_ENUMV(RealDouble), "CoordinateZ", zz.data(), &c));
}
// Создаёт и проверяет тетраэдральную сетку CGNS.
void cgns_tet(const std::string& path, int backend, bool mixed = false, bool poly = false) {
    CgFile f(path, backend);
    int b, z, s, bc;
    cgsize_t sizes[] = {4, 1, 0};
    cgcheck(cg_base_write(f.fn, "Base", 3, 3, &b));
    cgcheck(cg_zone_write(f.fn, b, "Zone", sizes, CGNS_ENUMV(Unstructured), &z));
    coordinates(f.fn, b, z, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1});
    const cgsize_t faces[] = {1, 3, 2, 1, 2, 4, 2, 3, 4, 3, 1, 4};
    if (poly) {
        cgsize_t offsets[] = {0, 3, 6, 9, 12};
        cgcheck(cg_poly_section_write(f.fn, b, z, "Faces", CGNS_ENUMV(NGON_n), 2, 5, 0, faces, offsets, &s));
        cgsize_t cell[] = {2, 3, 4, 5}, off[] = {0, 4};
        cgcheck(cg_poly_section_write(f.fn, b, z, "Cells", CGNS_ENUMV(NFACE_n), 1, 1, 0, cell, off, &s));
    } else if (mixed) {
        std::vector<cgsize_t> conn = {CGNS_ENUMV(TETRA_4), 1, 2, 3, 4}, offset = {0, 5};
        for (int i = 0; i < 4; ++i) {
            conn.push_back(CGNS_ENUMV(TRI_3));
            for (int j = 0; j < 3; ++j)
                conn.push_back(faces[3 * i + j]);
            offset.push_back(static_cast<cgsize_t>(conn.size()));
        }
        cgcheck(cg_poly_section_write(f.fn, b, z, "Mixed", CGNS_ENUMV(MIXED), 1, 5, 0, conn.data(),
                                      offset.data(), &s));
    } else {
        cgsize_t tet[] = {1, 2, 3, 4};
        cgcheck(cg_section_write(f.fn, b, z, "Cells", CGNS_ENUMV(TETRA_4), 1, 1, 0, tet, &s));
        cgcheck(cg_section_write(f.fn, b, z, "Faces", CGNS_ENUMV(TRI_3), 2, 5, 0, faces, &s));
    }
    cgsize_t range[] = {2, 5};
    cgcheck(cg_boco_write(f.fn, b, z, "wing", CGNS_ENUMV(BCWall), CGNS_ENUMV(PointRange), 2, range, &bc));
    cgcheck(cg_boco_gridlocation_write(f.fn, b, z, bc, CGNS_ENUMV(FaceCenter)));
    f.close();
}
// Создаёт и проверяет структурированную зону CGNS.
void cgns_structured(const std::string& path, int backend, int dim) {
    CgFile f(path, backend);
    int b, z, bc;
    cgsize_t sizes[9] = {0};
    for (int i = 0; i < dim; ++i) {
        sizes[i] = 2;
        sizes[dim + i] = 1;
    }
    if (dim == 1) {
        sizes[0] = 4;
        sizes[1] = 3;
    }
    cgcheck(cg_base_write(f.fn, "Base", dim, dim, &b));
    cgcheck(cg_zone_write(f.fn, b, "Zone", sizes, CGNS_ENUMV(Structured), &z));
    std::vector<double> x, y, zz;
    if (dim == 1) {
        x = {0, 0.25, 2, 7};
        y.assign(4, 0);
        zz.assign(4, 0);
    } else
        for (int k = 0; k < (dim == 3 ? 2 : 1); ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    x.push_back(i);
                    y.push_back(j);
                    zz.push_back(k);
                }
    coordinates(f.fn, b, z, x, y, zz, dim);
    if (dim == 1) {
        cgsize_t p = 1;
        cgcheck(cg_boco_write(f.fn, b, z, "left", CGNS_ENUMV(BCGeneral), CGNS_ENUMV(PointList), 1, &p, &bc));
        p = 4;
        cgcheck(cg_boco_write(f.fn, b, z, "right", CGNS_ENUMV(BCGeneral), CGNS_ENUMV(PointList), 1, &p, &bc));
    } else {
        cgsize_t range[6] = {1, 1, 1, 2, 2, 2};
        if (dim == 2) {
            range[2] = 2;
            range[3] = 2;
        }
        cgcheck(cg_boco_write(f.fn, b, z, "wall", CGNS_ENUMV(BCWall), CGNS_ENUMV(PointRange), 2, range, &bc));
    }
    f.close();
}
// Проверяет конвертацию многоугольных или полиэдральных элементов CGNS.
void cgns_polygon(const std::string& path, int backend) {
    CgFile f(path, backend);
    int b, z, s, bc;
    cgsize_t sizes[] = {5, 1, 0};
    cgcheck(cg_base_write(f.fn, "Base", 2, 2, &b));
    cgcheck(cg_zone_write(f.fn, b, "Zone", sizes, CGNS_ENUMV(Unstructured), &z));
    coordinates(f.fn, b, z, {0, 2, 2, 1, 0}, {0, 0, 2, 1, 2}, {}, 2);
    cgsize_t nodes[] = {1, 2, 3, 4, 5}, off[] = {0, 5};
    cgcheck(cg_poly_section_write(f.fn, b, z, "Polygon", CGNS_ENUMV(NGON_n), 1, 1, 0, nodes, off, &s));
    cgcheck(cg_boco_write(f.fn, b, z, "concave", CGNS_ENUMV(BCWall), CGNS_ENUMV(PointList), 5, nodes, &bc));
    f.close();
}
// Проверяет переиспользование общих граней при конвертации CGNS.
void cgns_shared(const std::string& path) {
    CgFile f(path, CG_FILE_HDF5);
    int b, z, s;
    cgsize_t sizes[] = {5, 2, 0};
    cgcheck(cg_base_write(f.fn, "Base", 3, 3, &b));
    cgcheck(cg_zone_write(f.fn, b, "Zone", sizes, CGNS_ENUMV(Unstructured), &z));
    coordinates(f.fn, b, z, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, -1});
    cgsize_t faces[] = {1, 3, 2, 1, 2, 4, 2, 3, 4, 3, 1, 4, 1, 3, 5, 3, 2, 5, 2, 1, 5},
             off[] = {0, 3, 6, 9, 12, 15, 18, 21};
    cgcheck(cg_poly_section_write(f.fn, b, z, "Faces", CGNS_ENUMV(NGON_n), 3, 9, 0, faces, off, &s));
    cgsize_t cells[] = {3, 4, 5, 6, -3, 7, 8, 9}, coff[] = {0, 4, 8};
    cgcheck(cg_poly_section_write(f.fn, b, z, "Cells", CGNS_ENUMV(NFACE_n), 1, 2, 0, cells, coff, &s));
    f.close();
}
#endif
// Импорт текстовых и двоичных сеток another с разными типами ячеек.
void register_another_meshes() {
    add("another 2D triangles and tensor quadrilateral", [] {
        for (bool triangles : {true, false}) {
            AnotherFixture f;
            f.square(triangles);
            Mesh m = read_another(f.path);
            require(m.view().dimension == 2 && m.view().node_count == 4 &&
                        m.view().cell_count == (triangles ? 2u : 1u),
                    "2D counts");
            require(m.view().boundary_face_count == 4 && m.view().patch_count == 4, "Boundary names");
            if (triangles)
                require(m.view().face_count == 5, "Shared interior edge");
            require(m.node(2).y() == 1 && m.node(2).z() == 0, "2D coordinates");
            roundtrip(m);
            Mesh via_header = read_another(f.path + "/mesh.txt");
            require(via_header.data().face_nodes.indices == m.data().face_nodes.indices, "Header path");
        }
    });
    add("another text 3D tensor hex", [] {
        AnotherFixture f;
        f.hex();
        Mesh m = read_another(f.path);
        require(m.view().dimension == 3 && m.cell(0).node_count() == 8 && m.cell(0).face_count() == 6 &&
                    m.cell(0).edge_count() == 12,
                "Hex topology");
        require(m.node(2).x() == 0 && m.node(2).y() == 1 && m.node(4).z() == 1,
                "Interleaved coordinates");
        require(m.data().patch_names == std::vector<std::string>({"wall"}), "Canonical label");
        require(m.view().boundary_face_count == 6, "All six faces mapped");
        roundtrip(m);
    });
    add("another 2D concave polygon", [] {
        AnotherFixture f;
        f.square(false);
        f.put("mesh.txt",
              "NumCoords 2\nNum_Node 5\nNum_Elem 1\nNum_BndFace 5\nMeshFormat text\nMeshType hybrid\n");
        f.put("coordinate.msh", "0 0\n2 0\n2 2\n1 1\n0 2\n");
        f.put("topo.msh", "5 0 1 2 3 4\n");
        f.put("bctopo.msh", "2 0 1 1\n2 1 2 1\n2 2 3 1\n2 3 4 1\n2 4 0 1\n");
        f.put("label.txt", "1 wing\n");
        Mesh m = read_another(f.path);
        require(m.cell(0).node_count() == 5, "Polygon arity");
        roundtrip(m);
    });
    add("another hybrid 3D tetrahedron prism pyramid", [] {
        const std::string coords[] = {"0 0 0\n1 0 0\n0 1 0\n0 0 1\n",
                                      "0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 1\n0 1 1\n",
                                      "0 0 0\n1 0 0\n0 1 0\n1 1 0\n0.5 0.5 1\n"};
        const std::string topo[] = {"4 0 1 2 3\n", "6 0 1 2 3 4 5\n", "5 0 1 2 3 4\n"};
        const std::string bc[] = {"3 0 2 1 1\n3 0 1 3 1\n3 1 2 3 1\n3 2 0 3 1\n",
                                  "3 0 2 1 1\n3 3 4 5 1\n4 0 1 4 3 1\n4 1 2 5 4 1\n4 2 0 3 5 1\n",
                                  "4 0 2 3 1 1\n3 0 1 4 1\n3 1 3 4 1\n3 3 2 4 1\n3 2 0 4 1\n"};
        const unsigned nodes[] = {4, 6, 5}, faces[] = {4, 5, 5};
        for (unsigned i = 0; i < 3; ++i) {
            AnotherFixture f;
            f.put("mesh.txt", "NumCoords 3\nNum_Node " + std::to_string(nodes[i]) +
                                  "\nNum_Elem 1\nNum_BndFace " + std::to_string(faces[i]) +
                                  "\nMeshFormat text\nMeshType hybrid\n");
            f.put("coordinate.msh", coords[i]);
            f.put("topo.msh", topo[i]);
            f.put("bctopo.msh", bc[i]);
            f.put("label.txt", "1 wall\n");
            Mesh m = read_another(f.path);
            require(m.cell(0).node_count() == nodes[i] && m.cell(0).face_count() == faces[i], "Hybrid shape");
            roundtrip(m);
        }
    });
}

// Проверяем формат входа, целостность текстовых массивов и связи сетки another.
void register_another_errors() {
    add("another rejects truncated or extra text arrays", [] {
        for (const char* name : {"coordinate.msh", "topo.msh", "bctopo.msh"}) {
            AnotherFixture f;
            f.square();
            f.put(name, "");
            rejects([&] { read_another(f.path); });
            f.square();
            std::ifstream in((f.path + "/" + name).c_str());
            std::ostringstream data;
            data << in.rdbuf();
            in.close();
            f.put(name, data.str() + "999\n");
            rejects([&] { read_another(f.path); });
        }
    });
    add("another rejects binary format before opening arrays", [] {
        for (const char* format : {"binary", "1"}) {
            AnotherFixture f;
            f.put("mesh.txt", std::string("MeshFormat ") + format + "\n");
            Temp output;
            const char* argv[] = {"mesh_converter", f.path.c_str(), output.path.c_str()};
            std::ostringstream out, err;
            require(run_cli(3, argv, out, err) != 0 &&
                    err.str().find("only text .msh input is supported") != std::string::npos,
                    "Binary another input must be rejected by its descriptor");
            require(!std::ifstream(output.path.c_str()).good(), "Rejected conversion created an output file");
        }
    });
    add("another rejects invalid text metadata and connectivity", [] {
        const std::pair<std::string, std::string> invalid[] = {
            {"coordinate.msh", "0 0\n1 0\n0 nan\n1 1\n"},
            {"topo.msh", "3 0 1 99\n3 0 3 2\n"},
            {"topo.msh", "3 0 1 1\n3 0 3 2\n"},
            {"bctopo.msh", "2 0 1 99\n2 1 3 2\n2 3 2 3\n2 2 0 4\n"},
            {"bctopo.msh", "2 0 1 1\n2 0 1 1\n2 3 2 3\n2 2 0 4\n"},
            {"bctopo.msh", "2 0 3 1\n2 1 3 2\n2 3 2 3\n2 2 0 4\n"},
            {"label.txt", "1 a\n1 b\n"},
            {"label.txt", "5\n1 a\n2 b\n3 c\n4 d\n"}};
        for (const auto& change : invalid) {
            AnotherFixture f;
            f.square();
            f.put(change.first, change.second);
            rejects([&] { read_another(f.path); });
        }
        for (const char* extra :
             {"Num_Node 4\n", "Num_Node_Base 3\n", "Num_Elem_Base 1\n", "BufferZoneType nodal\n"}) {
            AnotherFixture f;
            f.square();
            std::ifstream in((f.path + "/mesh.txt").c_str());
            std::ostringstream h;
            h << in.rdbuf();
            in.close();
            f.put("mesh.txt", h.str() + extra);
            rejects([&] { read_another(f.path); });
        }
        AnotherFixture f;
        f.square();
        f.put("periodic03.msh", "0\n");
        rejects([&] { read_another(f.path); });
    });
}

// Проверки параметров конвертера, распознавания каталогов и формата вывода.
void register_converter_cli() {
    add("another CLI directory detection and output encoding", [] {
        AnotherFixture f;
        f.hex();
        Options options;
        const Mesh reference = read_another(f.path, options);
        for (const char* format : {"text", "binary"}) {
            Temp output;
            const char* argv[] = {"mesh_converter", f.path.c_str(), output.path.c_str(),
                                  "--format", format, "--from", "another"};
            const bool binary = std::string(format) == "binary";
            std::ostringstream out, err;
            require(run_cli(binary ? 7 : 5, argv, out, err) == 0, "CLI conversion: " + err.str());
            // Сверяем весь файл, включая кодировку, координаты, связность и метки.
            std::ostringstream expected(std::ios::out | std::ios::binary);
            reference.write(expected, binary ? MeshFormat::Binary : MeshFormat::Text);
            std::ifstream file(output.path.c_str(), std::ios::binary);
            std::ostringstream actual;
            actual << file.rdbuf();
            require(actual.str() == expected.str(), "Wrong output encoding or mesh data");
            const Mesh loaded = Mesh::read(output.path);
            require(loaded.cell(0).node_count() == 8,
                    "Reload converted another mesh");
        }
    });
    add("CLI LZMA output", [] {
        Temp output;
        const char* argv[] = {"mesh_converter", "utils/converter/examples/line.msh", output.path.c_str(),
                              "--format", "lzma"};
        std::ostringstream out, err;
        require(run_cli(5, argv, out, err) == 0, "LZMA CLI: " + err.str());
        std::ifstream file(output.path.c_str(), std::ios::binary);
        char signature[8];
        file.read(signature, 8);
        require(std::string(signature, 8) == "AEROLZMA", "LZMA CLI signature");
        const Mesh loaded = Mesh::read(output.path);
        require(loaded.data().dimension == 1 && loaded.view().cell_count > 0, "LZMA CLI mesh");
    });
    add("CLI missing positional arguments prints manual", [] {
        const std::vector<std::vector<std::string>> cases = {
            {"mesh_converter"},
            {"mesh_converter", "utils/converter/examples/line.msh"},
            {"mesh_converter", "", "output.mesh"},
            {"mesh_converter", "input.msh", ""},
            {"mesh_converter", "input.msh", "--format", "text"},
            {"mesh_converter", "--format", "text"}};
        for (const auto& args : cases) {
            std::vector<const char*> argv;
            for (const std::string& arg : args)
                argv.push_back(arg.c_str());
            std::ostringstream out, err;
            require(run_cli(static_cast<int>(argv.size()), argv.data(), out, err) == 1,
                    "Missing argument exit code");
            require(out.str().empty(), "Missing argument help goes to stderr");
            require(err.str().find("Usage: mesh_converter INPUT OUTPUT") != std::string::npos &&
                        err.str().find("INPUT   Input mesh") != std::string::npos &&
                        err.str().find("OUTPUT  Output mesh path") != std::string::npos &&
                        err.str().find("--format text|binary") != std::string::npos &&
                        err.str().find("Example:") != std::string::npos,
                    "Brief argument manual");
            require(err.str().find("Conversion failed:") == std::string::npos, "No conversion attempted");
        }
    });
    add("CLI help flags succeed", [] {
        for (const char* flag : {"--help", "-h"}) {
            const char* argv[] = {"mesh_converter", flag};
            std::ostringstream out, err;
            require(run_cli(2, argv, out, err) == 0 && err.str().empty(), "Help exit/stream");
            require(out.str().find("--base N") != std::string::npos &&
                        out.str().find("--zone N") != std::string::npos &&
                        out.str().find("--scale") == std::string::npos &&
                        out.str().find("--unmarked NAME") != std::string::npos,
                    "Option descriptions");
        }
    });
    add("CLI two positional paths and output encoding", [] {
        for (bool text : {false, true}) {
            Temp output;
            output.path += " spaced output.mesh";
            const char* argv[] = {"mesh_converter", "utils/converter/examples/line.msh", output.path.c_str(),
                                  "--format", "text"};
            std::ostringstream out, err;
            require(run_cli(text ? 5 : 3, argv, out, err) == 0 && err.str().empty(),
                    "Conversion exit/stream");
            Mesh m = Mesh::read(output.path);
            require(m.view().dimension == 1 && m.view().cell_count == 3 &&
                        m.data().patch_names == std::vector<std::string>({"left", "right"}),
                    "Input/output argument order");
            std::ifstream file(output.path.c_str(), std::ios::binary);
            char magic[8];
            file.read(magic, 8);
            require(std::string(magic, 8) == (text ? "AEROMESH" : "AEROMSHB"), "Requested/default encoding");
            file.close();
            std::ostringstream again, errors;
            require(run_cli(text ? 5 : 3, argv, again, errors) == 1 &&
                        errors.str().find("already exists") != std::string::npos,
                    "Overwrite rejected");
        }
    });
    add("CLI invalid option does not create output", [] {
        Temp output;
        const char* argv[] = {"mesh_converter", "utils/converter/examples/line.msh", output.path.c_str(),
                              "--format", "wrong"};
        std::ostringstream out, err;
        require(run_cli(5, argv, out, err) == 1 &&
                    err.str().find("Output format must be") != std::string::npos,
                "Invalid format error");
        std::ifstream file(output.path.c_str());
        require(!file, "Output created before argument validation");
    });
}

// Сравнение версий Gmsh, разных порядков байтов, типов ячеек и граничных меток.
void register_gmsh() {
    for (bool modern : {false, true})
        for (bool binary : {false, true})
            for (bool reverse : {false, true}) {
                if (!binary && reverse)
                    continue;
                const std::string name = std::string("Gmsh ") + (modern ? "4.1 " : "2.2 ") +
                                         (binary ? (reverse ? "big endian" : "little endian") : "ASCII");
                add(name, [modern, binary, reverse] {
                    Mesh m = gmsh(triangle(modern, binary, reverse));
                    require(m.view().dimension == 2 && m.view().cell_count == 1 && m.view().face_count == 3,
                            "Triangle topology");
                    require(m.data().patch_names == std::vector<std::string>({"wing"}), "Physical name");
                    require(m.node(1).x() == 1 && m.node(2).y() == 1, "Sparse node tags");
                    roundtrip(m);
                });
                add(name + " truncated", [modern, binary, reverse] {
                    const std::string s = triangle(modern, binary, reverse);
                    const std::size_t end = s.rfind("$EndElements") + 12;
                    for (std::size_t i = 0; i < end; ++i)
                        rejects([&] { gmsh(s.substr(0, i)); });
                });
            }
    add("Gmsh nonuniform 1D", [] {
        Mesh m = gmsh("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n1 0 0 0\n2 0.25 0 0\n3 5 0 "
                      "0\n$EndNodes\n$Elements\n2\n1 1 0 1 2\n2 1 0 2 3\n$EndElements\n");
        require(m.view().dimension == 1 && m.view().cell_count == 2 && m.view().boundary_face_count == 2,
                "1D topology");
        roundtrip(m);
    });
    add("Gmsh linear 3D cell families", [] {
        const std::string inputs[] = {
            simple(4, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}),
            simple(5, {0, 1, 1, 0, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 0, 1, 1}, {0, 0, 0, 0, 1, 1, 1, 1}),
            simple(6, {0, 1, 0, 0, 1, 0}, {0, 0, 1, 0, 0, 1}, {0, 0, 0, 1, 1, 1}),
            simple(7, {0, 1, 1, 0, 0.5}, {0, 0, 1, 1, 0.5}, {0, 0, 0, 0, 1})};
        const MeshIndex counts[] = {4, 6, 5, 5};
        for (unsigned i = 0; i < 4; ++i) {
            Mesh m = gmsh(inputs[i]);
            require(m.cell(0).face_count() == counts[i], "Face templates");
            roundtrip(m);
        }
    });
    add("Gmsh rejects bad metadata and connectivity", [] {
        const std::string good = triangle(false, false);
        const std::pair<std::string, std::string> changes[] = {{"2.2 0 8", "3.0 0 8"},
                                                               {"2.2 0 8", "2.2 2 8"},
                                                               {"2.2 0 8", "2.2 0 4"},
                                                               {"30 0 0 0", "0 0 0 0"},
                                                               {"90 0 1 0", "10 0 1 0"},
                                                               {"4 2 2 0 1 30 10 90", "4 2 2 0 1 30 10 999"},
                                                               {"4 2 2 0 1 30 10 90", "4 9 2 0 1 30 10 90"},
                                                               {"$Nodes\n3", "$Nodes\n-3"},
                                                               {"$Nodes\n3", "$Nodes\n100000001"},
                                                               {"$EndNodes", "$WrongEnd"},
                                                               {"$Elements\n4", "$Elements\n5"}};
        for (const auto& change : changes)
            rejects([&] { gmsh(replace(good, change.first, change.second)); });
        rejects([&] { gmsh(good + "$Periodic\n0\n$EndPeriodic\n"); });
        rejects([&] { gmsh(good + "$Nodes\n"); });
    });
    add("Gmsh rejects multiple physical groups", [] {
        std::string s = triangle(true, false);
        s = replace(s, "1 7 0 ", "2 7 8 0 ");
        rejects([&] { gmsh(s); });
    });
    add("Gmsh coordinates and strict unmarked boundaries", [] {
        std::string s = simple(4, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1});
        Options opt;
        std::istringstream in(s);
        Mesh m = read_gmsh(in, opt);
        require(m.node(1).x() == 1, "Original coordinate");
        opt.unmarked_name.clear();
        std::istringstream bad(s);
        rejects([&] { read_gmsh(bad, opt); });
    });
    add("Builder rejects overlapping boundary names", [] {
        detail::Builder b(2, Options());
        b.data.x = {0, 1, 0};
        b.data.y = {0, 0, 1};
        b.data.z = {0, 0, 0};
        b.cell(detail::cell_faces(2, {0, 1, 2}));
        b.mark({0, 1}, "first");
        rejects([&] { b.mark({1, 0}, "second"); });
    });
}

// Проверки структурных и неструктурных CGNS-зон, а также сборки без CGNS.
void register_cgns() {
#ifdef AERO_WITH_CGNS
    for (int backend : {CG_FILE_ADF, CG_FILE_HDF5}) {
        const std::string p = backend == CG_FILE_ADF ? "CGNS ADF " : "CGNS HDF5 ";
        for (int mode = 0; mode < 3; ++mode)
            add(p + std::string(mode == 0 ? "tet" : (mode == 1 ? "mixed" : "NGON/NFACE")), [backend, mode] {
                Temp f;
                cgns_tet(f.path, backend, mode == 1, mode == 2);
                Mesh m = read_cgns(f.path);
                require(m.view().dimension == 3 && m.cell(0).node_count() == 4 && m.cell(0).face_count() == 4,
                        "CGNS tet");
                require(m.data().patch_names == std::vector<std::string>({"wing"}), "CGNS BC name only");
                roundtrip(m);
            });
        for (int dim = 1; dim <= 3; ++dim)
            add(p + "structured " + std::to_string(dim), [backend, dim] {
                Temp f;
                cgns_structured(f.path, backend, dim);
                Mesh m = read_cgns(f.path);
                require(m.view().dimension == static_cast<unsigned>(dim) &&
                            m.view().cell_count == (dim == 1 ? 3u : 1u),
                        "Structured conversion");
                roundtrip(m);
            });
        add(p + "concave polygon", [backend] {
            Temp f;
            cgns_polygon(f.path, backend);
            Mesh m = read_cgns(f.path);
            require(m.cell(0).node_count() == 5 && m.data().patch_names[0] == "concave", "NGON polygon");
            roundtrip(m);
        });
    }
    add("CGNS signed shared NFACE", [] {
        Temp f;
        cgns_shared(f.path);
        Mesh m = read_cgns(f.path);
        require(m.view().cell_count == 2 && m.view().face_count == 7 && m.view().boundary_face_count == 6,
                "Shared NGON");
        require(m.cell(0).face_orientation(0) == -m.cell(1).face_orientation(0), "Signed face reference");
        roundtrip(m);
    });
    add("CGNS missing file and invalid base/zone", [] {
        Temp f;
        rejects([&] { read_cgns(f.path); });
        cgns_tet(f.path, CG_FILE_HDF5);
        Options opt;
        opt.base = 2;
        rejects([&] { read_cgns(f.path, opt); });
        opt.base = 1;
        opt.zone = 2;
        rejects([&] { read_cgns(f.path, opt); });
    });
#else
    add("CGNS optional backend error", [] {
        require(!cgns_available(), "Backend disabled");
        rejects([] { read_cgns("unused"); });
    });
#endif
}

// Формирует набор проверок модуля и выполняемых тестовых сценариев.
void register_tests() {
    register_another_meshes();
    register_another_errors();
    register_converter_cli();
    register_gmsh();
    register_cgns();
}
}
// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main() {
    register_tests();
    std::size_t failed = 0;
    for (const Test& t : tests) {
        try {
            t.second();
            std::cout << "PASS " << t.first << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cerr << "FAIL " << t.first << ": " << e.what() << "\n";
        }
    }
    std::cout << tests.size() - failed << "/" << tests.size() << " converter tests passed\n";
    return failed ? 1 : 0;
}
