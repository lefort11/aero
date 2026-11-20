#include "mesh.h"
#include "../../qa/test_checks.h"
#include "../io/lzma_stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace {
using namespace aero;
typedef std::pair<std::string, std::function<void()>> Test;
std::vector<Test> tests;

using aero_test::require;
template <class F> void rejects(F operation, const std::string& expected = std::string()) {
    try {
        operation();
    } catch (const MeshError& error) {
        require(expected.empty() || std::string(error.what()).find(expected) != std::string::npos,
                "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected MeshError");
}
void add(const std::string& name, std::function<void()> test) {
    tests.emplace_back(name, std::move(test));
}

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

// Склеиваем полюса ориентированного икосаэдра. Связность по граням и кратность рёбер сохраняются,
// но окрестность получившейся вершины распадается на две части.
MeshData icosahedron() {
    MeshData d;
    d.patch_names = {"surface"};
    d.x.push_back(0);
    d.y.push_back(0);
    d.z.push_back(2);
    const double pi = std::acos(-1.0);
    for (unsigned layer = 0; layer < 2; ++layer)
        for (unsigned i = 0; i < 5; ++i) {
            const double angle = (2 * i + layer) * pi / 5;
            d.x.push_back(std::cos(angle));
            d.y.push_back(std::sin(angle));
            d.z.push_back(layer == 0 ? 1 : -1);
        }
    d.x.push_back(0);
    d.y.push_back(0);
    d.z.push_back(-2);
    for (MeshIndex i = 0; i < 5; ++i) {
        const MeshIndex j = (i + 1) % 5;
        append_face(d, {0, 1 + i, 1 + j});
        append_face(d, {1 + i, 6 + i, 1 + j});
        append_face(d, {1 + j, 6 + i, 6 + j});
        append_face(d, {11, 6 + j, 6 + i});
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
    // Четырёхугольник и треугольник имеют общую грань с номером 0.
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

bool same(const MeshData& a, const MeshData& b) {
    return a.dimension == b.dimension && a.length_scale_to_meters == b.length_scale_to_meters && a.x == b.x &&
           a.y == b.y && a.z == b.z && a.face_nodes.offsets == b.face_nodes.offsets &&
           a.face_nodes.indices == b.face_nodes.indices && a.cell_faces.offsets == b.cell_faces.offsets &&
           a.cell_faces.indices == b.cell_faces.indices &&
           a.cell_face_orientation == b.cell_face_orientation && a.boundary_faces == b.boundary_faces &&
           a.boundary_patch == b.boundary_patch && a.patch_names == b.patch_names;
}
std::string encode(const Mesh& mesh, MeshFormat format) {
    std::ostringstream out(std::ios::out | std::ios::binary);
    mesh.write(out, format);
    return out.str();
}
Mesh decode(const std::string& bytes, const MeshReadLimits& limits = MeshReadLimits()) {
    std::istringstream in(bytes, std::ios::in | std::ios::binary);
    return Mesh::read(in, limits);
}
std::string replace_once(std::string text, const std::string& before, const std::string& after) {
    const std::size_t p = text.find(before);
    require(p != std::string::npos, "Test token not found: " + before);
    text.replace(p, before.size(), after);
    return text;
}
struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};

class TempFile {
public:
    TempFile() {
        static unsigned count = 0;
        std::ostringstream name;
        name << "build/mesh-test-" << std::chrono::high_resolution_clock::now().time_since_epoch().count()
             << '-' << count++ << ".mesh";
        path = name.str();
        std::ifstream existing(path.c_str(), std::ios::binary);
        require(!existing, "Test file already exists");
    }
    ~TempFile() { std::remove(path.c_str()); }
    std::string path;

private:
    TempFile(const TempFile&);
    TempFile& operator=(const TempFile&);
};

// Чтение и запись сжатых сеток, контрольная сумма и повреждения потока.
void register_compressed_mesh() {
    add("LZMA round trips for all dimensions and empty meshes", [] {
        const MeshData cases[] = {MeshData(), line_mesh(), adjacent_polygons(), mixed(), polygonal_prism(40)};
        for (const MeshData& d : cases) {
            const Mesh original(d);
            const std::string packed = encode(original, MeshFormat::Lzma);
            require(packed.substr(0, 8) == "AEROLZMA", "Compressed signature");
            const Mesh loaded = decode(packed);
            require(same(d, loaded.data()), "Compressed data round trip");
            require(original.edge_begin() == loaded.edge_begin() && original.edge_end() == loaded.edge_end(),
                    "Derived edges");
            TempFile file;
            original.write(file.path, MeshFormat::Lzma);
            require(same(d, Mesh::read(file.path).data()), "Compressed file autodetection");
        }
    });
    add("LZMA exact floating point and UTF-8 names", [] {
        MeshData d = tetrahedron();
        d.y[0] = -0.0;
        d.x[1] = std::numeric_limits<double>::denorm_min();
        d.z[3] = std::numeric_limits<double>::max();
        d.patch_names[0] = u8"крыло \"A\"";
        const Mesh loaded = decode(encode(Mesh(d), MeshFormat::Lzma));
        require(same(d, loaded.data()) && std::signbit(loaded.data().y[0]), "Exact compressed values");
    });
    add("LZMA rejects every truncated prefix", [] {
        const std::string good = encode(Mesh(tetrahedron()), MeshFormat::Lzma);
        for (std::size_t n = 0; n < good.size(); ++n)
            rejects([&] { decode(good.substr(0, n)); });
    });
    add("LZMA rejects corrupt header blocks and trailing data", [] {
        const std::string good = encode(Mesh(tetrahedron()), MeshFormat::Lzma);
        // Последовательно идут версия, исходный и сжатый размеры, CRC, параметры, словарь, полезная нагрузка.
        for (std::size_t pos : {std::size_t(8), std::size_t(15), std::size_t(19), std::size_t(20),
                                std::size_t(24), std::size_t(28), std::size_t(29)}) {
            std::string bad = good;
            bad[pos] = static_cast<char>(static_cast<unsigned char>(bad[pos]) ^ 255);
            rejects([&] { decode(bad); });
        }
        rejects([&] { decode(good + "x"); }, "Trailing");
        rejects([&] { decode(good + good); }, "Trailing");
        std::string bad = good;
        bad.replace(12, 4, 4, '\0');
        rejects([&] { decode(bad); });
    });
    add("LZMA enforces payload and name limits", [] {
        const std::string packed = encode(Mesh(tetrahedron()), MeshFormat::Lzma);
        rejects([&] { decode(packed, MeshReadLimits(16)); }, "limit");
        rejects([&] { decode(packed, MeshReadLimits(100000, 2)); }, "name");
        require(decode(packed, MeshReadLimits(std::numeric_limits<std::uint64_t>::max())).view().cell_count ==
                    1,
                "Limit arithmetic");
    });
    add("LZMA multi-block mesh", [] {
        const Mesh original(polygon(18000));
        const std::string binary = encode(original, MeshFormat::Binary);
        require(binary.size() > 1024 * 1024, "Fixture crosses block boundary");
        const std::string packed = encode(original, MeshFormat::Lzma);
        require(packed.size() < binary.size(), "Compression reduces mesh size");
        require(same(original.data(), decode(packed).data()), "Multi-block mesh data");
        // Первый блок распаковывается, но ограничение на общий объём распаковки продолжает действовать.
        rejects([&] { decode(packed, MeshReadLimits(1024 * 1024)); }, "limit");
    });
    add("shared LZMA stream CRC and explicit finish", [] {
        std::ostringstream bytes(std::ios::binary);
        io::LzmaOutputBuffer buffer(bytes);
        std::ostream out(&buffer);
        out.exceptions(std::ios::badbit);
        out << "123456789";
        buffer.finish();
        buffer.finish();
        const std::string packed = bytes.str();
        // Стандартное контрольное значение CRC-32/ISO-HDLC равно 0xcbf43926, младший байт записан первым.
        require(packed.substr(20, 4) == std::string("\x26\x39\xf4\xcb", 4), "CRC known vector");
        bool failed = false;
        try {
            out.put('x');
        } catch (const io::CompressionError&) {
            failed = true;
        }
        require(failed, "Cannot write after finish");
        std::istringstream input(packed.substr(8), std::ios::binary);
        io::LzmaInputBuffer decoder(input, 9);
        std::istream in(&decoder);
        in.exceptions(std::ios::badbit);
        std::string result;
        std::getline(in, result);
        require(result == "123456789", "Shared stream independent of mesh");
    });
    add("shared LZMA stream flush empty and incompressible data", [] {
        for (bool empty : {true, false}) {
            std::string data(empty ? 0 : 2 * 1024 * 1024 + 19, '\0');
            std::uint32_t state = 123;
            for (char& c : data) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                c = static_cast<char>(state & 255);
            }
            std::ostringstream bytes(std::ios::binary);
            io::LzmaOutputBuffer buffer(bytes);
            std::ostream out(&buffer);
            out.exceptions(std::ios::badbit);
            const std::size_t split = data.size() / 2;
            out.write(data.data(), static_cast<std::streamsize>(split));
            out.flush();
            out.write(data.data() + split, static_cast<std::streamsize>(data.size() - split));
            buffer.finish();
            std::istringstream input(bytes.str().substr(8), std::ios::binary);
            io::LzmaInputBuffer decoder(input, data.size());
            std::istream in(&decoder);
            in.exceptions(std::ios::badbit);
            std::string result(data.size(), '\0');
            if (!result.empty())
                in.read(&result[0], static_cast<std::streamsize>(result.size()));
            require(in.peek() == std::char_traits<char>::eof() && result == data,
                    "Shared multi-block stream");
        }
    });
    add("LZMA rejects nonbinary inner payload and failed output", [] {
        std::ostringstream bytes;
        io::LzmaOutputBuffer buffer(bytes);
        std::ostream out(&buffer);
        out << encode(Mesh(tetrahedron()), MeshFormat::Text);
        buffer.finish();
        rejects([&] { decode(bytes.str()); }, "binary v2");
        std::ostringstream failed;
        failed.setstate(std::ios::badbit);
        rejects([&] { Mesh(tetrahedron()).write(failed, MeshFormat::Lzma); });
    });
}

// Проверки переходов от ячеек к граням, рёбрам и узлам.
void register_entity_access() {
    add("entity edge endpoints and node coordinates", [] {
        Mesh mesh(tetrahedron());
        const MeshEdge edge = mesh.edge(2); // Ребро (0,3) эталонного тетраэдра.
        require(edge.id() == 2 && edge.node_count() == 2, "Edge identity");
        require(edge.node_id(0) == 0 && edge.node_id(1) == 3, "Endpoint IDs");
        const MeshNode node = edge.node(1);
        require(node.id() == 3 && node.x() == 0 && node.y() == 0 && node.z() == 1, "Endpoint coordinates");
        require(mesh.node(1).x() == 1, "Direct node access");
    });
    add("entity face edge order and nested traversal", [] {
        Mesh mesh(tetrahedron());
        const MeshFace face = mesh.face(0);
        require(face.id() == 0 && face.node_count() == 3 && face.edge_count() == 3, "Face counts");
        const MeshIndex expected_nodes[] = {0, 2, 1}, expected_edges[] = {1, 3, 0};
        for (MeshIndex k = 0; k < 3; ++k) {
            require(face.node(k).id() == expected_nodes[k], "Face node order");
            require(face.edge(k).id() == expected_edges[k], "Face edge order");
        }
        require(face.edge(1).node(1).y() == 1, "Face to edge to node");
        MeshIndex i = 0;
        for (MeshIndex id : face.edge_ids())
            require(id == expected_edges[i++], "Range iteration");
        require(i == 3, "Iteration length");
    });
    add("entity cell unique nodes edges and faces in 3D", [] {
        Mesh mesh(mixed());
        const MeshIndex node_counts[] = {4, 8, 6}, edge_counts[] = {6, 12, 9}, face_counts[] = {4, 6, 5};
        const MeshIndex first_nodes[] = {0, 4, 12};
        for (MeshIndex c = 0; c < 3; ++c) {
            const MeshCell cell = mesh.cell(c);
            require(cell.id() == c && cell.node_count() == node_counts[c] &&
                        cell.edge_count() == edge_counts[c] && cell.face_count() == face_counts[c],
                    "Unique cell counts");
            for (MeshIndex k = 0; k < cell.node_count(); ++k)
                require(cell.node(k).id() == first_nodes[c] + k, "Sorted node IDs");
            MeshIndex count = 0;
            for (MeshIndex id : cell.edge_ids()) {
                const MeshEdge edge = mesh.edge(id);
                require(edge.node_id(0) >= first_nodes[c] &&
                            edge.node_id(1) < first_nodes[c] + node_counts[c],
                        "Cell edge membership");
                require(cell.edge(count++).id() == id, "Cell edge wrapper");
            }
            require(count == edge_counts[c], "No duplicate edges");
            for (MeshIndex k = 0; k < cell.face_count(); ++k)
                require(cell.face(k).id() == cell.face_id(k) && cell.face_orientation(k) == 1,
                        "Cell face and sign");
        }
    });
    add("entity cell face signs remain relative to each cell", [] {
        Mesh mesh(adjacent_tetrahedra());
        require(mesh.cell(0).face(0).id() == mesh.cell(1).face(0).id(), "Shared face identity");
        require(mesh.cell(0).face_orientation(0) == 1 && mesh.cell(1).face_orientation(0) == -1,
                "Relative signs");
        require(mesh.cell(0).face(0).node_id(1) == mesh.cell(1).face(0).node_id(1),
                "Stored face order unchanged");
    });
    add("entity 1D points and cell edge IDs independent of cell order", [] {
        MeshData d = line_mesh();
        d.face_nodes.indices = {1, 0, 2, 3};
        d.cell_faces.indices = {2, 3, 1, 2, 0, 1};
        Mesh mesh(d);
        const MeshIndex expected_edges[] = {2, 1, 0};
        for (MeshIndex c = 0; c < 3; ++c) {
            const MeshCell cell = mesh.cell(c);
            require(cell.node_count() == 2 && cell.edge_count() == 1 && cell.face_count() == 2,
                    "1D cell counts");
            require(cell.edge(0).id() == expected_edges[c], "1D edge lookup");
            require(cell.node_id(0) == cell.edge(0).node_id(0) && cell.node_id(1) == cell.edge(0).node_id(1),
                    "1D node membership");
            const MeshFace face = cell.face(0);
            require(face.node_count() == 1 && face.edge_count() == 0, "Point face counts");
            for (MeshIndex id : face.edge_ids()) {
                (void)id;
                require(false, "Empty range iterated");
            }
        }
    });
    add("entity 2D cells and large polygons", [] {
        Mesh mesh(adjacent_polygons());
        require(mesh.cell(0).node_count() == 4 && mesh.cell(0).edge_count() == 4, "Quadrilateral");
        require(mesh.cell(1).node_count() == 3 && mesh.cell(1).edge_count() == 3, "Triangle");
        require(mesh.cell(0).face(0).edge(0).id() == mesh.cell(1).face(0).edge(0).id(), "Shared edge");
        Mesh large(polygon(100));
        require(large.cell(0).node_count() == 100 && large.cell(0).edge_count() == 100,
                "Unbounded polygon size");
        require(large.face(99).node(1).id() == 0, "Closing edge");
    });
    add("entity handles survive temporary view and handle copies", [] {
        Mesh mesh(tetrahedron());
        const MeshCell cell = mesh.view().cell(0);
        const MeshCell copied = cell;
        require(copied.face(2).node(2).z() == 1, "Temporary descriptor lifetime");
        Mesh other = mesh;
        mesh = Mesh(line_mesh());
        require(other.cell(0).edge_count() == 6 && other.cell(0).node(3).z() == 1,
                "Mesh copy owns its caches");
        Mesh moved = std::move(other);
        require(moved.cell(0).node_count() == 4 && moved.cell(0).edge_count() == 6, "Moved caches");
        // После замены сетки проверка сознательно не разыменовывает старые дескрипторы.
    });
    add("entity checked host IDs reject invalid and empty access", [] {
        const Mesh meshes[] = {Mesh(), Mesh(line_mesh()), Mesh(adjacent_polygons()), Mesh(tetrahedron())};
        for (const Mesh& mesh : meshes) {
            const MeshView v = mesh.view();
            rejects([&] { mesh.node(v.node_count); }, "Node index");
            rejects([&] { mesh.edge(v.edge_count); }, "Edge index");
            rejects([&] { mesh.face(v.face_count); }, "Face index");
            rejects([&] { mesh.cell(v.cell_count); }, "Cell index");
            rejects([&] { mesh.cell(std::numeric_limits<MeshIndex>::max()); }, "Cell index");
        }
    });
    add("entity connectivity rebuilt by text and binary readers", [] {
        const MeshData cases[] = {line_mesh(), adjacent_polygons(), mixed(), polygonal_prism(40)};
        for (MeshFormat format : {MeshFormat::Text, MeshFormat::Binary}) {
            for (const MeshData& d : cases) {
                Mesh mesh(d);
                const Mesh loaded = decode(encode(mesh, format));
                for (MeshIndex c = 0; c < mesh.view().cell_count; ++c) {
                    const MeshCell a = mesh.cell(c), b = loaded.cell(c);
                    require(a.node_count() == b.node_count() && a.edge_count() == b.edge_count(),
                            "Derived CSR counts");
                    for (MeshIndex k = 0; k < a.node_count(); ++k)
                        require(a.node_id(k) == b.node_id(k), "Derived nodes");
                    for (MeshIndex k = 0; k < a.edge_count(); ++k)
                        require(a.edge_id(k) == b.edge_id(k), "Derived edges");
                }
            }
        }
    });
}

// Топология, ориентированные связи и ошибки сеток в одном и двух измерениях.
void register_lower_dimensions() {
    add("1D nonuniform cells and point boundaries", [] {
        Mesh mesh(line_mesh());
        const MeshView v = mesh.view();
        require(v.dimension == 1 && v.cell_count == 3 && v.face_count == 4 && v.edge_count == 3, "1D counts");
        require(v.x[1] - v.x[0] == 0.25 && v.x[2] - v.x[1] == 1.75 && v.x[3] - v.x[2] == 4,
                "Nonuniform spacing");
        require(mesh.edge_begin() == std::vector<MeshIndex>({0, 1, 2}) &&
                    mesh.edge_end() == std::vector<MeshIndex>({1, 2, 3}),
                "1D edges are cells");
        for (MeshIndex f = 0; f < v.face_count; ++f) {
            require(v.face_nodes(f).size == 1 && v.face_nodes(f)[0] == f, "Point face");
            require(v.face_edges(f).size == 0 && v.face_edges(f).data == nullptr, "Point has no edges");
        }
        require(v.boundary_face_count == 2 && v.boundary_faces[0] == 0 && v.boundary_faces[1] == 3,
                "Endpoints");
    });
    add("1D unordered cells, nodes and endpoint references", [] {
        MeshData d = line_mesh();
        std::swap(d.x[0], d.x[3]);
        for (MeshIndex& node : d.face_nodes.indices) {
            if (node == 0)
                node = 3;
            else if (node == 3)
                node = 0;
        }
        std::swap(d.cell_faces.indices[0], d.cell_faces.indices[1]);
        std::swap(d.cell_face_orientation[0], d.cell_face_orientation[1]);
        Mesh mesh(d);
        mesh.validate();
    });
    add("2D arbitrary polygon sizes and one edge per face", [] {
        for (std::size_t n : {std::size_t(3), std::size_t(4), std::size_t(5), std::size_t(100)}) {
            Mesh mesh(polygon(n));
            const MeshView v = mesh.view();
            require(v.dimension == 2 && v.cell_count == 1 && v.edge_count == n, "Polygon counts");
            for (MeshIndex f = 0; f < v.face_count; ++f) {
                const MeshIndexSpan nodes = v.face_nodes(f), edges = v.face_edges(f);
                require(nodes.size == 2 && edges.size == 1, "Face CSR lengths");
                require(v.edge_begin[edges[0]] == std::min(nodes[0], nodes[1]) &&
                            v.edge_end[edges[0]] == std::max(nodes[0], nodes[1]),
                        "Face-edge mapping");
            }
        }
    });
    add("2D concave polygon", [] {
        MeshData d = polygon(5);
        d.x = {0, 2, 2, 1, 0};
        d.y = {0, 0, 2, 1, 2};
        Mesh mesh(d);
        require(mesh.view().cell_faces(0).size == 5, "Concave cell");
    });
    add("2D mixed cells with shared oriented edge zero", [] {
        Mesh mesh(adjacent_polygons());
        const MeshView v = mesh.view();
        require(v.cell_count == 2 && v.edge_count == 6 && v.boundary_face_count == 5, "Shared edge counts");
        require(v.face_orientation(0, 0) == 1 && v.face_orientation(1, 0) == -1, "Opposite shared signs");
        MeshData d = mesh.data();
        std::swap(d.face_nodes.indices[0], d.face_nodes.indices[1]);
        d.cell_face_orientation[0] = -1;
        d.cell_face_orientation[4] = 1;
        Mesh reversed(d);
        require(reversed.edge_begin() == mesh.edge_begin() && reversed.edge_end() == mesh.edge_end(),
                "Stable edges");
    });
    add("2D cell contour independent of face reference order", [] {
        MeshData d = polygon(5);
        std::reverse(d.cell_faces.indices.begin(), d.cell_faces.indices.end());
        Mesh mesh(d);
        mesh.validate();
    });
    add("empty meshes in all dimensions", [] {
        for (unsigned dim = 1; dim <= 3; ++dim) {
            MeshData d;
            d.dimension = dim;
            Mesh mesh(d);
            require(mesh.view().dimension == dim && mesh.view().edge_count == 0, "Empty dimension");
            for (MeshFormat f : {MeshFormat::Text, MeshFormat::Binary})
                require(same(d, decode(encode(mesh, f)).data()), "Empty dimension round trip");
        }
    });
    const auto invalid_dimension_mesh = [](const std::string& name, MeshData d,
                                           std::function<void(MeshData&)> mutate) {
        add(name, [d, mutate]() mutable {
            mutate(d);
            rejects([&] { Mesh mesh(d); });
        });
    };
    invalid_dimension_mesh("reject dimension zero", line_mesh(), [](MeshData& d) { d.dimension = 0; });
    invalid_dimension_mesh("reject dimension four", polygon(3), [](MeshData& d) { d.dimension = 4; });
    invalid_dimension_mesh("1D wrong point arity", line_mesh(),
                           [](MeshData& d) { d.face_nodes.offsets[1] = 2; });
    invalid_dimension_mesh("1D duplicate point face", line_mesh(),
                           [](MeshData& d) { d.face_nodes.indices[0] = 1; });
    invalid_dimension_mesh("1D cell must have two faces", line_mesh(),
                           [](MeshData& d) { d.cell_faces.offsets[1] = 3; });
    invalid_dimension_mesh("1D equal endpoint signs", line_mesh(),
                           [](MeshData& d) { d.cell_face_orientation[0] = 1; });
    invalid_dimension_mesh("1D repeated endpoint", line_mesh(),
                           [](MeshData& d) { d.cell_faces.indices[0] = 1; });
    invalid_dimension_mesh("1D missing boundary point", line_mesh(), [](MeshData& d) {
        d.boundary_faces.pop_back();
        d.boundary_patch.pop_back();
    });
    invalid_dimension_mesh("1D interior point marked boundary", line_mesh(), [](MeshData& d) {
        d.boundary_faces.push_back(1);
        d.boundary_patch.push_back(0);
    });
    invalid_dimension_mesh("2D wrong edge arity", polygon(4),
                           [](MeshData& d) { d.face_nodes.offsets[1] = 3; });
    invalid_dimension_mesh("2D repeated edge vertex", polygon(4),
                           [](MeshData& d) { d.face_nodes.indices[0] = 1; });
    invalid_dimension_mesh("2D duplicate reversed edge", polygon(4),
                           [](MeshData& d) { append_face(d, {1, 0}); });
    invalid_dimension_mesh("2D inconsistent contour orientation", polygon(4),
                           [](MeshData& d) { d.cell_face_orientation[0] = -1; });
    invalid_dimension_mesh("2D open contour", polygon(4), [](MeshData& d) {
        d.x.push_back(3);
        d.y.push_back(3);
        d.z.push_back(0);
        d.face_nodes.indices.back() = 4;
    });
    invalid_dimension_mesh("2D disconnected contours", polygon(6),
                           [](MeshData& d) { d.face_nodes.indices = {0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3}; });
    invalid_dimension_mesh("2D pinched contour", polygon(6),
                           [](MeshData& d) { d.face_nodes.indices = {0, 1, 1, 2, 2, 0, 0, 4, 4, 5, 5, 0}; });
    invalid_dimension_mesh("2D shared edge wrong sign", adjacent_polygons(),
                           [](MeshData& d) { d.cell_face_orientation[4] = 1; });
    for (MeshFormat format : {MeshFormat::Text, MeshFormat::Binary}) {
        const std::string prefix = format == MeshFormat::Text ? "text: " : "binary: ";
        add(prefix + "1D and 2D round trips including files", [format] {
            const MeshData cases[] = {line_mesh(), polygon(3), polygon(100), adjacent_polygons()};
            for (const MeshData& d : cases) {
                Mesh mesh(d);
                const Mesh loaded = decode(encode(mesh, format));
                require(same(d, loaded.data()), "Dimension/data round trip");
                require(mesh.edge_begin() == loaded.edge_begin() && mesh.edge_end() == loaded.edge_end(),
                        "Derived edges");
                for (MeshIndex f = 0; f < mesh.view().face_count; ++f) {
                    const MeshIndexSpan a = mesh.view().face_edges(f), b = loaded.view().face_edges(f);
                    require(a.size == b.size, "Face edge size");
                    for (MeshIndex k = 0; k < a.size; ++k)
                        require(a[k] == b[k], "Face edge ID");
                }
                TempFile file;
                mesh.write(file.path, format);
                require(same(d, Mesh::read(file.path).data()), "File round trip");
            }
        });
        add(prefix + "reject lower-dimensional truncated inputs", [format] {
            const MeshData cases[] = {line_mesh(), adjacent_polygons()};
            for (const MeshData& d : cases) {
                const std::string bytes = encode(Mesh(d), format);
                const std::size_t required =
                    format == MeshFormat::Text ? bytes.rfind("END") + 3 : bytes.size();
                for (std::size_t n = 0; n < required; ++n)
                    rejects([&] { decode(bytes.substr(0, n)); });
            }
        });
        add(prefix + "reject invalid encoded dimensions", [format] {
            const std::string good = encode(Mesh(line_mesh()), format);
            for (unsigned dim : {0u, 4u, 255u}) {
                std::string bad = good;
                if (format == MeshFormat::Text)
                    bad = replace_once(bad, "DIMENSION\n1", "DIMENSION\n" + std::to_string(dim));
                else
                    bad[16] = static_cast<char>(dim);
                rejects([&] { decode(bad); }, "dimension");
            }
        });
    }
}

// Ориентация граней, произвольные многогранники и сохранение сетки.
void register_polyhedra() {
    add("empty mesh", [] {
        Mesh mesh;
        mesh.validate();
        const MeshView v = mesh.view();
        require(v.node_count == 0 && v.face_count == 0 && v.cell_count == 0 && v.edge_count == 0,
                "Empty view");
        for (MeshFormat f : {MeshFormat::Text, MeshFormat::Binary})
            require(same(mesh.data(), decode(encode(mesh, f)).data()), "Empty round trip");
    });
    add("tetrahedron, hexahedron and prism through faces", [] {
        Mesh mesh(mixed());
        const MeshView v = mesh.view();
        require(v.node_count == 18 && v.face_count == 15 && v.cell_count == 3, "Counts");
        require(v.cell_faces(0).size == 4 && v.cell_faces(1).size == 6 && v.cell_faces(2).size == 5,
                "Cell faces");
        require(v.face_nodes(4).size == 4 && v.face_nodes(4)[1] == 7, "Face cycle");
        require(v.edge_count == 27 && v.x == mesh.data().x.data(), "Edges and borrowed storage");
    });
    add("shared face ID zero has opposite orientations", [] {
        Mesh mesh(adjacent_tetrahedra());
        const MeshView v = mesh.view();
        require(v.face_count == 7 && v.edge_count == 9 && v.boundary_face_count == 6, "Shared topology");
        require(v.cell_faces(0)[0] == 0 && v.cell_faces(1)[0] == 0, "Shared ID");
        require(v.face_orientation(0, 0) == 1 && v.face_orientation(1, 0) == -1, "Shared signs");
    });
    add("arbitrary polygon and polyhedron sizes", [] {
        for (std::size_t n : {std::size_t(5), std::size_t(40)}) {
            Mesh mesh(polygonal_prism(n));
            const MeshView v = mesh.view();
            require(v.node_count == 2 * n && v.face_count == n + 2 && v.edge_count == 3 * n, "Prism counts");
            require(v.face_nodes(0).size == n && v.cell_faces(0).size == n + 2, "No fixed limits");
        }
        Mesh ico(icosahedron());
        require(ico.view().face_count == 20 && ico.view().edge_count == 30, "Icosahedron");
    });
    add("canonical edge endpoints and face-edge mapping", [] {
        Mesh mesh(tetrahedron());
        require(mesh.edge_begin() == std::vector<MeshIndex>({0, 0, 0, 1, 1, 2}), "Edge starts");
        require(mesh.edge_end() == std::vector<MeshIndex>({1, 2, 3, 2, 3, 3}), "Edge ends");
        const MeshIndexSpan edges = mesh.view().face_edges(0);
        require(edges.size == 3 && edges[0] == 1 && edges[1] == 3 && edges[2] == 0, "Face edge order");
    });
    add("face rotation and reversal with matching signs", [] {
        MeshData d = adjacent_tetrahedra();
        std::rotate(d.face_nodes.indices.begin(), d.face_nodes.indices.begin() + 1,
                    d.face_nodes.indices.begin() + 3);
        Mesh rotated(d);
        std::reverse(d.face_nodes.indices.begin(), d.face_nodes.indices.begin() + 3);
        d.cell_face_orientation[0] = -1;
        d.cell_face_orientation[4] = 1;
        Mesh reversed(d);
        require(rotated.edge_begin() == reversed.edge_begin() && rotated.edge_end() == reversed.edge_end(),
                "Stable edge IDs");
    });
    add("cell faces need no special local ordering", [] {
        MeshData d = tetrahedron();
        std::reverse(d.cell_faces.indices.begin(), d.cell_faces.indices.end());
        Mesh mesh(d);
        mesh.validate();
    });
    add("copy and move retain independent valid storage", [] {
        Mesh original(mixed());
        Mesh copy(original);
        require(copy.view().x != original.view().x && same(copy.data(), original.data()), "Copy");
        Mesh moved(std::move(copy));
        require(same(moved.data(), original.data()) && moved.edge_end() == original.edge_end(), "Move");
        moved.validate();
    });
    for (MeshFormat format : {MeshFormat::Text, MeshFormat::Binary}) {
        const std::string p = format == MeshFormat::Text ? "text: " : "binary: ";
        add(p + "mixed and arbitrary cells round trip", [format] {
            const MeshData cases[] = {mixed(), adjacent_tetrahedra(), polygonal_prism(5),
                                      polygonal_prism(40)};
            for (const MeshData& d : cases) {
                Mesh original(d);
                const Mesh loaded = decode(encode(original, format));
                require(same(d, loaded.data()), "Changed data");
                require(original.edge_begin() == loaded.edge_begin() &&
                            original.edge_end() == loaded.edge_end(),
                        "Changed edges");
                require(encode(original, format) == encode(loaded, format), "Unstable serialization");
            }
        });
        add(p + "floating point and UTF-8 round trip", [format] {
            MeshData d = tetrahedron();
            d.x[0] = std::nextafter(0.1, 1.0);
            d.y[0] = -0.0;
            d.z[0] = std::numeric_limits<double>::denorm_min();
            d.length_scale_to_meters = 0.001;
            d.patch_names[0] = u8"крыло \"левое\" \\ wing";
            Mesh mesh(d);
            const Mesh loaded = decode(encode(mesh, format));
            require(same(d, loaded.data()) && std::signbit(loaded.data().y[0]),
                    "Exact numeric/name round trip");
        });
        add(p + "reject every truncated payload prefix", [format] {
            const std::string bytes = encode(Mesh(tetrahedron()), format);
            const std::size_t required = format == MeshFormat::Text ? bytes.rfind("END") + 3 : bytes.size();
            for (std::size_t n = 0; n < required; ++n)
                rejects([&] { decode(bytes.substr(0, n)); });
        });
        add(p + "reject trailing data",
            [format] { rejects([&] { decode(encode(Mesh(tetrahedron()), format) + "X"); }, "Trailing"); });
        add(p + "allocation and name limits", [format] {
            const std::string bytes = encode(Mesh(tetrahedron()), format);
            rejects([&] { decode(bytes, MeshReadLimits(16, 4096)); }, "payload");
            rejects([&] { decode(bytes, MeshReadLimits(1024 * 1024, 3)); }, "name");
            require(same(tetrahedron(), decode(bytes, MeshReadLimits(1024 * 1024, 4)).data()),
                    "Exact name limit");
        });
    }
}

// Проверки повреждённых связей, оболочек и граничных граней.
void register_invalid_topology() {
    const auto invalid = [](const std::string& name, std::function<void(MeshData&)> mutate) {
        add(name, [mutate] {
            MeshData d = mixed();
            mutate(d);
            rejects([&] { Mesh mesh(d); });
        });
    };
    invalid("coordinate array sizes", [](MeshData& d) { d.y.pop_back(); });
    invalid("infinite coordinate", [](MeshData& d) { d.x[0] = std::numeric_limits<double>::infinity(); });
    invalid("NaN coordinate", [](MeshData& d) { d.z[0] = std::numeric_limits<double>::quiet_NaN(); });
    invalid("invalid length scale", [](MeshData& d) { d.length_scale_to_meters = 0; });
    invalid("missing face offsets", [](MeshData& d) { d.face_nodes.offsets.clear(); });
    invalid("missing cell offsets", [](MeshData& d) { d.cell_faces.offsets.clear(); });
    invalid("first CSR offset", [](MeshData& d) { d.face_nodes.offsets[0] = 1; });
    invalid("last CSR offset", [](MeshData& d) { ++d.cell_faces.offsets.back(); });
    invalid("decreasing offsets", [](MeshData& d) { d.face_nodes.offsets[1] = 7; });
    invalid("face node out of range", [](MeshData& d) { d.face_nodes.indices[0] = 100; });
    invalid("cell face out of range", [](MeshData& d) { d.cell_faces.indices[0] = 100; });
    invalid("face needs three vertices", [](MeshData& d) { d.face_nodes.offsets[1] = 2; });
    invalid("cell needs four faces", [](MeshData& d) { d.cell_faces.offsets[1] = 3; });
    invalid("repeated face vertex", [](MeshData& d) { d.face_nodes.indices[0] = 2; });
    invalid("repeated face in cell", [](MeshData& d) { d.cell_faces.indices[0] = 1; });
    invalid("missing orientations", [](MeshData& d) { d.cell_face_orientation.pop_back(); });
    invalid("zero orientation", [](MeshData& d) { d.cell_face_orientation[0] = 0; });
    invalid("invalid orientation magnitude", [](MeshData& d) { d.cell_face_orientation[0] = 2; });
    invalid("inconsistent oriented shell", [](MeshData& d) { d.cell_face_orientation[0] = -1; });
    invalid("boundary array sizes", [](MeshData& d) { d.boundary_patch.pop_back(); });
    invalid("boundary face out of range", [](MeshData& d) { d.boundary_faces[0] = 100; });
    invalid("boundary patch out of range", [](MeshData& d) { d.boundary_patch[0] = 100; });
    invalid("missing exterior boundary", [](MeshData& d) {
        d.boundary_faces.pop_back();
        d.boundary_patch.pop_back();
    });
    invalid("duplicate boundary face", [](MeshData& d) { d.boundary_faces[0] = 1; });
    invalid("duplicate face after rotation", [](MeshData& d) { append_face(d, {2, 1, 0}); });
    invalid("duplicate face after reversal", [](MeshData& d) { append_face(d, {1, 2, 0}); });
    invalid("orphan face", [](MeshData& d) { append_face(d, {0, 1, 17}); });
    invalid("empty patch name", [](MeshData& d) { d.patch_names[0].clear(); });
    invalid("duplicate patch name", [](MeshData& d) { d.patch_names[1] = d.patch_names[0]; });
    invalid("control character in name", [](MeshData& d) { d.patch_names[0] = "wing\n"; });
    invalid("invalid UTF-8", [](MeshData& d) { d.patch_names[0] = std::string("\xc0\x80", 2); });
    invalid("duplicate cell with different face ordering", [](MeshData& d) {
        d.cell_faces.indices.insert(d.cell_faces.indices.end(), {3, 2, 1, 0});
        d.cell_faces.offsets.push_back(19);
        d.cell_face_orientation.insert(d.cell_face_orientation.end(), 4, 1);
    });
    add("shared face signs must be opposite", [] {
        MeshData d = adjacent_tetrahedra();
        d.cell_face_orientation[4] = 1;
        rejects([&] { Mesh mesh(d); }, "opposite");
    });
    add("internal face cannot be boundary", [] {
        MeshData d = adjacent_tetrahedra();
        d.boundary_faces.push_back(0);
        d.boundary_patch.push_back(0);
        rejects([&] { Mesh mesh(d); }, "internal");
    });
    add("at most two cells per face", [] {
        MeshData d = adjacent_tetrahedra();
        d.cell_faces.indices.insert(d.cell_faces.indices.end(), {0, 1, 2, 4});
        d.cell_faces.offsets.push_back(12);
        d.cell_face_orientation.insert(d.cell_face_orientation.end(), 4, 1);
        rejects([&] { Mesh mesh(d); }, "more than two");
    });
    add("open cell shell", [] {
        MeshData d = polygonal_prism(4);
        d.face_nodes.offsets.pop_back();
        d.face_nodes.indices.resize(d.face_nodes.offsets.back());
        d.cell_faces.indices.pop_back();
        --d.cell_faces.offsets.back();
        d.cell_face_orientation.pop_back();
        d.boundary_faces.pop_back();
        d.boundary_patch.pop_back();
        rejects([&] { Mesh mesh(d); }, "open");
    });
    add("disconnected shells within one cell", [] {
        MeshData d = mixed();
        d.cell_faces.offsets = {0, 15};
        rejects([&] { Mesh mesh(d); }, "disconnected shells");
    });
    add("pinched vertex with otherwise closed shell", [] {
        MeshData d = icosahedron();
        for (MeshIndex& index : d.face_nodes.indices)
            if (index == 11)
                index = 0;
        rejects([&] { Mesh mesh(d); }, "at a vertex");
    });
}

// Независимые примеры формата, ошибки данных и файловых операций.
void register_mesh_format() {
    add("handwritten text fixture", [] {
        const std::string bytes = "AEROMESH 2 TEXT\nDIMENSION 3\nCOUNTS 4 4 12 1 4 4 1\n"
                                  "LENGTH_SCALE_TO_METERS 1\nX 0 1 0 0\nY 0 0 1 0\nZ 0 0 0 1\n"
                                  "FACE_OFFSETS 0 3 6 9 12\nFACE_NODES 0 2 1 0 1 3 1 2 3 2 0 3\n"
                                  "CELL_OFFSETS 0 4\nCELL_FACES 0 1 2 3\nCELL_ORIENTATIONS +1 1 1 1\n"
                                  "BOUNDARY_FACES 0 1 2 3\nBOUNDARY_PATCH 0 0 0 0\nPATCH_NAMES \"wing\"\nEND";
        require(same(tetrahedron(), decode(bytes).data()), "Independent text fixture");
    });
    add("independent empty binary fixture", [] {
        std::string bytes = "AEROMSHB";
        const auto u64 = [&](std::uint64_t n) {
            for (unsigned i = 0; i < 8; ++i)
                bytes.push_back(static_cast<char>((n >> (8 * i)) & 255));
        };
        u64(2);
        u64(3);
        for (unsigned i = 0; i < 7; ++i)
            u64(0);
        u64(0x3ff0000000000000ULL);
        u64(0);
        u64(0);
        require(bytes.size() == 104 && same(MeshData(), decode(bytes).data()), "Binary layout");
        require(bytes == encode(Mesh(), MeshFormat::Binary), "Unexpected empty serialization");
    });
    add("text rejects bad headers, tokens, connectivity and signs", [] {
        const std::string good = encode(Mesh(tetrahedron()), MeshFormat::Text);
        const std::vector<std::pair<std::string, std::string>> changes = {
            {"AEROMESH", "BADMAGIC"},
            {"2 TEXT", "1 TEXT"},
            {"2 TEXT", "3 TEXT"},
            {"TEXT", "ASCII"},
            {"DIMENSION\n3", "DIMENSION\n2"},
            {"COUNTS\n4", "COUNTS\n-4"},
            {"COUNTS\n4", "COUNTS\n18446744073709551616"},
            {"COUNTS\n4 4", "COUNTS\n4 18446744073709551615"},
            {"X\n0", "X\nwrong"},
            {"X\n0", "X\nnan"},
            {"X\n0", "X\n1e9999"},
            {"X\n0", "X\n0junk"},
            {"CELL_ORIENTATIONS\n1", "CELL_ORIENTATIONS\n0"},
            {"CELL_ORIENTATIONS\n1", "CELL_ORIENTATIONS\n-2"},
            {"FACE_NODES\n0", "FACE_NODES\n100"},
            {"FACE_NODES", "UNKNOWN_BLOCK"},
            {"\"wing\"", "wing"},
            {"\"wing\"", "\"wi\\qng\""},
            {"\"wing\"", "\"wing\"junk"}};
        for (const auto& change : changes)
            rejects([&] { decode(replace_once(good, change.first, change.second)); });
    });
    add("binary rejects version, overflow, sign and name corruption", [] {
        const std::string good = encode(Mesh(tetrahedron()), MeshFormat::Binary);
        std::string bytes = good;
        bytes[8] = 1;
        rejects([&] { decode(bytes); }, "version");
        bytes = good;
        std::fill(bytes.begin() + 24, bytes.begin() + 32, static_cast<char>(255));
        rejects([&] { decode(bytes); });
        // Заголовок занимает 88 байт, координаты 96, таблицы граней 40 + 96, таблицы ячеек 16 + 32.
        bytes = good;
        bytes[368] = 0;
        rejects([&] { decode(bytes); }, "orientation");
        bytes = good;
        std::fill(bytes.end() - 12, bytes.end() - 4, static_cast<char>(255));
        rejects([&] { decode(bytes); }, "name");
    });
    add("binary negative orientation encoding", [] {
        MeshData d = tetrahedron();
        std::reverse(d.face_nodes.indices.begin(), d.face_nodes.indices.begin() + 3);
        d.cell_face_orientation[0] = -1;
        const std::string bytes = encode(Mesh(d), MeshFormat::Binary);
        require(static_cast<unsigned char>(bytes[368]) == 255, "Negative sign must be 0xff");
        require(same(d, decode(bytes).data()), "Negative orientation round trip");
    });
    add("locale and formatting do not affect text I/O", [] {
        MeshData d = tetrahedron();
        d.x[0] = 0.125;
        Mesh mesh(d);
        std::ostringstream out;
        out.imbue(std::locale(std::locale::classic(), new CommaDecimal));
        out << std::hex << std::scientific << std::setw(30);
        const std::ios::fmtflags flags = out.flags();
        mesh.write(out, MeshFormat::Text);
        require(out.flags() == flags && out.width() == 30, "Changed caller stream formatting");
        std::istringstream in(out.str());
        in.imbue(out.getloc());
        require(same(d, Mesh::read(in).data()), "Locale-sensitive data");
    });
    add("failed streams and unsupported encoding", [] {
        Mesh mesh(tetrahedron());
        std::ostringstream out;
        rejects([&] { mesh.write(out, static_cast<MeshFormat>(99)); });
        out.setstate(std::ios::badbit);
        rejects([&] { mesh.write(out); });
        std::istringstream in;
        in.setstate(std::ios::badbit);
        rejects([&] { Mesh::read(in); });
    });
    add("filesystem round trip and overwrite", [] {
        Mesh mesh(polygonal_prism(5));
        for (MeshFormat f : {MeshFormat::Text, MeshFormat::Binary}) {
            TempFile file;
            mesh.write(file.path, f);
            require(same(mesh.data(), Mesh::read(file.path).data()), "File round trip");
            mesh.write(file.path, f);
            require(same(mesh.data(), Mesh::read(file.path).data()), "File overwrite");
        }
    });
    add("filesystem errors and invalid format preserve existing file", [] {
        TempFile file;
        Mesh mesh(tetrahedron());
        rejects([&] { Mesh::read(file.path); });
        rejects([&] { mesh.write(file.path + "/missing/out.mesh"); });
        {
            std::ofstream out(file.path.c_str(), std::ios::binary);
            out << "keep";
        }
        rejects([&] { mesh.write(file.path, static_cast<MeshFormat>(99)); });
        std::ifstream in(file.path.c_str(), std::ios::binary);
        std::string value;
        in >> value;
        require(value == "keep", "Invalid format truncated file");
        rejects([&] { Mesh::read(std::string("build/x\0ignored", 15)); }, "NUL");
    });
}

void register_tests() {
    register_compressed_mesh();
    register_entity_access();
    register_lower_dimensions();
    register_polyhedra();
    register_invalid_topology();
    register_mesh_format();
}
} // namespace

int main() {
    register_tests();
    std::size_t failed = 0;
    for (const Test& test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << tests.size() - failed << '/' << tests.size() << " tests passed\n";
    return failed == 0 ? 0 : 1;
}
