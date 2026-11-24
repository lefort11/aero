#include "tecplot.h"
#include "../../tests/test_checks.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace {
using namespace aero;
typedef std::int32_t I32;
using aero_test::require;
// Проверяет, что некорректный ввод завершается ожидаемой ошибкой.
template <class F> void rejects(F action) {
    try {
        action();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Expected rejection");
}
typedef std::pair<std::string, std::function<void()>> Test;
std::vector<Test> tests;
// Добавляет именованную проверку в список тестов модуля.
void add(const std::string& name, std::function<void()> action) {
    tests.push_back(std::make_pair(name, action));
}
// Добавляет грань в заготовку тестовой сетки Tecplot.
void face(MeshData& d, std::initializer_list<MeshIndex> nodes) {
    d.face_nodes.indices.insert(d.face_nodes.indices.end(), nodes.begin(), nodes.end());
    d.face_nodes.offsets.push_back(d.face_nodes.indices.size());
}
// Создаёт сетку из одной ячейки с заданной оболочкой.
void single(MeshData& d) {
    d.patch_names = {"wall"};
    const std::size_t n = d.face_nodes.offsets.size() - 1;
    for (std::size_t i = 0; i < n; ++i) {
        d.cell_faces.indices.push_back(i);
        d.boundary_faces.push_back(i);
    }
    d.cell_faces.offsets.push_back(n);
    d.cell_face_orientation.assign(n, 1);
    d.boundary_patch.assign(n, 0);
}
// Создаёт одномерную сетку для проверки алгоритма.
Mesh line(std::size_t n = 2) {
    MeshData d;
    d.dimension = 1;
    d.patch_names = {"ends"};
    for (std::size_t i = 0; i <= n; ++i) {
        const double x = static_cast<double>(i);
        d.x.push_back(x * x / 4);
        d.y.push_back(0);
        d.z.push_back(0);
        face(d, {i});
    }
    for (std::size_t i = 0; i < n; ++i) {
        d.cell_faces.indices.push_back(i);
        d.cell_faces.indices.push_back(i + 1);
        d.cell_faces.offsets.push_back(2 * (i + 1));
        d.cell_face_orientation.push_back(-1);
        d.cell_face_orientation.push_back(1);
    }
    d.boundary_faces = {0, n};
    d.boundary_patch = {0, 0};
    return Mesh(d);
}
// Создаёт многоугольную ячейку тестовой сетки.
Mesh polygon() {
    MeshData d;
    d.dimension = 2;
    d.x = {0, 2, 2, 1, 0};
    d.y = {0, 0, 2, 0.5, 2};
    d.z.assign(5, 0);
    for (MeshIndex i = 0; i < 5; ++i)
        face(d, {i, (i + 1) % 5});
    single(d);
    return Mesh(d);
}
// Создаёт полосу соседних ячеек для проверки общей топологии зоны.
Mesh strip(std::size_t n) {
    MeshData d;
    d.dimension = 2;
    d.patch_names = {"wall"};
    for (std::size_t i = 0; i <= n; ++i) {
        d.x.push_back(static_cast<double>(i));
        d.x.push_back(static_cast<double>(i));
        d.y.push_back(0);
        d.y.push_back(1);
        d.z.push_back(0);
        d.z.push_back(0);
        face(d, {2 * i, 2 * i + 1});
    }
    for (std::size_t i = 0; i < n; ++i) {
        face(d, {2 * i, 2 * i + 2});
        face(d, {2 * i + 3, 2 * i + 1});
        d.cell_faces.indices.insert(d.cell_faces.indices.end(), {n + 1 + 2 * i, i + 1, n + 2 + 2 * i, i});
        d.cell_faces.offsets.push_back(4 * (i + 1));
        d.cell_face_orientation.insert(d.cell_face_orientation.end(), {1, 1, 1, -1});
    }
    d.boundary_faces = {0, n};
    for (std::size_t i = n + 1; i < d.face_nodes.offsets.size() - 1; ++i)
        d.boundary_faces.push_back(i);
    d.boundary_patch.assign(d.boundary_faces.size(), 0);
    return Mesh(d);
}
// Создаёт соседние тетраэдры с общей гранью.
Mesh tetrahedra() {
    MeshData d;
    d.x = {0, 1, 0, 0, 0};
    d.y = {0, 0, 1, 0, 0};
    d.z = {0, 0, 0, 1, -1};
    face(d, {0, 2, 1});
    face(d, {0, 1, 3});
    face(d, {1, 2, 3});
    face(d, {2, 0, 3});
    face(d, {0, 2, 4});
    face(d, {2, 1, 4});
    face(d, {1, 0, 4});
    d.cell_faces = MeshCsr({0, 4, 8}, {0, 1, 2, 3, 0, 4, 5, 6});
    d.cell_face_orientation = {1, 1, 1, 1, -1, 1, 1, 1};
    d.boundary_faces = {1, 2, 3, 4, 5, 6};
    d.boundary_patch.assign(6, 0);
    d.patch_names = {"wall"};
    return Mesh(d);
}
// Создаёт призматическую сетку для проверки геометрии и связности.
Mesh prism(std::size_t n) {
    MeshData d;
    const double pi = std::acos(-1.0);
    for (unsigned layer = 0; layer < 2; ++layer)
        for (std::size_t i = 0; i < n; ++i) {
            const double angle = 2 * pi * static_cast<double>(i) / static_cast<double>(n);
            d.x.push_back(std::cos(angle));
            d.y.push_back(std::sin(angle));
            d.z.push_back(layer);
        }
    for (std::size_t i = 0; i < n; ++i)
        d.face_nodes.indices.push_back(n - i - 1);
    d.face_nodes.offsets.push_back(n);
    for (std::size_t i = 0; i < n; ++i)
        d.face_nodes.indices.push_back(n + i);
    d.face_nodes.offsets.push_back(2 * n);
    for (std::size_t i = 0; i < n; ++i)
        face(d, {i, (i + 1) % n, (i + 1) % n + n, i + n});
    single(d);
    return Mesh(d);
}
// Создаёт тестовые поля с разными размещениями и числом компонент.
FieldSnapshot fields(const Mesh& mesh, bool all = true) {
    FieldSnapshot result(mesh, 123, 0.25);
    for (FieldLocation loc :
         {FieldLocation::Node, FieldLocation::Cell, FieldLocation::Edge, FieldLocation::Face}) {
        if (!all && (loc == FieldLocation::Edge || loc == FieldLocation::Face))
            continue;
        Field f;
        f.name = "velocity";
        f.location = loc;
        f.components = {"x", "y"};
        f.units = "m/s";
        for (unsigned k = 0; k < 2; ++k)
            for (MeshIndex i = 0; i < result.entity_count(loc); ++i)
                f.values.push_back(100 * k + static_cast<double>(i) + 0.5);
        result.fields.push_back(f);
    }
    return result;
}
// Экспортирует тестовую сетку в строку двоичного Tecplot.
std::string export_bytes(const Mesh& mesh, const FieldSnapshot& data, int threads = 1) {
    std::ostringstream out(std::ios::binary);
    tecplot::Options options;
    options.threads = threads;
    tecplot::write(out, mesh, data, options);
    return out.str();
}

// Независимый читатель теста следует описанию PLT v112. Проверяем типы, длины, диапазоны, конец файла и связность.
struct DecodedZone {
    std::string name;
    I32 type, nodes, cells, faces, total_face_nodes;
    double time;
    std::vector<I32> locations, passive, connectivity, offsets, left, right;
    std::map<std::string, std::string> auxiliary;
    std::vector<double> data;
    // В зоне узловая и ячеечная переменные имеют разную длину. Их начала
    // указывают на срезы единого массива, пассивная переменная даёт пустой срез.
    std::vector<std::size_t> starts;
    // Получает значение переменной с учётом перенумерации узлов зоны.
    double value(std::size_t variable,std::size_t i) const { return data[starts[variable]+i]; }
};
// Декодирует двоичный Tecplot в независимые тестовые структуры.
class PltReader {
public:
    std::vector<std::string> variables;
    std::vector<DecodedZone> zones;
    std::map<std::pair<I32, std::string>, std::string> variable_aux;
    // Начинает независимое тестовое декодирование двоичного Tecplot.
    explicit PltReader(const std::string& bytes) : bytes_(bytes), pos_(0) {
        require(bytes.substr(0, 8) == "#!TDV112", "PLT signature");
        pos_ = 8;
        require(integer() == 1 && integer() == 0, "Endian/full-file flags");
        string();
        const I32 nv = integer();
        require(nv >= 3 && nv < 100000, "Variable count");
        for (I32 i = 0; i < nv; ++i)
            variables.push_back(string());
        float marker = real32();
        while (marker != 357) {
            if (marker == 899) {
                const I32 index = integer();
                const std::string key = string();
                require(integer() == 0, "Aux type");
                variable_aux[std::make_pair(index, key)] = string();
                marker = real32();
                continue;
            }
            require(marker == 299, "Zone marker");
            DecodedZone z;
            z.name = string();
            integer();
            integer();
            z.time = real64();
            integer();
            z.type = integer();
            require(z.type == 0 || z.type == 1 || z.type == 6 || z.type == 7, "Supported zone type");
            const I32 specified = integer();
            require(specified == 0 || specified == 1, "Location flag");
            for (I32 i = 0; i < nv; ++i)
                z.locations.push_back(specified ? integer() : 0);
            require(integer() == 0 && integer() == 0, "No face-neighbor records");
            z.nodes = integer();
            z.faces = 0;
            z.total_face_nodes = 0;
            if (z.type == 0) {
                require(integer() == 1 && integer() == 1, "Point cloud dimensions");
                z.cells = 0;
            } else {
                if (z.type >= 6) {
                    z.faces = integer();
                    z.total_face_nodes = integer();
                    require(integer() == 0 && integer() == 0, "No interzone boundaries");
                }
                z.cells = integer();
                require(integer() == 0 && integer() == 0 && integer() == 0, "FE future dimensions");
            }
            while (integer()) {
                const std::string key = string();
                require(integer() == 0, "Zone aux type");
                z.auxiliary[key] = string();
            }
            require(z.nodes > 0 && z.cells >= 0, "Nonempty zone");
            zones.push_back(z);
            marker = real32();
        }
        for (DecodedZone& z : zones) {
            require(real32() == 299, "Data zone marker");
            for (I32 i = 0; i < nv; ++i)
                require(integer() == 2, "Float64 variable");
            const I32 has_passive = integer();
            for (I32 i = 0; i < nv; ++i)
                z.passive.push_back(has_passive ? integer() : 0);
            require(integer() == 0 && integer() == -1, "No sharing");
            std::vector<std::pair<double, double>> ranges(static_cast<std::size_t>(nv));
            for (I32 i = 0; i < nv; ++i)
                if (!z.passive[static_cast<std::size_t>(i)]) {
                    const double lo = real64(), hi = real64();
                    ranges[static_cast<std::size_t>(i)] = std::make_pair(lo, hi);
                }
            z.starts.reserve(static_cast<std::size_t>(nv)+1);
            for (I32 i = 0; i < nv; ++i) {
                const std::size_t k = static_cast<std::size_t>(i);
                z.starts.push_back(z.data.size());
                if (z.passive[k])
                    continue;
                const I32 count = z.locations[k] ? z.cells : z.nodes;
                for (I32 j = 0; j < count; ++j)
                    z.data.push_back(real64());
                const auto mm = std::minmax_element(z.data.data()+z.starts.back(),z.data.data()+z.data.size());
                require(*mm.first == ranges[k].first && *mm.second == ranges[k].second,
                        "Exact zone variable min/max");
            }
            z.starts.push_back(z.data.size());
            if (z.type == 1)
                for (I32 i = 0; i < 2 * z.cells; ++i)
                    z.connectivity.push_back(integer());
            if (z.type >= 6) {
                if (z.type == 7)
                    for (I32 i = 0; i <= z.faces; ++i)
                        z.offsets.push_back(integer());
                for (I32 i = 0; i < z.total_face_nodes; ++i)
                    z.connectivity.push_back(integer());
                for (I32 i = 0; i < z.faces; ++i)
                    z.left.push_back(integer());
                for (I32 i = 0; i < z.faces; ++i)
                    z.right.push_back(integer());
                if (z.type == 7)
                    require(z.offsets.front() == 0 && z.offsets.back() == z.total_face_nodes,
                            "Polyhedron offsets");
                else
                    require(z.total_face_nodes == z.faces * 2, "Polygon edge count");
                for (I32 i = 0; i < z.faces; ++i) {
                    const I32 l = z.left[static_cast<std::size_t>(i)],
                              r = z.right[static_cast<std::size_t>(i)];
                    require(l >= -1 && l < z.cells && r >= -1 && r < z.cells && l != r, "Face neighbors");
                }
            }
            for (I32 n : z.connectivity)
                require(n >= 0 && n < z.nodes, "Zero-based connectivity in bounds");
        }
        require(pos_ == bytes_.size(), "Exact PLT EOF");
    }

private:
    // Преобразует числовое значение в его двоичное представление.
    std::uint64_t bits(unsigned size) {
        require(size <= bytes_.size() - pos_, "Truncated PLT");
        std::uint64_t result = 0;
        for (unsigned i = 0; i < size; ++i)
            result |= std::uint64_t(static_cast<unsigned char>(bytes_[pos_++])) << (8 * i);
        return result;
    }
    // Декодирует 32-битное целое из тестового потока Tecplot.
    I32 integer() {
        const std::uint32_t b = static_cast<std::uint32_t>(bits(4));
        I32 value;
        std::memcpy(&value, &b, 4);
        return value;
    }
    // Декодирует вещественный маркер одинарной точности Tecplot.
    float real32() {
        const std::uint32_t b = static_cast<std::uint32_t>(bits(4));
        float value;
        std::memcpy(&value, &b, 4);
        return value;
    }
    // Декодирует значение двойной точности Tecplot.
    double real64() {
        const std::uint64_t b = bits(8);
        double value;
        std::memcpy(&value, &b, 8);
        return value;
    }
    // Записывает строку в представлении кодов символов Tecplot.
    std::string string() {
        std::string result;
        for (I32 c = integer(); c; c = integer()) {
            require(c > 0 && c < 256, "PLT string byte");
            result += static_cast<char>(c);
        }
        return result;
    }
    const std::string& bytes_;
    std::size_t pos_;
};
// Владеет временным файлом проверки и удаляет его при выходе из области видимости.
struct Temp {
    std::string path;
    // Создаёт уникальное имя временного файла теста.
    Temp() {
        static unsigned counter = 0;
        path = "build/tecplot-test-" +
               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + "-" +
               std::to_string(counter++);
        std::ifstream in(path.c_str());
        require(!in, "Test file exists");
    }
    // Удаляет временные файлы и каталоги, созданные объектом.
    ~Temp() { std::remove(path.c_str()); }
};
// Читает содержимое тестового файла для проверки результата.
std::string read_file(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}
// Проверяет запуск утилиты Tecplot через интерфейс командной строки.
int cli(const std::vector<std::string>& args, std::string* errors = nullptr) {
    std::vector<const char*> argv;
    for (const std::string& s : args)
        argv.push_back(s.c_str());
    std::ostringstream out, err;
    const int code = tecplot::run_cli(static_cast<int>(argv.size()), argv.data(), out, err);
    if (errors)
        *errors = err.str();
    return code;
}
// Сверяем поля по данным независимого читателя PLT, включая их расположение по зонам.
void check_fields(const Mesh& mesh, const FieldSnapshot& snapshot, const PltReader& reader) {
    require(reader.zones.size() == 3, "Main edge and face zones");
    const auto& z = reader.zones[0];
    require(z.time == snapshot.time && z.auxiliary.at("Aero.Step") == "123", "Time and step");
    require(z.auxiliary.at("Aero.MeshSHA256") == snapshot.mesh_sha256, "Fingerprint metadata");
    require(!z.auxiliary.count("Aero.CoordinateUnits"), "Raw coordinates have no declared physical units");
    std::size_t variable = 3;
    for (const Field& field : snapshot.fields)
        for (std::size_t component = 0; component < field.components.size(); ++component, ++variable) {
            const std::size_t count = static_cast<std::size_t>(snapshot.entity_count(field.location));
            const std::size_t zi = field.location == FieldLocation::Edge   ? 1
                                   : field.location == FieldLocation::Face ? 2
                                                                           : 0;
            const auto& zone = reader.zones[zi];
            require(zone.starts[variable+1]-zone.starts[variable] == count, "Full field values");
            require(std::equal(zone.data.data()+zone.starts[variable], zone.data.data()+zone.starts[variable+1],
                               field.values.begin() + static_cast<std::ptrdiff_t>(component * count)),
                    "Component-major fields");
            require(reader.variable_aux.at(std::make_pair(static_cast<I32>(variable), "Aero.Component")) ==
                        field.components[component],
                    "Component metadata");
            if (zi)
                require(reader.zones[0].passive[variable] == 1, "Inactive field not interpolated on volume");
        }
    for (std::size_t i = 0; i < mesh.data().x.size(); ++i)
        require(z.value(0,i) == mesh.data().x[i], "Coordinate units");
}
// Проверки топологии зон и значений полей с разным расположением.
void register_zones() {
    add("1D nonuniform topology and all field locations", [] {
        const Mesh mesh = line();
        const FieldSnapshot data = fields(mesh);
        const std::string bytes = export_bytes(mesh, data);
        const PltReader reader(bytes);
        check_fields(mesh, data, reader);
        require(reader.zones[0].connectivity == std::vector<I32>({0, 1, 1, 2}), "Line segments");
        require(reader.zones[2].type == 0 && reader.zones[2].nodes == 3, "Point faces");
    });
    add("2D concave polygon and shared face orientation", [] {
        for (const Mesh& mesh : {polygon(), strip(2)}) {
            const auto data = fields(mesh);
            const std::string bytes = export_bytes(mesh, data);
            const PltReader reader(bytes);
            check_fields(mesh, data, reader);
            const auto& z = reader.zones[0];
            require(z.type == 6 && z.cells == static_cast<I32>(mesh.view().cell_count), "Unsplit polygons");
            require(std::equal(z.connectivity.begin(), z.connectivity.end(),
                               mesh.data().face_nodes.indices.begin()),
                    "Original face nodes");
            if (mesh.view().cell_count == 2)
                require(z.left[1] == 0 && z.right[1] == 1, "Shared edge sides");
            require(reader.zones[2].type == 1, "Face segments");
        }
    });
    add("3D tetrahedra prisms hexahedra arbitrary polyhedra", [] {
        for (const Mesh& mesh : {tetrahedra(), prism(3), prism(4), prism(7)}) {
            const auto data = fields(mesh);
            const std::string bytes = export_bytes(mesh, data);
            const PltReader reader(bytes);
            check_fields(mesh, data, reader);
            const auto& z = reader.zones[0];
            require(z.type == 7, "Polyhedron zone");
            require(std::equal(z.offsets.begin(), z.offsets.end(), mesh.data().face_nodes.offsets.begin()),
                    "Original polyhedron CSR");
            for (MeshIndex c = 0; c < mesh.view().cell_count; ++c)
                for (MeshIndex i = 0; i < mesh.cell(c).face_count(); ++i) {
                    const std::size_t f = static_cast<std::size_t>(mesh.cell(c).face_id(i));
                    require((mesh.cell(c).face_orientation(i) > 0 ? z.left[f] : z.right[f]) ==
                                static_cast<I32>(c),
                            "Outward face belongs to left cell");
                }
            const auto& surface = reader.zones[2];
            require(surface.type == 6 && surface.cells == static_cast<I32>(mesh.view().face_count),
                    "Whole face polygons");
            for (std::size_t i = 0; i < mesh.data().face_nodes.indices.size(); ++i)
                require(surface.value(0,i) ==
                            mesh.data().x[static_cast<std::size_t>(mesh.data().face_nodes.indices[i])],
                        "Surface node remapping");
        }
    });
    add("geometry-only and scalar cell data", [] {
        const Mesh mesh = polygon();
        FieldSnapshot data(mesh);
        std::string bytes = export_bytes(mesh, data);
        PltReader empty(bytes);
        require(empty.variables.size() == 3 && empty.zones.size() == 1, "No invented fields");
        Field f;
        f.name = "p";
        f.components = {"value"};
        f.values = {101325};
        data.fields.push_back(f);
        bytes = export_bytes(mesh, data);
        PltReader scalar(bytes);
        require(scalar.zones[0].locations[3] == 1 && scalar.zones[0].value(3,0) == 101325, "Cell scalar");
    });
    add("coordinates long UTF-8 names and metadata", [] {
        MeshData d = polygon().data();
        const Mesh mesh(d);
        auto data = fields(mesh);
        data.fields[0].name = std::string(160, 'a') + u8"крыло";
        data.fields[1].name = data.fields[0].name;
        const std::string bytes = export_bytes(mesh, data);
        const PltReader reader(bytes);
        check_fields(mesh, data, reader);
        std::set<std::string> names;
        for (const auto& name : reader.variables) {
            require(name.size() <= 128, "Tecplot variable label size");
            require(names.insert(name).second, "Unique labels");
        }
        require(reader.variable_aux.at(std::make_pair(3, "Aero.Field")) == data.fields[0].name,
                "Full name retained");
    });
    add("OpenMP one and multiple threads produce identical output", [] {
        for (const Mesh& mesh : {line(3000), strip(2500), prism(2200)}) {
            const auto data = fields(mesh);
            const std::string serial = export_bytes(mesh, data, 1);
            if (tecplot::openmp_available())
                require(serial == export_bytes(mesh, data, 2), "Deterministic threaded PLT");
            else
                rejects([&] { export_bytes(mesh, data, 2); });
            PltReader reader(serial);
            check_fields(mesh, data, reader);
        }
    });
}

// Проверки экспорта через командную строку, обработки ошибок и защиты файлов.
void register_cli() {
    add("CLI accepts every mesh and fields encoding in all dimensions", [] {
        for (const Mesh& mesh : {line(), polygon(), prism(5)})
            for (MeshFormat mf : {MeshFormat::Text, MeshFormat::Binary, MeshFormat::Lzma})
                for (FieldFormat ff : {FieldFormat::Text, FieldFormat::Binary, FieldFormat::Lzma}) {
                    Temp input, results, output;
                    mesh.write(input.path, mf);
                    const auto data = fields(mesh);
                    data.write(results.path, ff);
                    std::string err;
                    require(
                        cli({"fields_to_tecplot", input.path, results.path, output.path, "--threads", "1"},
                            &err) == 0,
                        err);
                    const std::string bytes = read_file(output.path);
                    PltReader reader(bytes);
                    check_fields(mesh, data, reader);
                }
    });
    add("CLI mesh only exports original geometry in 1D, 2D and 3D", [] {
        for (const Mesh& mesh : {line(), polygon(), tetrahedra()}) {
            const std::string reference = export_bytes(mesh, fields(mesh));
            const PltReader with_fields(reference);
            const DecodedZone& expected = with_fields.zones.front();
            for (MeshFormat format : {MeshFormat::Text, MeshFormat::Binary, MeshFormat::Lzma}) {
                Temp input, output;
                mesh.write(input.path, format);
                std::vector<std::string> args = {"fields_to_tecplot", input.path, output.path};
                if (format == MeshFormat::Lzma)
                    args.insert(args.end(), {"--threads", "1"});
                std::string err;
                require(cli(args, &err) == 0, err);
                const std::string bytes = read_file(output.path);
                const PltReader geometry(bytes);
                require(geometry.variables == std::vector<std::string>({"X", "Y", "Z"}),
                        "Coordinate variables");
                require(geometry.zones.size() == 1, "Single geometry zone");
                const DecodedZone& z = geometry.zones.front();
                require(z.name == "Cells" && z.type == expected.type && z.cells == expected.cells &&
                            z.nodes == expected.nodes,
                        "Native cell zone");
                require(z.connectivity == expected.connectivity && z.offsets == expected.offsets &&
                            z.left == expected.left && z.right == expected.right,
                        "Same topology as field export");
                require(z.data.size()==expected.starts.at(3) &&
                            std::equal(z.starts.begin(),z.starts.end(),expected.starts.begin()) &&
                            std::equal(z.data.begin(), z.data.end(), expected.data.begin()),
                        "Same coordinates as field export");
                require(z.auxiliary.count("Aero.Step") == 0 &&
                            z.auxiliary.at("Aero.MeshSHA256") == expected.auxiliary.at("Aero.MeshSHA256"),
                        "Mesh metadata");
                require(cli(args) == 0 && read_file(output.path) == bytes,
                        "Repeated geometry export");
            }
        }
    });
    add("CLI help missing arguments and unknown options", [] {
        std::string errors;
        require(cli({"fields_to_tecplot", "--help"}, &errors) == 0 && errors.empty(), "Help succeeds");
        require(cli({"fields_to_tecplot", "-h"}) == 0, "Short help");
        for (const auto& args :
             std::vector<std::vector<std::string>>{{"fields_to_tecplot"},
                                                   {"fields_to_tecplot", "x"},
                                                   {"fields_to_tecplot", "x", "y", "z", "extra"},
                                                   {"fields_to_tecplot", "x", "y", "--threads"}})
            require(cli(args, &errors) == 1 && errors.find("Usage:") != std::string::npos, "Brief manual");
        for (const char* n : {"0", "-1", "1x", "2147483648", ""})
            require(cli({"fields_to_tecplot", "x", "y", "z", "--threads", n}) == 1, "Invalid threads");
        require(cli({"fields_to_tecplot", "x", "y", "z", "--unknown"}) == 1, "Unknown option");
        require(cli({"fields_to_tecplot", "x", "y", "z", "--threads"}) == 1, "Missing option value");
    });
    add("mesh mismatch invalid fields and empty meshes write nothing", [] {
        const Mesh mesh = polygon();
        const auto data = fields(mesh);
        MeshData changed = mesh.data();
        changed.x[0] += 0.1;
        Temp output;
        rejects([&] { tecplot::write(output.path, Mesh(changed), data); });
        require(read_file(output.path).empty(), "No output on mismatch");
        for (double value :
             {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            auto bad = data;
            bad.fields[0].values[0] = value;
            std::ostringstream out;
            rejects([&] { tecplot::write(out, mesh, bad); });
            require(out.str().empty(), "No invalid field output");
        }
        Mesh empty;
        rejects([&] { export_bytes(empty, FieldSnapshot(empty)); });
        tecplot::Options options;
        options.threads = -1;
        std::ostringstream out;
        rejects([&] { tecplot::write(out, mesh, data, options); });
        changed = mesh.data();
        changed.x[1] = std::numeric_limits<double>::max();
        const Mesh large(changed);
        check_fields(large, fields(large), PltReader(export_bytes(large, fields(large))));
    });
    add("existing outputs inputs and failed streams are protected", [] {
        const Mesh mesh = line();
        const auto data = fields(mesh);
        Temp output;
        {
            std::ofstream out(output.path.c_str());
            out << "keep";
        }
        tecplot::write(output.path, mesh, data);
        require(read_file(output.path) == export_bytes(mesh, data), "Existing output replaced");
        Temp input, results;
        mesh.write(input.path);
        data.write(results.path);
        const std::string before = read_file(input.path);
        require(cli({"fields_to_tecplot", input.path, results.path, input.path}) == 1, "Input alias refused");
        require(read_file(input.path) == before, "Input preserved");
        rejects([&] { tecplot::write(output.path + "/missing", mesh, data); });
        rejects([&] { tecplot::write(std::string("build/x\0bad", 11), mesh, data); });
        std::ostringstream failed;
        failed.setstate(std::ios::badbit);
        rejects([&] { tecplot::write(failed, mesh, data); });
    });
}

// Формирует набор проверок модуля и выполняемых тестовых сценариев.
void register_tests() {
    register_zones();
    register_cli();
}
}
// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main() {
    register_tests();
    std::size_t failed = 0;
    for (const auto& test : tests)
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& e) {
            ++failed;
            std::cerr << "FAIL " << test.first << ": " << e.what() << '\n';
        }
    std::cout << tests.size() - failed << '/' << tests.size()
              << " Tecplot tests passed (OpenMP=" << (aero::tecplot::openmp_available() ? "on" : "off")
              << ")\n";
    return failed ? 1 : 0;
}
