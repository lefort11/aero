#include "fields.h"
#include "../../tests/test_checks.h"
#include "sha256.h"
#include "../mesh/mesh.h"
#include "lzma_stream.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

namespace {
using namespace aero;
typedef std::uint64_t U64;
std::vector<std::pair<std::string, std::function<void()>>> tests;
using aero_test::require;
template <class F> void rejects(F action, const std::string& part = "") {
    try {
        action();
    } catch (const FieldError& error) {
        require(part.empty() || std::string(error.what()).find(part) != std::string::npos,
                "Unexpected error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected FieldError");
}
void add(const std::string& name, std::function<void()> test) {
    tests.push_back(std::make_pair(name, test));
}
Mesh mesh(unsigned dimension) {
    MeshData d;
    d.dimension = dimension;
    d.patch_names = {"wall"};
    if (dimension == 1) {
        d.x = {0, 0.25, 2};
        d.y = {0, 0, 0};
        d.z = {0, 0, 0};
        d.face_nodes = MeshCsr({0, 1, 2, 3}, {0, 1, 2});
        d.cell_faces = MeshCsr({0, 2, 4}, {0, 1, 1, 2});
        d.cell_face_orientation = {-1, 1, -1, 1};
        d.boundary_faces = {0, 2};
    } else if (dimension == 2) {
        d.x = {0, 2, 1, 0};
        d.y = {0, 0, 0.25, 1};
        d.z = {0, 0, 0, 0};
        d.face_nodes = MeshCsr({0, 2, 4, 6, 8}, {0, 1, 1, 2, 2, 3, 3, 0});
        d.cell_faces = MeshCsr({0, 4}, {0, 1, 2, 3});
        d.cell_face_orientation = {1, 1, 1, 1};
        d.boundary_faces = {0, 1, 2, 3};
    } else {
        d.x = {0, 1, 0, 0};
        d.y = {0, 0, 1, 0};
        d.z = {0, 0, 0, 1};
        d.face_nodes = MeshCsr({0, 3, 6, 9, 12}, {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3});
        d.cell_faces = MeshCsr({0, 4}, {0, 1, 2, 3});
        d.cell_face_orientation = {1, 1, 1, 1};
        d.boundary_faces = {0, 1, 2, 3};
    }
    d.boundary_patch.assign(d.boundary_faces.size(), 0);
    return Mesh(d);
}
FieldSnapshot sample(unsigned dim = 2) {
    FieldSnapshot s(mesh(dim), 100, 0.001);
    for (FieldLocation location :
         {FieldLocation::Node, FieldLocation::Edge, FieldLocation::Face, FieldLocation::Cell}) {
        Field f;
        f.name = u8"скорость \"A\"\\";
        f.location = location;
        f.components = {"x", "y", "z"};
        f.units = "m/s";
        for (U64 component = 0; component < 3; ++component)
            for (U64 i = 0; i < s.entity_count(location); ++i)
                f.values.push_back(static_cast<double>(component * 100 + i) + 0.25);
        s.fields.push_back(f);
    }
    return s;
}
// Сравниваем метаданные, порядок компонент и все значения снимка после записи и чтения.
bool same(const FieldSnapshot& a, const FieldSnapshot& b) {
    if (a.dimension != b.dimension || a.counts != b.counts || a.mesh_sha256 != b.mesh_sha256 ||
        a.step != b.step || a.time != b.time || a.fields.size() != b.fields.size())
        return false;
    for (std::size_t i = 0; i < a.fields.size(); ++i) {
        const Field& x = a.fields[i];
        const Field& y = b.fields[i];
        if (x.name != y.name || x.location != y.location || x.components != y.components ||
            x.units != y.units || x.values.size() != y.values.size())
            return false;
        for (std::size_t j = 0; j < x.values.size(); ++j) {
            if (std::isnan(x.values[j]) && std::isnan(y.values[j]))
                continue;
            if (std::memcmp(&x.values[j], &y.values[j], 8) != 0)
                return false;
        }
    }
    return true;
}
std::string encode(const FieldSnapshot& s, FieldFormat format) {
    std::ostringstream out(std::ios::binary);
    s.write(out, format);
    return out.str();
}
FieldSnapshot decode(const std::string& bytes, const FieldReadLimits& limits = FieldReadLimits()) {
    std::istringstream in(bytes, std::ios::binary);
    return FieldSnapshot::read(in, limits);
}
std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const std::size_t pos = text.find(from);
    require(pos != std::string::npos, "Missing test token: " + from);
    text.replace(pos, from.size(), to);
    return text;
}
struct TempFile {
    std::string path;
    TempFile() {
        static unsigned counter = 0;
        path = "build/output-test-" +
               std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + "-" +
               std::to_string(counter++) + ".fields";
        std::ifstream existing(path.c_str());
        require(!existing, "Temporary file already exists");
    }
    ~TempFile() { std::remove(path.c_str()); }
};
void u64(std::string& bytes, U64 n) {
    for (unsigned i = 0; i < 8; ++i)
        bytes += static_cast<char>((n >> (8 * i)) & 255);
}
void string(std::string& bytes, const std::string& text) {
    u64(bytes, text.size());
    bytes += text;
}
// Искажаем отдельное слово двоичного файла, сохраняя окружающие байты для проверки диагностики.
void set_u64(std::string& bytes, std::size_t offset, U64 n) {
    require(offset + 8 <= bytes.size(), "Bad test offset");
    for (unsigned i = 0; i < 8; ++i)
        bytes[offset + i] = static_cast<char>((n >> (8 * i)) & 255);
}
std::string packed(const std::string& bytes) {
    std::ostringstream out(std::ios::binary);
    io::LzmaOutputBuffer buffer(out);
    std::ostream stream(&buffer);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    buffer.finish();
    return out.str();
}
struct Comma : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};
struct NoSeek : std::stringbuf {
    explicit NoSeek(const std::string& text) : std::stringbuf(text, std::ios::in) {}
    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override {
        return pos_type(off_type(-1));
    }
    pos_type seekpos(pos_type, std::ios_base::openmode) override { return pos_type(off_type(-1)); }
};
struct ShortOutput : std::streambuf {
    std::size_t remaining;
    explicit ShortOutput(std::size_t n) : remaining(n) {}
    int_type overflow(int_type value) override {
        if (!remaining)
            return traits_type::eof();
        --remaining;
        return value;
    }
};
// Проверки подписи сетки и обнаружения несовместимых снимков.
void register_mesh_identity() {
    add("SHA-256 standard vectors and incremental input", [] {
        const std::pair<std::string, std::string> vectors[] = {
            {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
            {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
            {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
             "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
            {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}};
        for (const auto& v : vectors)
            for (bool single : {false, true}) {
                io_detail::Sha256Buffer buffer;
                std::ostream out(&buffer);
                if (single)
                    for (char c : v.first)
                        out.put(c);
                else
                    out.write(v.first.data(), static_cast<std::streamsize>(v.first.size()));
                require(buffer.hex_digest() == v.second && buffer.hex_digest() == v.second,
                        "SHA-256 known digest");
            }
    });
    add("mesh fingerprint independent of mesh file encoding", [] {
        for (unsigned dim : {1U, 2U, 3U}) {
            const Mesh original = mesh(dim);
            const FieldSnapshot s(original);
            for (MeshFormat f : {MeshFormat::Text, MeshFormat::Binary, MeshFormat::Lzma}) {
                std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
                original.write(bytes, f);
                const Mesh loaded = Mesh::read(bytes);
                s.check_mesh(loaded);
            }
        }
    });
    add("mesh mismatch includes geometry names topology and counts", [] {
        const Mesh original = mesh(2);
        const FieldSnapshot s(original);
        for (unsigned change = 0; change < 3; ++change) {
            MeshData d = original.data();
            if (change == 0)
                d.x[0] += 0.01;
            if (change == 1)
                d.patch_names[0] = "other";
            if (change == 2)
                std::swap(d.face_nodes.indices[0], d.face_nodes.indices[1]), d.cell_face_orientation[0] = -1;
            rejects([&] { s.check_mesh(Mesh(d)); }, "SHA-256");
        }
        rejects([&] { s.check_mesh(mesh(1)); }, "dimension/counts");
        FieldSnapshot changed = s;
        changed.counts[0]++;
        rejects([&] { changed.check_mesh(original); });
    });
}

// Один набор данных проходит через каждый из трёх форматов.
void register_formats() {
    for (FieldFormat format : {FieldFormat::Text, FieldFormat::Binary, FieldFormat::Lzma}) {
        const std::string prefix = format == FieldFormat::Text     ? "text: "
                                   : format == FieldFormat::Binary ? "binary: "
                                                                   : "lzma: ";
        add(prefix + "1D 2D 3D all locations and component ordering", [format] {
            for (unsigned dim : {1U, 2U, 3U}) {
                const FieldSnapshot s = sample(dim);
                const FieldSnapshot loaded = decode(encode(s, format));
                require(same(s, loaded), "Full snapshot round trip");
                loaded.check_mesh(mesh(dim));
                TempFile file;
                s.write(file.path, format);
                require(same(s, FieldSnapshot::read(file.path)), "File round trip");
                FieldSnapshot later = s;
                later.step++;
                later.write(file.path, format);
                require(FieldSnapshot::read(file.path).step == 101, "File overwrite");
                NoSeek buffer(encode(s, format));
                std::istream input(&buffer);
                require(same(s, FieldSnapshot::read(input)), "Forward-only stream");
            }
        });
        add(prefix + "empty fields and empty meshes", [format] {
            for (unsigned dim : {1U, 2U, 3U}) {
                MeshData d;
                d.dimension = dim;
                FieldSnapshot s{Mesh(d)};
                require(same(s, decode(encode(s, format))), "Empty snapshot");
                Field f;
                f.name = "empty";
                f.components = {"value"};
                s.fields.push_back(f);
                require(same(s, decode(encode(s, format))), "Zero-entity field");
            }
        });
        add(prefix + "exact numbers special values and locale independence", [format] {
            FieldSnapshot s = sample();
            s.step = std::numeric_limits<U64>::max();
            s.fields[0].units = "";
            std::vector<double>& v = s.fields[0].values;
            v[0] = -0.0;
            v[1] = std::numeric_limits<double>::denorm_min();
            v[2] = std::numeric_limits<double>::max();
            v[3] = std::numeric_limits<double>::quiet_NaN();
            v[4] = std::numeric_limits<double>::infinity();
            v[5] = -v[4];
            std::ostringstream out(std::ios::binary);
            out.imbue(std::locale(std::locale::classic(), new Comma));
            out << std::hex << std::scientific << std::setprecision(2) << std::setw(30);
            const auto flags = out.flags();
            s.write(out, format);
            require(out.flags() == flags && out.precision() == 2 && out.width() == 30,
                    "Caller formatting preserved");
            require(same(s, decode(out.str())), "Exact field values");
        });
        add(prefix + "every truncated prefix and trailing bytes", [format] {
            FieldSnapshot s = sample();
            s.fields.resize(1);
            const std::string good = encode(s, format);
            const std::size_t required = format == FieldFormat::Text ? good.rfind("END") + 3 : good.size();
            for (std::size_t i = 0; i < required; ++i)
                rejects([&] { decode(good.substr(0, i)); });
            rejects([&] { decode(good + "x"); }, "Trailing");
        });
        add(prefix + "all configurable read limits", [format] {
            const std::string bytes = encode(sample(), format);
            for (unsigned limit = 0; limit < 5; ++limit) {
                FieldReadLimits l;
                if (limit == 0)
                    l.max_payload_bytes = 1;
                if (limit == 1)
                    l.max_decoded_bytes = 32;
                if (limit == 2)
                    l.max_string_bytes = 1;
                if (limit == 3)
                    l.max_fields = 1;
                if (limit == 4)
                    l.max_components = 1;
                rejects([&] { decode(bytes, l); }, "limit");
            }
            FieldReadLimits exact;
            exact.max_decoded_bytes =
                encode(sample(), format == FieldFormat::Lzma ? FieldFormat::Binary : format).size();
            require(same(sample(), decode(bytes, exact)), "Exact decoded size allowed");
        });
        add(prefix + "failed output including late write failure", [format] {
            for (std::size_t n : {std::size_t(0), std::size_t(32)}) {
                ShortOutput buffer(n);
                std::ostream out(&buffer);
                rejects([&] { sample().write(out, format); });
            }
            ShortOutput buffer(32);
            std::ostream out(&buffer);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            rejects([&] { sample().write(out, format); });
        });
    }
}

// Проверки метаданных, размеров и пределов чтения повреждённых файлов.
void register_validation() {
    add("snapshot validation rejects invalid data", [] {
        for (unsigned change = 0; change < 12; ++change) {
            FieldSnapshot s = sample();
            switch (change) {
            case 0:
                s.dimension = 4;
                break;
            case 1:
                s.mesh_sha256 = "invalid";
                break;
            case 2:
                s.time = std::numeric_limits<double>::infinity();
                break;
            case 3:
                s.fields[0].location = static_cast<FieldLocation>(99);
                break;
            case 4:
                s.fields[0].name = "";
                break;
            case 5:
                s.fields.push_back(s.fields[0]);
                break;
            case 6:
                s.fields[0].components.clear();
                break;
            case 7:
                s.fields[0].components[1] = "x";
                break;
            case 8:
                s.fields[0].values.pop_back();
                break;
            case 9:
                s.fields[0].units = "bad\nunit";
                break;
            case 10:
                s.fields[0].name = std::string("\xc0\xaf", 2);
                break;
            case 11:
                s.counts[0] = std::numeric_limits<U64>::max();
                break;
            }
            rejects([&] { s.validate(); });
            std::ostringstream out;
            rejects([&] { s.write(out); });
            require(out.str().empty(), "No write before validation");
        }
        rejects([] { FieldSnapshot().validate(); });
    });
    add("text rejects malformed metadata and values", [] {
        const std::string good = encode(sample(), FieldFormat::Text);
        const std::pair<std::string, std::string> changes[] = {{"AEROFIELDS 1", "AEROFIELDS 2"},
                                                               {"DIMENSION 2", "DIMENSION 0"},
                                                               {"LOCATION NODE", "LOCATION UNKNOWN"},
                                                               {"COMPONENTS 3", "COMPONENTS 0"},
                                                               {"VALUES 4", "VALUES 3"},
                                                               {"FIELDS 4", "FIELDS 18446744073709551616"},
                                                               {"STEP 100", "STEP -1"},
                                                               {"TIME 0.001", "TIME nan"},
                                                               {"DATA 0.25", "DATA 1.2bad"},
                                                               {"\"x\"", "\"\\q\""},
                                                               {"\"x\"", "\"x\"oops"},
                                                               {"\"y\"", "\"x\""}};
        for (const auto& c : changes)
            rejects([&] { decode(replaced(good, c.first, c.second)); });
    });
}

// Двоичная разметка, адресное чтение, сжатие и ошибки файловых операций.
void register_binary_layout() {
    add("independent binary layout fixture", [] {
        std::string bytes = "AEROFLDB";
        u64(bytes, 1);
        u64(bytes, 2);
        bytes.append(32, '\0');
        for (U64 n : {U64(3), U64(3), U64(3), U64(1)})
            u64(bytes, n);
        u64(bytes, 7);
        u64(bytes, 0x3fe0000000000000ULL);
        u64(bytes, 1);
        string(bytes, "p");
        bytes += char(3);
        u64(bytes, 1);
        string(bytes, "value");
        string(bytes, "Pa");
        u64(bytes, 1);
        u64(bytes, 0x4000000000000000ULL);
        const FieldSnapshot s = decode(bytes);
        require(s.step == 7 && s.time == 0.5 && s.fields[0].values[0] == 2 &&
                    s.mesh_sha256 == std::string(64, '0'),
                "External binary layout");
        require(encode(s, FieldFormat::Binary) == bytes, "Stable portable binary layout");
    });
    add("binary rejects invalid version lengths counts and location", [] {
        const std::string good = encode(sample(), FieldFormat::Binary);
        for (std::size_t pos : {std::size_t(8), std::size_t(16), std::size_t(104), std::size_t(112)}) {
            std::string bad = good;
            set_u64(bad, pos, std::numeric_limits<U64>::max());
            rejects([&] { decode(bad); });
        }
        std::string bad = good;
        bad[120 + sample().fields[0].name.size()] = char(99);
        rejects([&] { decode(bad); }, "location");
    });
    add("LZMA detects corruption and rejects wrong inner formats", [] {
        const std::string good = encode(sample(), FieldFormat::Lzma);
        for (std::size_t p :
             {std::size_t(8), std::size_t(15), std::size_t(20), std::size_t(24), std::size_t(29)}) {
            std::string bad = good;
            bad[p] = static_cast<char>(static_cast<unsigned char>(bad[p]) ^ 255);
            rejects([&] { decode(bad); });
        }
        rejects([&] { decode(packed(encode(sample(), FieldFormat::Text))); }, "binary");
        rejects([&] { decode(packed(good)); }, "binary");
    });
    add("multi-block LZMA and component-major large arrays", [] {
        FieldSnapshot s = sample();
        s.fields.resize(1);
        s.fields[0].components.resize(3);
        s.counts[0] = 100000;
        s.fields[0].values.resize(300000);
        for (std::size_t i = 0; i < s.fields[0].values.size(); ++i)
            s.fields[0].values[i] = static_cast<double>(i % 997);
        const std::string binary = encode(s, FieldFormat::Binary), compressed = encode(s, FieldFormat::Lzma);
        require(binary.size() > 2 * 1024 * 1024 && compressed.size() < binary.size(),
                "Multiple blocks compressed");
        require(same(s, decode(compressed)), "Large snapshot round trip");
        FieldReadLimits limit;
        limit.max_decoded_bytes = 1024 * 1024;
        rejects([&] { decode(compressed, limit); }, "limit");
    });
    add("selected nodes preserve component order across short and long gaps", [] {
        auto snapshot = sample();
        snapshot.counts[0] = 4096;
        Field& node = snapshot.fields.front();
        node.values.resize(snapshot.counts[0] * node.components.size());
        for (std::size_t i = 0; i < node.values.size(); ++i)
            node.values[i] = static_cast<double>(i) + 0.25;

        TempFile file;
        // В одной выборке есть смежные узлы, короткий промежуток и пропуск длиннее буфера чтения.
        const std::vector<U64> ids = {1, 2, 4, 3000, 3001};
        const FieldNodeSelection selection = {ids, snapshot.mesh_sha256, snapshot.dimension, snapshot.counts};
        std::vector<double> expected;
        for (std::size_t c = 0; c < node.components.size(); ++c)
            for (auto id : ids)
                expected.push_back(node.values[c * snapshot.counts[0] + id]);

        for (FieldFormat format : {FieldFormat::Binary, FieldFormat::Text, FieldFormat::Lzma}) {
            snapshot.write(file.path, format);
            const auto selected = FieldSnapshot::read_nodes(file.path, selection);
            require(selected.size() == snapshot.fields.size(), "Selection field metadata");
            require(selected.front().values == expected, "Selected values must follow component order");
            for (std::size_t f = 1; f < selected.size(); ++f)
                require(selected[f].values.empty(), "Only nodal values requested");
        }

        // Последний запрошенный узел находится до конца файла. Усечение пропущенного хвоста тоже является ошибкой.
        snapshot.fields.resize(1);
        auto bytes = encode(snapshot, FieldFormat::Binary);
        bytes.pop_back();
        std::ofstream damaged(file.path.c_str(), std::ios::binary);
        damaged.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        damaged.close();
        require(bool(damaged), "Could not create truncated fixture");
        rejects([&] { FieldSnapshot::read_nodes(file.path, selection); });
    });
    add("indexed binary layout matches ordinary AEROFIELDS", [] {
        const auto original = sample();
        auto metadata = original;
        for (auto& f : metadata.fields)
            f.values.clear();
        TempFile file;
        const auto offsets = metadata.write_binary_layout(file.path);
        require(offsets.size() == original.fields.size(), "One range per field");
        std::fstream out(file.path, std::ios::in | std::ios::out | std::ios::binary);
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            out.seekp(static_cast<std::streamoff>(offsets[i]));
            std::string data;
            for (double value : original.fields[i].values) {
                U64 bits;
                std::memcpy(&bits, &value, 8);
                u64(data, bits);
            }
            out.write(data.data(), static_cast<std::streamsize>(data.size()));
        }
        out.close();
        require(out && same(original, FieldSnapshot::read(file.path)), "Indexed field write round trip");
        rejects([&] { original.write_binary_layout(file.path); }, "size mismatch");
        metadata.counts[0] = std::numeric_limits<U64>::max();
        rejects([&] { metadata.write_binary_layout(file.path); }, "overflow");
    });
    add("filesystem and stream errors preserve existing files", [] {
        TempFile file;
        {
            std::ofstream out(file.path.c_str());
            out << "keep";
        }
        FieldSnapshot s = sample();
        rejects([&] { s.write(file.path, static_cast<FieldFormat>(99)); });
        s.fields[0].values.clear();
        rejects([&] { s.write(file.path); });
        std::ifstream in(file.path.c_str());
        std::string value;
        in >> value;
        require(value == "keep", "Validation before truncation");
        rejects([&] { FieldSnapshot::read(file.path + "/missing"); });
        rejects([&] { sample().write(file.path + "/missing"); });
        rejects([&] { FieldSnapshot::read(std::string("build/x\0bad", 11)); });
        rejects([&] { sample().write(std::string("build/x\0bad", 11)); });
        std::istringstream failed;
        failed.setstate(std::ios::badbit);
        rejects([&] { FieldSnapshot::read(failed); });
    });
}

void register_tests() {
    register_mesh_identity();
    register_formats();
    register_validation();
    register_binary_layout();
}
}
int main() {
    register_tests();
    std::size_t failed = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << tests.size() - failed << '/' << tests.size() << " field I/O tests passed\n";
    return failed ? 1 : 0;
}
