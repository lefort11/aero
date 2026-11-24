#include "meshdec.h"
#include "../../source/mesh/mesh.h"
#include "../../source/io/lzma_stream.h"
#include "../../source/io/sha256.h"
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace {
using namespace aero;
using namespace aero::meshdec;
// Читает целое слово из двоичного представления.
Id read_word(std::istream& in) {
    Id v = 0;
    for (unsigned k = 0; k < 8; ++k) {
        const int byte = in.get();
        require(byte != EOF, "Truncated archive");
        v |= static_cast<Id>(byte) << (8 * k);
    }
    return v;
}
// Восстанавливает вещественное число из двоичного представления.
double read_real(std::istream& in) {
    const Id bits = read_word(in);
    double value;
    std::memcpy(&value, &bits, 8);
    return value;
}
// Читает заданный диапазон байтов тестового архива.
std::string bytes(std::istream& in, Id n) {
    std::string text(static_cast<std::size_t>(n), '\0');
    if (n)
        in.read(&text[0], position(n));
    return text;
}
// Читает полное содержимое созданного тестового файла.
std::string contents(const std::string& path) {
    auto f = input(path);
    return bytes(f, file_size(path));
}
// Создаёт архив разбиения тестовой сетки для проверки его структуры.
void make_archive(const Options& o, const std::string& scratch, int workers) {
    Scratch work;
    work.create(scratch);
    const auto& d = work.path;
    unpack(o, d);
    index_topology(d, o.memory);
    for (int r = 0; r < workers; ++r)
        order_entities(d, o.memory, r, workers);
    if (workers > 1)
        choose_splitters(d, o.memory, workers);
    for (int r = 0; r < workers; ++r)
        sort_range(d, o.memory, r, workers);
    plan_leaves(d, o, workers);
    for (int r = 0; r < workers; ++r)
        pack_leaves(d, o.compress, r, workers);
    write_catalog(d, o.target, o.compress, workers);
    for (int r = 0; r < workers; ++r)
        write_part(d, o.target, r);
}
using Key = std::pair<Id, Id>;
// Адрес и исходная сущность контрольной ссылки архива.
struct Ref {
    Key address, entity;
};
// Читает контрольную ссылку и сохраняет её для проверки связности архива.
void reference(std::istream& in, std::vector<Ref>& refs, Id kind, Id id) {
    const Id leaf = read_word(in), offset = read_word(in);
    refs.push_back({{leaf, offset}, {kind, id}});
}
// Читаем двоичную структуру архива напрямую, чтобы независимо сверить нумерацию и ссылки.
void check(const std::string& path, const Mesh& mesh, bool compressed) {
    const auto m = mesh.view();
    auto f = input(path);
    require(bytes(f, 8) == "AERODEC1" && read_word(f) == 5, "Magic/version mismatch");
    require(read_word(f) == static_cast<Id>(compressed) && read_word(f) == m.dimension,
            "Mesh header mismatch");
    for (Id n : {m.node_count, m.face_count, m.cell_count, m.boundary_face_count, m.patch_count})
        require(read_word(f) == n, "Mesh count mismatch");
    const Id leaves = read_word(f), branches = read_word(f), root = read_word(f), names = read_word(f);
    const Id index = read_word(f), tree_offset = read_word(f), payload = read_word(f);
    require(branches == 2 * leaves - 1 && root == branches - 1 && f.tellg() == position(names),
            "Invalid tree header");
    for (const auto& name : mesh.data().patch_names)
        require(bytes(f, read_word(f)) == name, "Patch name mismatch");
    require(f.tellg() == position(index), "Patch table length mismatch");
    std::vector<std::vector<Id>> nf(m.node_count), fc(m.face_count);
    std::vector<Boundary> labels(m.face_count, {none, none});
    for (Id i = 0; i < m.boundary_face_count; ++i)
        labels[m.boundary_faces[i]] = {i, m.boundary_patch[i]};
    for (Id i = 0; i < m.face_count; ++i)
        for (Id n : m.face_nodes(i))
            nf[n].push_back(i);
    for (Id i = 0; i < m.cell_count; ++i)
        for (Id s : m.cell_faces(i))
            fc[s].push_back(i);
    std::map<Key, Key> objects;
    std::set<Key> unique;
    std::vector<Ref> refs;
    std::vector<Id> lengths;
    Id next = payload;
    for (Id i = 0; i < leaves; ++i) {
        f.seekg(position(index + 48 * i));
        const Id offset = read_word(f), packed = read_word(f), raw = read_word(f);
        const std::array<Id, 3> expected = {{read_word(f), read_word(f), read_word(f)}};
        std::array<Id, 3> found{};
        require(offset == next, "Discontinuous leaf payload");
        next += packed;
        f.seekg(position(offset));
        auto data = bytes(f, packed);
        if (compressed) {
            std::istringstream container(data);
            require(bytes(container, 8) == "AEROLZMA", "Leaf missing compression header");
            io::LzmaInputBuffer decoder(container, raw);
            std::istream stream(&decoder);
            stream.exceptions(std::ios::badbit);
            data = bytes(stream, raw);
            require(stream.peek() == EOF && !stream.bad(), "Invalid leaf compression frame");
        }
        require(raw == data.size(), "Wrong decoded leaf size");
        lengths.push_back(raw);
        std::istringstream in(data);
        while (in.tellg() < position(raw)) {
            const Id local = static_cast<Id>(in.tellg());
            const Id kind = read_word(in), id = read_word(in);
            require(kind < 3 && unique.insert({kind, id}).second, "Duplicate or unknown entity");
            ++found[static_cast<std::size_t>(kind)];
            objects[{i, local}] = {kind, id};
            if (kind == 0) {
                require(id < m.node_count && read_real(in) == m.x[id] && read_real(in) == m.y[id] &&
                            read_real(in) == m.z[id],
                        "Coordinate mismatch");
                require(read_word(in) == nf[id].size(), "Incomplete incident faces");
                for (Id face : nf[id])
                    reference(in, refs, 1, face);
            } else if (kind == 1) {
                require(id < m.face_count && read_word(in) == labels[id].id &&
                            read_word(in) == labels[id].patch,
                        "Face boundary mismatch");
                const auto nodes = m.face_nodes(id);
                require(read_word(in) == nodes.size && read_word(in) == fc[id].size(),
                        "Face incidence mismatch");
                for (Id n : nodes)
                    reference(in, refs, 0, n);
                for (Id c : fc[id])
                    reference(in, refs, 2, c);
            } else {
                require(id < m.cell_count, "Cell ID out of range");
                const auto faces = m.cell_faces(id);
                require(read_word(in) == faces.size, "Cell incidence mismatch");
                for (Id k = 0; k < faces.size; ++k) {
                    reference(in, refs, 1, faces[k]);
                    require(in.get() == (m.face_orientation(id, k) > 0 ? 1 : 255),
                            "Cell orientation mismatch");
                }
            }
        }
        require(in.tellg() == position(raw) && found == expected, "Unexpected end of leaf");
    }
    require(next == file_size(path) && unique.size() == m.node_count + m.face_count + m.cell_count,
            "Lost entity or trailing data");
    for (const auto& r : refs)
        require(objects.at(r.address) == r.entity, "Incorrect local/remote address");

    f.seekg(position(tree_offset));
    std::vector<Chunk> ranges;
    for (Id b = 0; b < branches; ++b) {
        const Id first = read_word(f), count = read_word(f), left = read_word(f), right = read_word(f);
        require(count && first + count <= leaves, "Bad tree range");
        Id raw = 0;
        for (Id k = 0; k < count; ++k)
            raw += lengths.at(first + k);
        require(read_word(f) == raw, "Bad tree size");
        const Id n = read_word(f), s = read_word(f), c = read_word(f);
        const Box box = {{read_real(f), read_real(f), read_real(f)},
                         {read_real(f), read_real(f), read_real(f)}};
        if (count == 1)
            require(left == none && right == none, "Bad leaf pointers");
        else
            require(left < b && right < b && ranges.at(left).begin == first &&
                        ranges.at(left).end == ranges.at(right).begin &&
                        ranges.at(right).end == first + count,
                    "Bad subtree pointers");
        ranges.push_back({first, first + count});
        if (b == root) {
            require(first == 0 && count == leaves && n == m.node_count && s == m.face_count &&
                        c == m.cell_count,
                    "Bad root");
            const double* axes[] = {m.x, m.y, m.z};
            for (unsigned a = 0; a < 3; ++a)
                require(box.low[a] == *std::min_element(axes[a], axes[a] + n) &&
                            box.high[a] == *std::max_element(axes[a], axes[a] + n),
                        "Bad root bounds");
        }
    }
    // Независимо получаем смежность из Mesh, включая ячейки, у которых общая только одна вершина.
    std::vector<std::set<Id>> stars(m.node_count), halo(2 * leaves);
    std::vector<Id> owner(m.node_count);
    for (const auto& o : objects)
        if (o.second.first == 0)
            owner[o.second.second] = o.first.first;
    for (Id c = 0; c < m.cell_count; ++c)
        for (Id n : m.cell_nodes(c)) {
            stars[n].insert(c);
            halo[2 * owner[n]].insert(c);
        }
    std::vector<bool> boundary_nodes(m.node_count), near_patch(m.node_count);
    for (Id b = 0; b < m.boundary_face_count; ++b)
        for (Id n : m.face_nodes(m.boundary_faces[b]))
            boundary_nodes[n] = true;
    for (Id e = 0; e < m.edge_count; ++e) {
        const Id a = m.edge_begin[e], b = m.edge_end[e];
        if (boundary_nodes[a] || boundary_nodes[b])
            near_patch[a] = near_patch[b] = true;
        if (owner[a] == owner[b])
            continue;
        for (Id c : stars[b])
            if (!halo[2 * owner[a]].count(c))
                halo[2 * owner[a] + 1].insert(c);
        for (Id c : stars[a])
            if (!halo[2 * owner[b]].count(c))
                halo[2 * owner[b] + 1].insert(c);
    }
    const Id star_count = read_word(f), halo_count = read_word(f);
    // Читает или проверяет диапазоны строк индексированной таблицы архива.
    const auto rows = [&](const std::vector<std::set<Id>>& expected, Id count) {
        require(read_word(f) == 0, "First halo offset must be zero");
        Id k = 0;
        for (const auto& row : expected) {
            k += row.size();
            require(read_word(f) == k, "Incorrect halo row extent");
        }
        require(k == count, "Incorrect halo count");
        for (const auto& row : expected)
            for (Id c : row) {
                const Id leaf = read_word(f), off = read_word(f);
                require(objects.at({leaf, off}) == Key(2, c), "Incorrect halo cell address");
            }
    };
    std::vector<std::set<Id>> far(leaves);
    for (Id leaf = 0; leaf < leaves; ++leaf) {
        const auto& core = halo[2 * leaf];
        const auto& outer = halo[2 * leaf + 1];
        for (const auto* near : {&core, &outer})
            for (Id c : *near)
                for (Id n : m.cell_nodes(c))
                    for (Id other : stars[n])
                        if (!core.count(other) && !outer.count(other))
                            far[leaf].insert(other);
    }
    auto band = near_patch;
    for (Id n = 0; n < m.node_count; ++n)
        if (near_patch[n])
            for (Id c : stars[n])
                for (Id vertex : m.cell_nodes(c))
                    band[vertex] = true;
    for (Id n = 0; n < m.node_count; ++n)
        if (!band[n])
            stars[n].clear();
    rows(stars, star_count);
    rows(halo, halo_count);
    rows(far, read_word(f));
    io_detail::Sha256Buffer hash;
    std::ostream canonical(&hash);
    mesh.write_fingerprint_payload(canonical);
    require(bytes(f, 64) == hash.hex_digest(), "Incorrect source mesh fingerprint");
    require(f.tellg() == position(payload), "Incorrect halo directory length");
}
// Выполняет проверки каталога, геометрии и halo-связей архива meshdec.
void tests() {
    Scratch temp;
    temp.create(".");
    Options o;
    o.memory = 2048;
    const char* fixtures[] = {"pulse_1d", "shear_jump_2d", "wave_3d"};
    for (unsigned d = 1; d <= 3; ++d) {
        const auto mesh = Mesh::read(std::string("qa/cases/") + fixtures[d-1] + "/grid.mesh");
        o.source = temp.path + "/source";
        mesh.write(o.source, d == 1 ? MeshFormat::Text : (d == 2 ? MeshFormat::Binary : MeshFormat::Lzma));
        o.compress = d == 3;
        o.leaf_bytes = d == 1 ? 60 : (d == 2 ? none : 4096);
        o.target = temp.path + "/first";
        make_archive(o, temp.path, 1);
        check(o.target, mesh, o.compress);
        const auto original = contents(o.target);
        o.target = temp.path + "/second";
        make_archive(o, temp.path, 3);
        require(contents(o.target) == original, "Archive depends on process count");
    }
    const auto line = Mesh::read("qa/cases/pulse_1d/grid.mesh");
    line.write(o.source, MeshFormat::Binary);
    o.compress = false;
    o.leaf_bytes = none;
    o.target_ranks = 5;
    make_archive(o, temp.path, 1);
    check(o.target, line, false);
    {
        auto f = input(o.target);
        f.seekg(72);
        const Id leaves = read_word(f);
        f.seekg(104);
        const Id index = read_word(f), nodes = line.view().node_count;
        require(leaves >= o.target_ranks, "Future solver rank hint ignored");
        for (Id i = 0; i < leaves; ++i) {
            f.seekg(position(index + i * 48 + 24));
            require(read_word(f) <= (nodes + 19) / 20, "Too many vertices in a planned leaf");
        }
    }
    const auto with_hint = contents(o.target);
    make_archive(o, temp.path, 4);
    require(contents(o.target) == with_hint, "Future solver size tied to preprocessing size");
    Mesh(MeshData{}).write(o.source, MeshFormat::Text);
    make_archive(o, temp.path, 3);
    {
        auto f = input(o.target);
        f.seekg(72);
        require(read_word(f) == 0 && read_word(f) == 0 && read_word(f) == none && file_size(o.target) == 240,
                "Invalid empty archive");
    }
    auto truncated = output(o.source);
    truncated.write("AEROMSHB", 8);
    truncated.close();
    bool rejected = false;
    try {
        make_archive(o, temp.path, 1);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Truncated input accepted");
}
}
// Запускает проверки модуля и возвращает ненулевой код при ошибке.
int main() {
    try {
        tests();
        std::cout << "meshdec tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "meshdec tests: " << e.what() << "\n";
        return 1;
    }
}
