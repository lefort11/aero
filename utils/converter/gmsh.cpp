#include "detail.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
namespace aero {
namespace converter {
namespace {
typedef std::uint64_t Tag;
const Tag max_items = 100000000; // Ограничение объявленных размеров до выделения памяти.
// Читает числа Gmsh с учётом текстового режима, разрядности и порядка байтов.
struct Reader {
    std::istream& in;
    bool binary, swap;
    int width;
    // Начинает чтение Gmsh с исходными параметрами до разбора заголовка.
    explicit Reader(std::istream& s) : in(s), binary(false), swap(false), width(8) {}
    // Проверяет обязательный маркер или значение входного формата.
    void expect(const std::string& s) {
        std::string t;
        if (!(in >> t) || t != s)
            throw MeshError("Gmsh: expected " + s);
    }
    // Пропускает требуемый перевод строки после двоичного блока Gmsh.
    void newline() {
        std::string rest;
        if (!std::getline(in, rest))
            throw MeshError("Gmsh: truncated line");
        if (rest.find_first_not_of(" \t\r") != std::string::npos)
            throw MeshError("Gmsh: unexpected header data");
    }
    // Читает двоичное значение исходного типа с учётом порядка байтов.
    template <class T> T raw() {
        char bytes[sizeof(T)];
        if (!in.read(bytes, sizeof(T)))
            throw MeshError("Gmsh: truncated binary input");
        if (swap)
            std::reverse(bytes, bytes + sizeof(T));
        T value;
        std::memcpy(&value, bytes, sizeof(T));
        return value;
    }
    // Читает неотрицательное целое из текстового раздела Gmsh.
    Tag ascii_uint() {
        std::string t;
        if (!(in >> t) || t.empty() || t.size() > 20)
            throw MeshError("Gmsh: invalid integer");
        Tag n = 0;
        for (char c : t) {
            if (c < '0' || c > '9' ||
                n > (std::numeric_limits<Tag>::max() - static_cast<unsigned>(c - '0')) / 10)
                throw MeshError("Gmsh: invalid/overflowing integer");
            n = n * 10 + static_cast<unsigned>(c - '0');
        }
        return n;
    }
    // Читает размер в разрядности, указанной заголовком Gmsh.
    Tag size() { return binary ? (width == 8 ? raw<std::uint64_t>() : raw<std::uint32_t>()) : ascii_uint(); }
    // Читает целое значение с учётом формата и порядка байтов.
    int integer() {
        if (binary)
            return raw<std::int32_t>();
        long long n;
        if (!(in >> n) || n < std::numeric_limits<int>::min() || n > std::numeric_limits<int>::max())
            throw MeshError("Gmsh: invalid int32");
        return static_cast<int>(n);
    }
    // Читает вещественное значение с учётом формата и порядка байтов.
    double real() {
        if (binary)
            return raw<double>();
        double d;
        if (!(in >> d))
            throw MeshError("Gmsh: invalid coordinate");
        return d;
    }
    // Проверяет размер входного массива перед приведением к внутреннему типу.
    Tag count() {
        Tag n = size();
        if (n > max_items)
            throw MeshError("Gmsh: count exceeds converter limit");
        return n;
    }
    // Читает идентификатор сущности Gmsh заданной разрядности.
    Tag tag() {
        Tag n = size();
        if (n == 0)
            throw MeshError("Gmsh: tags must be positive");
        return n;
    }
};
// Тип элемента, геометрическая сущность, физическая метка и номера его вершин.
struct Record {
    int kind, entity;
    unsigned dim;
    Tag physical;
    detail::Nodes nodes;
};
// Для разбора используем собственную локаль потока и возвращаем исходную при выходе.
struct Locale {
    std::istream& s;
    std::locale old;
    // Временно задаёт потоку независимую от ОС локаль разбора чисел.
    explicit Locale(std::istream& x) : s(x), old(x.getloc()) { s.imbue(std::locale::classic()); }
    // Освобождает принадлежащие объекту ресурсы при завершении его времени жизни.
    ~Locale() { s.imbue(old); }
};

// Состояние чтения версии, сущностей и элементов файла Gmsh.
class GmshFile {
    Reader r;            // Источник токенов или двоичных величин в порядке байтов файла.
    bool legacy = false; // В 2.2 метки сопоставлены элементам, в 4.1 — геометрическим сущностям.
    std::map<Tag, MeshIndex> node_ids; // Метка Gmsh → позиция в массивах координат.
    std::vector<double> x, y, z;
    std::vector<Record> records; // Элементы с исходными тегами до разрешения ссылок.
    std::map<std::pair<unsigned, int>, std::vector<Tag>> entities;
    std::map<std::pair<unsigned, Tag>, std::string> names;
    bool nodes_seen = false, elements_seen = false, entities_seen = false, names_seen = false;
    unsigned dim = 0; // Максимальная размерность прочитанных элементов.

    // Проверяет версию и настраивает порядок байтов до чтения узлов и элементов.
    void read_header() {
        r.expect("$MeshFormat");
        std::string version;
        int filetype, datasize;
        if (!(r.in >> version >> filetype >> datasize) || (version != "2.2" && version != "4.1") ||
            (filetype != 0 && filetype != 1) || datasize != 8)
            throw MeshError("Gmsh: supported formats are MSH 2.2/4.1, ASCII or binary, data-size 8");
        r.newline();
        r.binary = filetype == 1;
        if (r.binary) {
            const std::int32_t marker = r.raw<std::int32_t>();
            if (marker != 1) {
                r.swap = true;
                if (marker != 16777216)
                    throw MeshError("Gmsh: invalid byte order marker");
            }
        }
        r.expect("$EndMeshFormat");
        r.newline();

        legacy = version == "2.2";
    }

    // Связывает размерность и номер физической группы с её именем в файле.
    void read_names() {
        if (names_seen)
            throw MeshError("Gmsh: repeated PhysicalNames");
        names_seen = true;
        Tag count = r.ascii_uint();
        if (count > max_items)
            throw MeshError("Gmsh: too many names");
        for (Tag i = 0; i < count; ++i) {
            Tag d = r.ascii_uint(), tag = r.ascii_uint();
            if (d > 3 || !tag)
                throw MeshError("Gmsh: invalid physical group");
            r.in >> std::ws;
            if (r.in.get() != '"')
                throw MeshError("Gmsh: expected quoted physical name");
            std::string name;
            char c;
            while (r.in.get(c) && c != '"') {
                if (name.size() == 4096)
                    throw MeshError("Gmsh: name too long");
                name += c;
            }
            if (!r.in || name.empty())
                throw MeshError("Gmsh: invalid physical name");
            if (!names.insert(std::make_pair(std::make_pair(static_cast<unsigned>(d), tag), name)).second)
                throw MeshError("Gmsh: duplicate physical name tag");
        }
        r.expect("$EndPhysicalNames");
    }

    // В MSH 4.1 физические группы привязаны к геометрическим сущностям, а элементы ссылаются на эти сущности.
    void read_entities() {
        if (legacy || entities_seen)
            throw MeshError("Gmsh: unexpected/repeated Entities");
        entities_seen = true;
        Tag counts[4];
        for (unsigned d = 0; d < 4; ++d)
            counts[d] = r.count();
        for (unsigned d = 0; d < 4; ++d)
            for (Tag i = 0; i < counts[d]; ++i) {
                const int tag = r.integer();
                if (tag <= 0)
                    throw MeshError("Gmsh: invalid entity tag");
                // Точке соответствуют три координаты, остальным сущностям — шесть координат ограничивающего параллелепипеда.
                for (unsigned k = 0; k < (d == 0 ? 3u : 6u); ++k)
                    r.real();
                std::vector<Tag> physical;
                Tag n = r.count();
                for (Tag k = 0; k < n; ++k) {
                    int p = r.integer();
                    if (p <= 0)
                        throw MeshError("Gmsh: invalid physical tag");
                    physical.push_back(static_cast<Tag>(p));
                }
                if (!entities.insert(std::make_pair(std::make_pair(d, tag), physical)).second)
                    throw MeshError("Gmsh: duplicate entity");
                if (d) {
                    Tag nbound = r.count();
                    for (Tag k = 0; k < nbound; ++k)
                        r.integer();
                }
            }
        r.expect("$EndEntities");
    }

    // В Gmsh 4.x заголовок содержит крайние теги, в 2.x на том же месте заканчивается строка.
    std::pair<Tag, Tag> read_tag_range() {
        if (legacy) {
            r.newline();
            return {0, 0};
        }
        const Tag first = r.tag();
        const Tag last = r.tag();
        return {first, last};
    }

    // Читает координаты и переводит произвольные положительные метки узлов в плотные номера сетки.
    void read_nodes() {
        if (nodes_seen)
            throw MeshError("Gmsh: repeated Nodes");
        nodes_seen = true;
        const Tag blocks = legacy ? 1 : r.count();
        const Tag total = legacy ? r.ascii_uint() : r.count();
        if (total == 0 || total > max_items)
            throw MeshError("Gmsh: invalid node count");
        const auto range = read_tag_range();
        for (Tag block = 0; block < blocks; ++block) {
            int entity_dim = 0, parametric = 0;
            Tag n = total;
            if (!legacy) {
                entity_dim = r.integer();
                int entity = r.integer();
                parametric = r.integer();
                n = r.count();
                if (entity_dim < 0 || entity_dim > 3 || entity <= 0 || (parametric != 0 && parametric != 1))
                    throw MeshError("Gmsh: invalid node block");
            }
            if (n > total - x.size())
                throw MeshError("Gmsh: inconsistent node block count");
            std::vector<Tag> tags;
            if (!legacy) {
                tags.resize(static_cast<std::size_t>(n));
                for (Tag& t : tags)
                    t = r.tag();
            }
            for (Tag i = 0; i < n; ++i) {
                Tag tag;
                if (legacy) {
                    if (r.binary) {
                        int t = r.integer();
                        if (t <= 0)
                            throw MeshError("Gmsh: invalid node tag");
                        tag = static_cast<Tag>(t);
                    } else
                        tag = r.tag();
                } else
                    tag = tags[static_cast<std::size_t>(i)];
                // Запоминаем позицию до добавления координат: ею затем заменим Gmsh-тег в элементах.
                if (!node_ids.insert(std::make_pair(tag, x.size())).second)
                    throw MeshError("Gmsh: duplicate node tag");
                x.push_back(r.real());
                y.push_back(r.real());
                z.push_back(r.real());
                if (parametric)
                    for (int k = 0; k < entity_dim; ++k)
                        r.real();
            }
        }
        if (x.size() != total ||
            (!legacy && (node_ids.begin()->first != range.first || node_ids.rbegin()->first != range.second)))
            throw MeshError("Gmsh: inconsistent node counts/tag range");
        r.expect("$EndNodes");
    }

    // Сохраняет элементы до конца чтения. В файле секция узлов может встретиться после элементов.
    void read_elements() {
        std::set<Tag> element_ids;
        if (elements_seen)
            throw MeshError("Gmsh: repeated Elements");
        elements_seen = true;
        Tag blocks = legacy ? 0 : r.count(), total = legacy ? r.ascii_uint() : r.count();
        if (!total || total > max_items)
            throw MeshError("Gmsh: invalid element count");
        const auto range = read_tag_range();
        Tag read = 0, block = 0;
        while (read < total) {
            int kind = 0, entity = 0, ntags = 0, entity_dim = -1;
            Tag n = 1, first_id = 0;
            if (!legacy) {
                entity_dim = r.integer();
                entity = r.integer();
                kind = r.integer();
                n = r.count();
                ++block;
            } else if (r.binary) {
                kind = r.integer();
                int nn = r.integer();
                ntags = r.integer();
                if (nn <= 0)
                    throw MeshError("Gmsh: invalid block size");
                n = static_cast<Tag>(nn);
            } else {
                first_id = r.tag();
                kind = r.integer();
                ntags = r.integer();
            }
            // Тип элемента однозначно задаёт его размерность и число последующих ссылок на узлы.
            const unsigned edim = detail::dimension(kind);
            const std::size_t npe = detail::node_count(kind);
            if ((!legacy && (entity_dim != static_cast<int>(edim) || entity <= 0)) || ntags < 0 ||
                ntags > 2 || n > total - read || n == 0)
                throw MeshError("Gmsh: unsupported partition tags or invalid element block");
            for (Tag j = 0; j < n; ++j) {
                Tag id = first_id;
                if (!legacy)
                    id = r.tag();
                else if (r.binary) {
                    int t = r.integer();
                    if (t <= 0)
                        throw MeshError("Gmsh: invalid element tag");
                    id = static_cast<Tag>(t);
                }
                if (!element_ids.insert(id).second)
                    throw MeshError("Gmsh: duplicate element tag");
                Record e;
                e.kind = kind;
                e.entity = entity;
                e.dim = edim;
                e.physical = 0;
                for (int k = 0; k < ntags; ++k) {
                    int t = r.integer();
                    if (t < 0)
                        throw MeshError("Gmsh: negative tag");
                    if (k == 0)
                        e.physical = static_cast<Tag>(t);
                }
                for (std::size_t k = 0; k < npe; ++k) {
                    Tag t;
                    if (legacy && r.binary) {
                        int q = r.integer();
                        if (q <= 0)
                            throw MeshError("Gmsh: invalid node reference");
                        t = static_cast<Tag>(q);
                    } else
                        t = r.tag();
                    e.nodes.push_back(t);
                }
                records.push_back(std::move(e));
                dim = std::max(dim, edim);
            }
            read += n;
        }
        if ((!legacy && (block != blocks || *element_ids.begin() != range.first ||
                         *element_ids.rbegin() != range.second)))
            throw MeshError("Gmsh: inconsistent element counts/tag range");
        r.expect("$EndElements");
    }

    // Пропускает необязательную текстовую секцию до её парной завершающей метки.
    void skip_section(const std::string& section) {
        // Секции периодичности и разбиения содержат связи сетки. Их нельзя пропускать при преобразовании.
        if (section == "$Periodic" || section == "$PartitionedEntities" || section == "$GhostElements")
            throw MeshError("Gmsh: periodic/partitioned input is not yet supported");
        if (r.binary)
            throw MeshError("Gmsh: unsupported binary section " + section);
        if (section.empty() || section[0] != '$' || section.compare(0, 4, "$End") == 0)
            throw MeshError("Gmsh: unexpected section " + section);
        const std::string end = "$End" + section.substr(1);
        std::string line;
        bool found = false;
        while (std::getline(r.in, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line == end) {
                found = true;
                break;
            }
        }
        if (!found)
            throw MeshError("Gmsh: unterminated section " + section);
    }
    // Только после чтения всех секций разрешаем ссылки на узлы и размечаем внешние грани.
    Mesh build_mesh(const Options& options) {
        if (r.in.bad() || !nodes_seen || !elements_seen || dim == 0)
            throw MeshError("Gmsh: missing mesh data");
        detail::Builder builder(dim, options);
        builder.data.x = std::move(x);
        builder.data.y = std::move(y);
        builder.data.z = std::move(z);
        for (Record& e : records) {
            for (MeshIndex& tag : e.nodes) {
                auto it = node_ids.find(tag);
                if (it == node_ids.end())
                    throw MeshError("Gmsh: unknown node reference");
                tag = it->second;
            }
            if (e.dim == dim)
                builder.cell(detail::cell_faces(e.kind, e.nodes));
        }
        // Элементы размерности d−1 задают метки границы построенных d-мерных ячеек.
        for (const Record& e : records)
            if (e.dim + 1 == dim) {
                Tag tag = e.physical;
                if (!legacy) {
                    const auto it = entities.find(std::make_pair(e.dim, e.entity));
                    if (it != entities.end()) {
                        if (it->second.size() > 1)
                            throw MeshError("Gmsh: boundary entity belongs to several physical groups");
                        if (!it->second.empty())
                            tag = it->second[0];
                    }
                }
                if (tag) {
                    const auto it = names.find(std::make_pair(e.dim, tag));
                    const std::string name =
                        it == names.end() ? "physical_" + std::to_string(tag) : it->second;
                    builder.mark(e.nodes, name);
                }
            }
        return builder.finish();
    }

public:
    // Связывает разборщик сетки Gmsh с входным потоком.
    explicit GmshFile(std::istream& input) : r(input) {}
    // Разбирает секции Gmsh и собирает внутреннюю сетку.
    Mesh read(const Options& options) {
        read_header();
        std::string section;
        while (r.in >> section) {
            r.newline();
            if (section == "$PhysicalNames")
                read_names();
            else if (section == "$Entities")
                read_entities();
            else if (section == "$Nodes")
                read_nodes();
            else if (section == "$Elements")
                read_elements();
            else
                skip_section(section);
        }
        return build_mesh(options);
    }
};
}
// Читает сетку Gmsh и переводит её топологию во внутренний формат.
Mesh read_gmsh(std::istream& input, const Options& options) {
    Locale locale(input);
    return GmshFile(input).read(options);
}
// Читает сетку Gmsh и переводит её топологию во внутренний формат.
Mesh read_gmsh(const std::string& path, const Options& options) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in)
        throw MeshError("Cannot open Gmsh file: " + path);
    return read_gmsh(in, options);
}
}
}
