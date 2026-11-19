#include "converter.h"
#include "detail.h"
#include "../../third_party/lzma/LzmaDec.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <set>
namespace aero {
namespace converter {
namespace {
typedef std::uint64_t U64;
std::string lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string clean(std::string s) {
    if (s.compare(0, 3, "\xef\xbb\xbf") == 0)
        s.erase(0, 3);
    const std::size_t comment = s.find('#');
    if (comment != std::string::npos)
        s.erase(comment);
    return s;
}
U64 number(const std::string& s) {
    if (s.empty())
        throw MeshError("another: missing integer");
    U64 n = 0;
    for (char c : s) {
        if (c < '0' || c > '9' || n > (std::numeric_limits<U64>::max() - static_cast<unsigned>(c - '0')) / 10)
            throw MeshError("another: invalid nonnegative integer: " + s);
        n = n * 10 + static_cast<unsigned>(c - '0');
    }
    return n;
}
std::string directory(std::string path) {
    if (path.empty() || path.find('\0') != std::string::npos)
        throw MeshError("another: invalid input path");
    const std::size_t slash = path.find_last_of("/\\");
    if (lower(path.substr(slash == std::string::npos ? 0 : slash + 1)) == "mesh.txt")
        return slash == std::string::npos ? "." : path.substr(0, slash + 1);
    return path;
}
std::string join(const std::string& dir, const char* file) {
    return dir + "/" + file;
}
bool exists(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    return static_cast<bool>(in);
}
typedef std::map<std::string, std::string> Header;
Header header(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in)
        throw MeshError("another: cannot open " + path);
    Header h;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(clean(line));
        std::string key, value;
        if (!(row >> key))
            continue;
        if (!(row >> value))
            throw MeshError("another: missing header value: " + key);
        if (!h.insert(std::make_pair(lower(key), value)).second)
            throw MeshError("another: duplicate header key: " + key);
    }
    if (in.bad())
        throw MeshError("another: header read failed");
    return h;
}
std::string get(const Header& h, const char* name, const char* alias = nullptr) {
    auto i = h.find(name);
    if (i == h.end() && alias)
        i = h.find(alias);
    if (i == h.end())
        throw MeshError(std::string("another: missing header key: ") + name);
    if (alias && h.count(name) && h.count(alias) && h.find(name)->second != h.find(alias)->second)
        throw MeshError("another: conflicting header aliases");
    return i->second;
}
U64 bounded(U64 n) {
    if (n > 1000000000ULL)
        throw MeshError("another: array count exceeds converter limit");
    return n;
}
U64 little(const unsigned char* p, unsigned size) {
    U64 v = 0;
    for (unsigned k = 0; k < size; ++k)
        v |= static_cast<U64>(p[k]) << (8 * k);
    return v;
}
void* allocate(ISzAllocPtr, std::size_t size) {
    return std::malloc(size);
}
void release(ISzAllocPtr, void* p) {
    std::free(p);
}

// Заголовок .msb содержит число значений, ширину в байтах и код типа (1266 — int32, 1273 — float64).
// Далее идут независимые блоки: размер сжатых данных uint32, исходный размер uint32, поток LZMA-alone.
// Граница блока может проходить внутри числа. Целые записаны младшим байтом вперёд.
class Array {
public:
    Array(const std::string& path, bool binary, unsigned width)
        : in_(path.c_str(), std::ios::binary), binary_(binary), width_(width), remaining_(0), declared_(0),
          pos_(0) {
        if (!in_)
            throw MeshError("another: cannot open " + path);
        if (binary_) {
            declared_ = u32();
            const U64 stored_width = u32(), type = u32();
            bounded(declared_);
            if (stored_width != width || type != (width == 8 ? 1273u : 1266u))
                throw MeshError("another: unsupported MSB scalar encoding");
            remaining_ = declared_ * width;
        }
    }
    U64 declared() const { return declared_; }
    U64 integer() {
        if (!binary_)
            return number(token());
        unsigned char b[4];
        bytes(b, 4);
        U64 v = little(b, 4);
        if (v > 2147483647ULL)
            throw MeshError("another: negative/invalid node index");
        return v;
    }
    double real() {
        if (binary_) {
            unsigned char b[8];
            bytes(b, 8);
            const U64 bits = little(b, 8);
            double d;
            std::memcpy(&d, &bits, 8);
            return d;
        }
        std::istringstream value(token());
        value.imbue(std::locale::classic());
        double d;
        if (!(value >> d) || value.peek() != std::char_traits<char>::eof())
            throw MeshError("another: invalid coordinate");
        return d;
    }
    void finish() {
        if (binary_) {
            if (remaining_ || pos_ != buffer_.size() || in_.peek() != std::char_traits<char>::eof())
                throw MeshError("another: incorrect MSB length/trailing data");
        } else {
            std::string t;
            if (next(t))
                throw MeshError("another: trailing array data");
        }
        if (in_.bad())
            throw MeshError("another: array read failed");
    }

private:
    U64 u32() {
        unsigned char b[4];
        if (!in_.read(reinterpret_cast<char*>(b), 4))
            throw MeshError("another: truncated MSB header");
        return little(b, 4);
    }
    bool next(std::string& t) {
        while (!(row_ >> t)) {
            std::string line;
            if (!std::getline(in_, line))
                return false;
            if (line.size() > 1048576)
                throw MeshError("another: text line too long");
            row_.clear();
            row_.str(clean(line));
        }
        return true;
    }
    std::string token() {
        std::string t;
        if (!next(t))
            throw MeshError("another: truncated text array");
        return t;
    }
    void block() {
        const U64 packed = u32(), unpacked = u32();
        if (packed < 13 || packed > 64 * 1024 * 1024 || !unpacked || unpacked > 64 * 1024 * 1024 ||
            unpacked > remaining_)
            throw MeshError("another: invalid MSB block lengths");
        std::vector<unsigned char> encoded(static_cast<std::size_t>(packed));
        if (!in_.read(reinterpret_cast<char*>(encoded.data()), static_cast<std::streamsize>(packed)))
            throw MeshError("another: truncated compressed block");
        if (little(encoded.data() + 5, 8) != unpacked || little(encoded.data() + 1, 4) > 64 * 1024 * 1024)
            throw MeshError("another: invalid LZMA header");
        buffer_.resize(static_cast<std::size_t>(unpacked));
        pos_ = 0;
        SizeT dest = buffer_.size(), source = encoded.size() - 13;
        ELzmaStatus status = LZMA_STATUS_NOT_SPECIFIED;
        const ISzAlloc allocator = {allocate, release};
        const SRes code = LzmaDecode(buffer_.data(), &dest, encoded.data() + 13, &source, encoded.data(), 5,
                                     LZMA_FINISH_END, &status, &allocator);
        if (code != SZ_OK || dest != unpacked || source != encoded.size() - 13 ||
            (status != LZMA_STATUS_FINISHED_WITH_MARK && status != LZMA_STATUS_MAYBE_FINISHED_WITHOUT_MARK))
            throw MeshError("another: corrupt LZMA block");
    }
    void bytes(unsigned char* out, unsigned n) {
        if (n > remaining_)
            throw MeshError("another: truncated scalar array");
        for (unsigned i = 0; i < n; ++i) {
            if (pos_ == buffer_.size())
                block();
            out[i] = buffer_[pos_++];
            --remaining_;
        }
    }
    std::ifstream in_;
    bool binary_;
    unsigned width_;
    U64 remaining_, declared_;
    std::vector<unsigned char> buffer_;
    std::size_t pos_;
    std::istringstream row_;
};
std::map<U64, std::string> labels(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in)
        throw MeshError("another: cannot open " + path);
    std::map<U64, std::string> result;
    std::string line;
    U64 declared = 0;
    bool have_count = false, first = true;
    while (std::getline(in, line)) {
        std::istringstream row(clean(line));
        std::string id, name;
        if (!(row >> id))
            continue;
        if (!(row >> name)) {
            if (!first)
                throw MeshError("another: boundary label has no name");
            declared = bounded(number(id));
            have_count = true;
            first = false;
            continue;
        }
        first = false;
        if (!result.insert(std::make_pair(number(id), name)).second)
            throw MeshError("another: duplicate boundary label ID");
        // Последующие слова задают псевдонимы и классификацию. Для основного имени берём первое.
    }
    if (in.bad() || (have_count && declared != result.size()))
        throw MeshError("another: incorrect label count");
    return result;
}

// Для двумерной сетки третью координату оставляем нулевой.
void read_coordinates(detail::Builder& builder, const std::string& dir, bool binary, U64 dim,
                      U64 nodes_count) {
    builder.data.x.resize(static_cast<std::size_t>(nodes_count));
    builder.data.y.resize(static_cast<std::size_t>(nodes_count));
    builder.data.z.resize(static_cast<std::size_t>(nodes_count), 0);
    Array coordinates(join(dir, binary ? "coordinate.msb" : "coordinate.msh"), binary, 8);
    if (binary && coordinates.declared() != nodes_count * dim)
        throw MeshError("another: coordinate count mismatch");
    for (std::size_t i = 0; i < nodes_count; ++i) {
        builder.data.x[i] = coordinates.real();
        builder.data.y[i] = coordinates.real();
        if (dim == 3)
            builder.data.z[i] = coordinates.real();
    }
    coordinates.finish();
}

// Гибридная запись начинается числом вершин, регулярная использует фиксированную ширину.
void read_cells(detail::Builder& builder, const std::string& dir, bool binary, bool hybrid, U64 dim,
                U64 nodes_count, U64 cells_count) {
    Array topology(join(dir, binary ? "topo.msb" : "topo.msh"), binary, 4);
    const U64 fixed = dim == 2 ? 4 : 8;
    if (binary && !hybrid && topology.declared() != cells_count * fixed)
        throw MeshError("another: cell connectivity count mismatch");
    for (U64 cell = 0; cell < cells_count; ++cell) {
        const U64 n = hybrid ? topology.integer() : fixed;
        if (n < 3 || n > 1000000 || n > nodes_count)
            throw MeshError("another: invalid element size");
        detail::Nodes nodes(static_cast<std::size_t>(n));
        for (MeshIndex& v : nodes) {
            v = topology.integer();
            if (v >= nodes_count)
                throw MeshError("another: cell node index out of range");
        }
        int kind = 100;
        if (dim == 3) {
            if (n == 4)
                kind = 4;
            else if (n == 5)
                kind = 7;
            else if (n == 6)
                kind = 6;
            else if (n == 8)
                kind = 5;
            else
                throw MeshError("another: unsupported 3D element arity");
        }
        // Переставляем вершины четырёхугольника из тензорного порядка another в циклический порядок Gmsh.
        if (n == 4 && dim == 2)
            std::swap(nodes[2], nodes[3]);
        if (dim == 3 && (n == 5 || n == 8))
            std::swap(nodes[2], nodes[3]);
        if (dim == 3 && n == 8)
            std::swap(nodes[6], nodes[7]);
        builder.cell(detail::cell_faces(kind, nodes));
    }
    topology.finish();
}

// Отдельная таблица связывает набор узлов внешней грани с числовой меткой и её именем.
void read_boundaries(detail::Builder& builder, const std::string& dir, bool binary, bool hybrid, U64 dim,
                     U64 nodes_count, U64 boundary_count) {
    const std::map<U64, std::string> names =
        boundary_count ? labels(join(dir, "label.txt")) : std::map<U64, std::string>();
    Array boundary(join(dir, binary ? "bctopo.msb" : "bctopo.msh"), binary, 4);
    if (binary && !hybrid && boundary.declared() != boundary_count * (dim == 2 ? 3 : 5))
        throw MeshError("another: boundary connectivity count mismatch");
    std::set<detail::Nodes> seen;
    for (U64 f = 0; f < boundary_count; ++f) {
        const U64 n = hybrid ? boundary.integer() : (dim == 2 ? 2 : 4);
        if ((dim == 2 && n != 2) || (dim == 3 && (n < 3 || n > 1000000)) || n > nodes_count)
            throw MeshError("another: invalid boundary face size");
        detail::Nodes nodes(static_cast<std::size_t>(n));
        for (MeshIndex& v : nodes) {
            v = boundary.integer();
            if (v >= nodes_count)
                throw MeshError("another: boundary node index out of range");
        }
        const U64 tag = boundary.integer();
        const auto name = names.find(tag);
        if (name == names.end())
            throw MeshError("another: missing name for boundary label " + std::to_string(tag));
        if (!seen.insert(detail::key(nodes)).second)
            throw MeshError("another: duplicate boundary face");
        builder.mark(nodes, name->second);
    }
    boundary.finish();
}
}

Mesh read_another(const std::string& input, const Options& options) {
    const std::string dir = directory(input);
    const Header h = header(join(dir, "mesh.txt"));
    const U64 dim = number(get(h, "numcoords"));
    if (dim != 2 && dim != 3)
        throw MeshError("another: only 2D and 3D meshes are supported");
    const U64 nn = bounded(number(get(h, "num_node", "nn"))), ne = bounded(number(get(h, "num_elem", "nt"))),
              nb = bounded(number(get(h, "num_bndface", "nfacebc")));
    if (!nn || !ne)
        throw MeshError("another: empty volume/area mesh");
    for (const auto& pair : {std::make_pair("num_node_base", nn), std::make_pair("nnbase", nn),
                             std::make_pair("num_elem_base", ne), std::make_pair("ntbase", ne)}) {
        const auto i = h.find(pair.first);
        if (i != h.end() && i->second != "-1" && number(i->second) != pair.second)
            throw MeshError("another: buffer/periodic meshes are not supported; use the ordinary _ntte mesh");
    }
    auto buffer = h.find("bufferzonetype");
    if ((buffer != h.end() && lower(buffer->second) != "no") || exists(join(dir, "periodic03.msh")) ||
        exists(join(dir, "periodic03.msb")))
        throw MeshError("another: buffer/periodic meshes are not supported; use the ordinary _ntte mesh");
    const std::string fmt = lower(get(h, "meshformat"));
    const bool binary = fmt == "binary" || fmt == "1";
    if (!binary && fmt != "text" && fmt != "0")
        throw MeshError("another: unsupported MeshFormat");
    const auto type = h.find("meshtype");
    const bool hybrid = type != h.end() && lower(type->second) == "hybrid";
    if (type != h.end() && !hybrid && lower(type->second) != "hex" && lower(type->second) != "hexa" &&
        lower(type->second) != "quad")
        throw MeshError("another: unsupported MeshType");
    detail::Builder builder(static_cast<unsigned>(dim), options);
    read_coordinates(builder, dir, binary, dim, nn);
    read_cells(builder, dir, binary, hybrid, dim, nn, ne);
    read_boundaries(builder, dir, binary, hybrid, dim, nn, nb);
    // У исходной сетки каждая внешняя грань должна иметь метку. Отсутствующую метку считаем ошибкой.
    Mesh result = builder.finish();
    if (result.view().boundary_face_count != nb)
        throw MeshError("another: incomplete boundary topology");
    return result;
}
}
}
