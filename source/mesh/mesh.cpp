#include "mesh.h"
#include "node_incidence.h"
#include "../parallel/parallel_for.h"
#include "mesh_reader.h"
#include <memory>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <utility>

namespace aero {
namespace {

typedef MeshIndex Index;
typedef std::pair<Index, Index> Edge;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "The mesh format requires IEEE-754 binary64");

[[noreturn]] void fail(const std::string& message) {
    throw MeshError(message);
}

std::size_t rows(const MeshCsr& csr) {
    return csr.offsets.empty() ? 0 : csr.offsets.size() - 1;
}

void validate_csr(const MeshCsr& csr, std::size_t target_count, const char* name) {
    if (csr.offsets.empty())
        fail(std::string(name) + ": missing offsets");
    if (csr.offsets.front() != 0 || csr.offsets.back() != csr.indices.size())
        fail(std::string(name) + ": incorrect first/last offset");
    for (std::size_t i = 1; i < csr.offsets.size(); ++i)
        if (csr.offsets[i - 1] > csr.offsets[i] || csr.offsets[i] > csr.indices.size())
            fail(std::string(name) + ": invalid CSR offsets");
    for (const Index index : csr.indices)
        if (index >= target_count)
            fail(std::string(name) + ": index out of range");
}

std::vector<Index> sorted_row(const MeshCsr& csr, std::size_t row) {
    const std::size_t first = static_cast<std::size_t>(csr.offsets[row]);
    const std::size_t last = static_cast<std::size_t>(csr.offsets[row + 1]);
    std::vector<Index> result(csr.indices.begin() + static_cast<std::ptrdiff_t>(first),
                              csr.indices.begin() + static_cast<std::ptrdiff_t>(last));
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
        fail("Repeated entry within a face or cell");
    return result;
}

// Приводим цикл многоугольника к одной записи независимо от поворота и направления обхода.
// Число вершин заранее не ограничиваем.
std::vector<Index> canonical_face(const MeshCsr& faces, std::size_t face) {
    const Index* begin = faces.indices.data() + faces.offsets[face];
    const std::size_t n = static_cast<std::size_t>(faces.offsets[face + 1] - faces.offsets[face]);
    const std::size_t start = static_cast<std::size_t>(std::min_element(begin, begin + n) - begin);
    const bool forward = begin[(start + 1) % n] < begin[(start + n - 1) % n];
    std::vector<Index> result;
    result.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        result.push_back(begin[forward ? (start + k) % n : (start + n - k) % n]);
    return result;
}

class DisjointSet {
public:
    explicit DisjointSet(std::size_t count) : parent_(count) {
        std::iota(parent_.begin(), parent_.end(), std::size_t(0));
    }
    std::size_t root(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    void join(std::size_t a, std::size_t b) { parent_[root(a)] = root(b); }
    bool connected() {
        if (parent_.empty())
            return true;
        const std::size_t first = root(0);
        for (std::size_t i = 1; i < parent_.size(); ++i)
            if (root(i) != first)
                return false;
        return true;
    }

private:
    std::vector<std::size_t> parent_;
};

struct EdgeUse {
    Edge edge;
    std::size_t face;
    int direction;
    bool operator<(const EdgeUse& other) const { return edge < other.edge; }
};

void validate_cell_shell(const MeshData& d, std::size_t cell) {
    const Index first = d.cell_faces.offsets[cell], last = d.cell_faces.offsets[cell + 1];
    // Соседние ячейки одного потока переиспользуют память для проверки оболочки.
    thread_local std::vector<EdgeUse> edges;
    edges.clear();
    for (Index entry = first; entry < last; ++entry) {
        const Index face = d.cell_faces.indices[entry];
        const Index a = d.face_nodes.offsets[face], b = d.face_nodes.offsets[face + 1];
        for (Index k = a; k < b; ++k) {
            const Index v = d.face_nodes.indices[k], w = d.face_nodes.indices[k + 1 == b ? a : k + 1];
            EdgeUse use;
            use.edge = std::make_pair(std::min(v, w), std::max(v, w));
            use.face = static_cast<std::size_t>(entry - first);
            use.direction = (v < w ? 1 : -1) * d.cell_face_orientation[entry];
            edges.push_back(use);
        }
    }
    std::sort(edges.begin(), edges.end());
    DisjointSet shell(static_cast<std::size_t>(last - first));
    typedef std::pair<std::size_t, std::size_t> Link;
    using VertexLink = std::pair<Index,Link>;
    thread_local std::vector<VertexLink> vertex_links;
    vertex_links.clear();
    for (std::size_t i = 0; i < edges.size();) {
        std::size_t end = i + 1;
        while (end < edges.size() && edges[end].edge == edges[i].edge)
            ++end;
        if (end - i != 2)
            fail("Cell boundary is open or non-manifold at an edge");
        if (edges[i].direction + edges[i + 1].direction != 0)
            fail("Cell face orientations do not form a closed oriented surface");
        const Link link(edges[i].face, edges[i + 1].face);
        shell.join(link.first, link.second);
        vertex_links.emplace_back(edges[i].edge.first,link);
        vertex_links.emplace_back(edges[i].edge.second,link);
        i = end;
    }
    if (!shell.connected())
        fail("Cell boundary has disconnected shells");
    // Проверяем связность кольца вокруг вершины, чтобы обнаружить две оболочки, соединённые только этой вершиной.
    std::sort(vertex_links.begin(),vertex_links.end());
    for (auto begin=vertex_links.begin(); begin!=vertex_links.end();) {
        const auto first_link=begin++;
        while (begin!=vertex_links.end() && begin->first==first_link->first) ++begin;
        // Проверка рёбер выше гарантирует степень два у каждой грани в кольце.
        // Петля на одной грани невозможна (вершины грани различны). Поэтому
        // два отдельных цикла требуют минимум четырёх связей в этой вершине.
        if (begin-first_link<=3) continue;
        std::vector<std::size_t> incident;
        for (auto it=first_link; it!=begin; ++it) {
            const Link& link = it->second;
            incident.push_back(link.first);
            incident.push_back(link.second);
        }
        std::sort(incident.begin(), incident.end());
        incident.erase(std::unique(incident.begin(), incident.end()), incident.end());
        DisjointSet link_graph(incident.size());
        for (auto it=first_link; it!=begin; ++it) {
            const Link& link = it->second;
            const std::size_t a = static_cast<std::size_t>(
                std::lower_bound(incident.begin(), incident.end(), link.first) - incident.begin());
            const std::size_t b = static_cast<std::size_t>(
                std::lower_bound(incident.begin(), incident.end(), link.second) - incident.begin());
            link_graph.join(a, b);
        }
        if (!link_graph.connected())
            fail("Cell boundary is non-manifold at a vertex");
    }
}

void validate_cell_boundary(const MeshData& d, std::size_t cell) {
    const Index first = d.cell_faces.offsets[cell], last = d.cell_faces.offsets[cell + 1];
    if (d.dimension == 3) {
        validate_cell_shell(d, cell);
        return;
    }
    if (d.dimension == 1) {
        if (d.cell_face_orientation[first] + d.cell_face_orientation[first + 1] != 0)
            fail("Segment endpoints must have opposite orientations");
        return;
    }
    // У каждой вершины контура должно быть ровно по одному входящему и выходящему ребру.
    // Контур должен быть связным при любом порядке граней в исходной таблице.
    std::map<Index, Index> next;
    std::set<Index> incoming;
    for (Index k = first; k < last; ++k) {
        const Index offset = d.face_nodes.offsets[d.cell_faces.indices[k]];
        Index a = d.face_nodes.indices[offset], b = d.face_nodes.indices[offset + 1];
        if (d.cell_face_orientation[k] == -1)
            std::swap(a, b);
        if (!next.insert(std::make_pair(a, b)).second || !incoming.insert(b).second)
            fail("Polygon contour branches or has inconsistent orientations");
    }
    for (const auto& edge : next)
        if (incoming.count(edge.first) == 0)
            fail("Polygon contour is open");
    const Index start = next.begin()->first;
    Index vertex = start;
    std::size_t visited = 0;
    do {
        vertex = next.find(vertex)->second;
        ++visited;
    } while (vertex != start);
    if (visited != next.size())
        fail("Polygon has disconnected contours");
}

// Равные строки имеют одинаковый минимальный номер, в том числе после
// перестановки элементов. Они всегда попадают в одну группу. Группы
// можно проверять независимо, без блокировки общего множества ключей.
template<class Key>
void check_unique_rows(const MeshCsr& csr, const Key& key, const char* duplicate) {
    const auto count = rows(csr);
    if (!count) return;
    std::vector<std::vector<std::size_t>> groups(1 + (count - 1) / 1024);
    for (std::size_t i = 0; i < count; ++i) {
        const auto begin = csr.indices.begin() + csr.offsets[i];
        const auto end = csr.indices.begin() + csr.offsets[i + 1];
        // Пустая строка попадёт в штатную проверку числа вершин или граней.
        const Index minimum = begin == end ? 0 : *std::min_element(begin, end);
        groups[minimum % groups.size()].push_back(i);
    }
    parallel_for(groups.size(), [&](std::size_t g) {
        std::set<std::vector<Index>> seen;
        for (const auto row : groups[g])
            if (!seen.insert(key(row)).second) fail(duplicate);
    }, 2);
}

// Одна геометрическая грань имеет один номер, даже при обратном обходе или другом начале цикла.
void validate_unique_faces(const MeshData& d) {
    check_unique_rows(d.face_nodes, [&](std::size_t f) {
        const Index count = d.face_nodes.offsets[f + 1] - d.face_nodes.offsets[f];
        if ((d.dimension < 3 && count != d.dimension) || (d.dimension == 3 && count < 3))
            fail("Face vertex count is invalid for the mesh dimension");
        sorted_row(d.face_nodes, f);
        return canonical_face(d.face_nodes, f);
    }, "Duplicate face cycle; shared faces must use one face ID");
}

// Считает, сколько ячеек используют каждую грань. На общей грани знаки двух ячеек взаимно уничтожаются.
std::vector<unsigned char> count_incident_cells(const MeshData& d) {
    const auto face_count = rows(d.face_nodes), cell_count = rows(d.cell_faces);
    std::vector<unsigned char> incidence(face_count, 0);
    std::vector<int> orientation_sum(face_count, 0);
    check_unique_rows(d.cell_faces, [&](std::size_t c) {
        const Index count = d.cell_faces.offsets[c + 1] - d.cell_faces.offsets[c];
        if ((d.dimension == 1 && count != 2) || count < d.dimension + 1)
            fail("Cell face count is invalid for the mesh dimension");
        return sorted_row(d.cell_faces, c);
    }, "Duplicate cell");
    // Внутренняя грань встречается в двух ячейках. Накапливаем её
    // ориентированную сумму за один линейный проход без атомарных операций.
    for (std::size_t c = 0; c < cell_count; ++c) {
        for (Index k = d.cell_faces.offsets[c]; k < d.cell_faces.offsets[c + 1]; ++k) {
            const Index f = d.cell_faces.indices[k];
            const int sign = d.cell_face_orientation[k];
            if (sign != 1 && sign != -1)
                fail("Cell face orientation must be +1 or -1");
            if (++incidence[f] > 2)
                fail("A face has more than two incident cells");
            orientation_sum[f] += sign;
        }
    }
    for (std::size_t f = 0; f < face_count; ++f) {
        if (incidence[f] == 0)
            fail("Orphan face without an incident cell");
        if (incidence[f] == 2 && orientation_sum[f] != 0)
            fail("Shared face must have opposite orientations in its two cells");
    }

    return incidence;
}

// Внешняя грань имеет одну соседнюю ячейку и ровно одну метку граничного участка.
void validate_exterior_faces(const MeshData& d, const std::vector<unsigned char>& incidence) {
    const auto face_count = incidence.size();
    std::vector<bool> boundary(face_count, false);
    for (std::size_t b = 0; b < d.boundary_faces.size(); ++b) {
        const Index f = d.boundary_faces[b];
        if (f >= face_count)
            fail("Boundary face index out of range");
        if (boundary[f])
            fail("Duplicate boundary face");
        if (incidence[f] != 1)
            fail("An internal face cannot be a boundary face");
        if (d.boundary_patch[b] >= d.patch_names.size())
            fail("Boundary patch index out of range");
        boundary[f] = true;
    }
    for (std::size_t f = 0; f < face_count; ++f)
        if (incidence[f] == 1 && !boundary[f])
            fail("Exterior boundary is incomplete");
}

// Проверки идут от отдельных граней к ячейкам и затем к границе всей области.
void validate_topology(const MeshData& d) {
    validate_unique_faces(d);
    const auto incidence = count_incident_cells(d);
    // Проверка оболочки использует только грани данной ячейки и локальные
    // рабочие массивы. Все ячейки читают одну и ту же неизменяемую сетку.
    parallel_for(rows(d.cell_faces), [&](std::size_t cell) {
        validate_cell_boundary(d, cell);
    });
    validate_exterior_faces(d, incidence);
}

class Writer {
public:
    Writer(std::ostream& output, MeshFormat format, Index version) : output_(output), text_(format == MeshFormat::Text) {
        if (format != MeshFormat::Text && format != MeshFormat::Binary)
            fail("Unsupported mesh encoding");
        numeric_.imbue(std::locale::classic());
        numeric_ << std::setprecision(std::numeric_limits<double>::max_digits10);
        output_.write(text_ ? "AEROMESH" : "AEROMSHB", 8);
        if (text_)
            output_.put(' ');
        u64(version);
        if (text_)
            literal("TEXT\n");
    }
    void block(const char* name) {
        if (text_) {
            output_.put('\n');
            literal(name);
            output_.put('\n');
        }
    }
    void u64(std::uint64_t value) { number_codec::write_word(output_, value, text_); }
    void orientation(std::int8_t value) {
        if (text_)
            literal(value == 1 ? "1 " : "-1 ");
        else
            output_.put(static_cast<char>(value == 1 ? 1 : 255));
    }
    void real(double value) {
        if (text_) {
            numeric_.clear();
            numeric_.str(std::string());
            numeric_ << value;
            const std::string text = numeric_.str();
            output_.write(text.data(), static_cast<std::streamsize>(text.size()));
            output_.put(' ');
        } else {
            u64(number_codec::real_bits(value));
        }
    }
    // Запись крупными блоками применяется также при вычислении SHA-256 сетки.
    // Для текста и машин с другим порядком байтов остаётся обычный кодек v2.
    template<class T, class WriteOne>
    void array(const char* name, const std::vector<T>& values, WriteOne one) {
        block(name);
        if (!text_ && (sizeof(T) == 1 || number_codec::native_little_endian)) {
            if (!values.empty())
                output_.write(reinterpret_cast<const char*>(values.data()),
                              static_cast<std::streamsize>(values.size() * sizeof(T)));
        } else {
            for (T value : values) (this->*one)(value);
        }
    }
    void name(const std::string& value) {
        io::NativeFileCodec::write_string(output_,text_,value,'\n',fail,"Boundary name is too large to write");
    }
    void finish() {
        block("END");
        if (!output_)
            fail("Mesh output failed");
    }

private:
    void literal(const char* text) { output_.write(text, static_cast<std::streamsize>(std::strlen(text))); }
    std::ostream& output_;
    bool text_;
    std::ostringstream numeric_;
};

// Порядок массивов — часть формата сетки и отпечатка её полей. Один обход
// используется при записи и чтении; способ передачи данных задаёт архив.
template<class Archive, class Data> void transfer_mesh_arrays(Archive& archive, Data& data) {
    archive.array("X", data.x, &Archive::real);
    archive.array("Y", data.y, &Archive::real);
    archive.array("Z", data.z, &Archive::real);
    archive.array("FACE_OFFSETS", data.face_nodes.offsets, &Archive::u64);
    archive.array("FACE_NODES", data.face_nodes.indices, &Archive::u64);
    archive.array("CELL_OFFSETS", data.cell_faces.offsets, &Archive::u64);
    archive.array("CELL_FACES", data.cell_faces.indices, &Archive::u64);
    archive.array("CELL_ORIENTATIONS", data.cell_face_orientation, &Archive::orientation);
    archive.array("BOUNDARY_FACES", data.boundary_faces, &Archive::u64);
    archive.array("BOUNDARY_PATCH", data.boundary_patch, &Archive::u64);
}

void write_payload(const MeshData& d, std::ostream& output, MeshFormat format,
                   const std::uint64_t* fingerprint_word = nullptr) {
    Writer writer(output, format, fingerprint_word ? mesh_format::legacy_version : mesh_format::current_version);
    writer.block("DIMENSION");
    writer.u64(d.dimension);
    writer.block("COUNTS");
    const std::size_t counts[] = {
        d.x.size(),          rows(d.face_nodes),          d.face_nodes.indices.size(),
        rows(d.cell_faces),  d.cell_faces.indices.size(), d.boundary_faces.size(),
        d.patch_names.size()};
    for (const std::size_t count : counts)
        writer.u64(count);
    // Это слово требуется только хеширующему потоку формата полей v1.
    if (fingerprint_word)
        writer.u64(*fingerprint_word);
    transfer_mesh_arrays(writer, d);
    writer.block("PATCH_NAMES");
    for (const std::string& name : d.patch_names)
        writer.name(name);
    writer.finish();
}

} // namespace

Mesh::Mesh(MeshData data) : data_(std::move(data)) {
    validate();
    build_edges();
    build_cell_connectivity();
}

void Mesh::build_edges() {
    if (data_.dimension == 1) {
        build_line_edges();
        return;
    }
    const auto node_offsets = group_edges_by_node();
    number_face_edges(node_offsets);
}

void Mesh::build_line_edges() {
    const auto m = view();
    // В одном измерении каждая ячейка уже является отдельным ребром.
    face_edge_offsets_.assign(m.face_count + 1,0);
    std::vector<Edge> edges(m.cell_count);
    parallel_for(m.cell_count, [&](Index c) {
        const Index k = data_.cell_faces.offsets[c];
        const Index a = data_.face_nodes.indices[data_.face_nodes.offsets[data_.cell_faces.indices[k]]];
        const Index b =
            data_.face_nodes.indices[data_.face_nodes.offsets[data_.cell_faces.indices[k + 1]]];
        edges[c] = std::minmax(a,b);
    });
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    edge_begin_.resize(edges.size());
    edge_end_.resize(edges.size());
    parallel_for(edges.size(), [&](Index e) { edge_begin_[e]=edges[e].first; edge_end_[e]=edges[e].second; });
}

namespace {
// Примыкающее к узлу ребро ведёт к предыдущей или следующей вершине грани.
// Возвращаем только большие номера: ребро будет включено ровно у одного из концов.
void collect_edge_neighbours(MeshView m, Index n, MeshIndexSpan incident_faces, std::vector<Index>& neighbours) {
    neighbours.clear();
    for (Index f : incident_faces) {
        const auto v = m.face_nodes(f);
        const auto pos = std::find(v.begin(),v.end(),n)-v.begin();
        const Index j = static_cast<Index>(pos);
        const Index next = v[j+1 == v.size ? 0 : j+1];
        if (next>n) neighbours.push_back(next);
        if (v.size>2) {
            const Index prev = v[j ? j-1 : v.size-1];
            if (prev>n) neighbours.push_back(prev);
        }
    }
    std::sort(neighbours.begin(),neighbours.end());
    neighbours.erase(std::unique(neighbours.begin(), neighbours.end()), neighbours.end());
}
}

std::vector<MeshIndex> Mesh::group_edges_by_node() {
    const auto m = view();
    // Ребро принадлежит меньшему из двух номеров. Каждый поток сортирует только
    // соседей своей вершины, сохраняя исходную лексикографическую нумерацию рёбер.
    const NodeIncidence faces(m.node_count,m.face_count,[&](Index f){ return m.face_nodes(f); });
    const Index block_size = 1024, blocks = (m.node_count+block_size-1)/block_size;
    std::vector<std::vector<Index>> parts(blocks);
    std::vector<Index> offsets(m.node_count+1);
    parallel_for(blocks, [&](Index b) {
        std::vector<Index> neighbours;
        for (Index n=b*block_size; n<std::min(m.node_count,(b+1)*block_size); ++n) {
            collect_edge_neighbours(m, n, faces[n], neighbours);
            offsets[n+1] = neighbours.size();
            parts[b].insert(parts[b].end(),neighbours.begin(),neighbours.end());
        }
    },2);
    std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
    edge_begin_.resize(offsets.back());
    edge_end_.resize(offsets.back());
    parallel_for(blocks, [&](Index b) {
        const Index first=b*block_size;
        std::copy(parts[b].begin(),parts[b].end(),edge_end_.begin()+offsets[first]);
        for (Index n=first; n<std::min(m.node_count,first+block_size); ++n)
            std::fill(edge_begin_.begin()+offsets[n],edge_begin_.begin()+offsets[n+1],n);
    },2);
    return offsets;
}

void Mesh::number_face_edges(const std::vector<MeshIndex>& offsets) {
    const auto m = view();
    // В 3D на каждую вершину контура приходится одно ребро, в 2D грань сама является ребром.
    if (m.dimension==3) face_edge_offsets_=data_.face_nodes.offsets;
    else {
        face_edge_offsets_.resize(m.face_count+1);
        std::iota(face_edge_offsets_.begin(),face_edge_offsets_.end(),Index(0));
    }
    face_edge_indices_.resize(face_edge_offsets_.back());
    parallel_for(m.face_count,[&](Index f) {
        const auto v=m.face_nodes(f);
        const Index begin=face_edge_offsets_[f], count=face_edge_offsets_[f+1]-begin;
        for (Index j=0;j<count;++j) {
            const Index a=v[j], b=v[j+1==v.size ? 0 : j+1], owner=std::min(a,b);
            const auto found=std::lower_bound(edge_end_.begin()+offsets[owner],edge_end_.begin()+offsets[owner+1],std::max(a,b));
            face_edge_indices_[begin+j]=static_cast<Index>(found-edge_end_.begin());
        }
    });
}

// Две грани одной ячейки могут содержать один узел или ребро. Сортировка
// и удаление повторов задают одинаковую нумерацию при любом порядке обхода.
void Mesh::collect_cell_connectivity(Index c, std::vector<Index>& nodes, std::vector<Index>& edges) const {
    nodes.clear();
    edges.clear();
    for (Index k = data_.cell_faces.offsets[c]; k < data_.cell_faces.offsets[c + 1]; ++k) {
        const Index f = data_.cell_faces.indices[k];
        for (Index j = data_.face_nodes.offsets[f]; j < data_.face_nodes.offsets[f + 1]; ++j)
            nodes.push_back(data_.face_nodes.indices[j]);
        for (Index j = face_edge_offsets_[f]; j < face_edge_offsets_[f + 1]; ++j)
            edges.push_back(face_edge_indices_[j]);
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    if (data_.dimension == 1) {
        // Рёбра отсортированы лексикографически по упорядоченным парам номеров концов.
        const auto first = std::lower_bound(edge_begin_.begin(), edge_begin_.end(), nodes[0]);
        const auto last = std::upper_bound(first, edge_begin_.end(), nodes[0]);
        const std::ptrdiff_t a = first - edge_begin_.begin(), b = last - edge_begin_.begin();
        const auto edge = std::lower_bound(edge_end_.begin() + a, edge_end_.begin() + b, nodes[1]);
        edges.push_back(static_cast<Index>(edge - edge_end_.begin()));
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
}

void Mesh::build_cell_connectivity() {
    // Независимые блоки формируют строки в прежнем порядке. Один рабочий
    // буфер блока обслуживает много ячеек, в том числе произвольные многогранники.
    const Index count = rows(data_.cell_faces), block_size = 1024;
    const Index blocks = (count + block_size - 1) / block_size;
    struct Block { std::vector<Index> nodes, edges; };
    std::vector<Block> parts(blocks);
    cell_nodes_.offsets.resize(count + 1);
    cell_edges_.offsets.resize(count + 1);
    parallel_for(blocks, [&](Index part) {
        auto& block = parts[part];
        std::vector<Index> nodes, edges;
        const Index end = std::min(count, (part + 1) * block_size);
        for (Index c = part * block_size; c < end; ++c) {
            collect_cell_connectivity(c, nodes, edges);
            cell_nodes_.offsets[c+1] = nodes.size();
            cell_edges_.offsets[c+1] = edges.size();
            block.nodes.insert(block.nodes.end(), nodes.begin(), nodes.end());
            block.edges.insert(block.edges.end(), edges.begin(), edges.end());
        }
    }, 2);
    // Смещения разделяют готовые строки. Разные потоки пишут в непересекающиеся участки.
    for (MeshCsr* table : {&cell_nodes_, &cell_edges_}) {
        std::partial_sum(table->offsets.begin(), table->offsets.end(), table->offsets.begin());
        table->indices.resize(table->offsets.back());
    }
    parallel_for(blocks, [&](Index part) {
        const Index first = part * block_size;
        const auto& block = parts[part];
        std::copy(block.nodes.begin(), block.nodes.end(), cell_nodes_.indices.begin() + cell_nodes_.offsets[first]);
        std::copy(block.edges.begin(), block.edges.end(), cell_edges_.indices.begin() + cell_edges_.offsets[first]);
    }, 2);
}

MeshNode Mesh::node(MeshIndex id) const {
    if (id >= data_.x.size())
        fail("Node index out of range");
    return view().node(id);
}
MeshEdge Mesh::edge(MeshIndex id) const {
    if (id >= edge_begin_.size())
        fail("Edge index out of range");
    return view().edge(id);
}
MeshFace Mesh::face(MeshIndex id) const {
    if (id >= rows(data_.face_nodes))
        fail("Face index out of range");
    return view().face(id);
}
MeshCell Mesh::cell(MeshIndex id) const {
    if (id >= rows(data_.cell_faces))
        fail("Cell index out of range");
    return view().cell(id);
}

MeshView Mesh::view() const noexcept {
    MeshView v = {};
    v.dimension = data_.dimension;
    v.node_count = data_.x.size();
    v.face_count = rows(data_.face_nodes);
    v.cell_count = rows(data_.cell_faces);
    v.edge_count = edge_begin_.size();
    v.boundary_face_count = data_.boundary_faces.size();
    v.patch_count = data_.patch_names.size();
    v.x = data_.x.data();
    v.y = data_.y.data();
    v.z = data_.z.data();
    v.face_offsets = data_.face_nodes.offsets.data();
    v.face_indices = data_.face_nodes.indices.data();
    v.cell_offsets = data_.cell_faces.offsets.data();
    v.cell_indices = data_.cell_faces.indices.data();
    v.cell_orientation = data_.cell_face_orientation.data();
    v.boundary_faces = data_.boundary_faces.data();
    v.boundary_patch = data_.boundary_patch.data();
    v.edge_begin = edge_begin_.data();
    v.edge_end = edge_end_.data();
    v.face_edge_indices = face_edge_indices_.data();
    v.face_edge_offsets = face_edge_offsets_.data();
    v.cell_node_offsets = cell_nodes_.offsets.data();
    v.cell_node_indices = cell_nodes_.indices.data();
    v.cell_edge_offsets = cell_edges_.offsets.data();
    v.cell_edge_indices = cell_edges_.indices.data();
    return v;
}

void Mesh::validate() const {
    const MeshData& d = data_;
    if (d.dimension < 1 || d.dimension > 3)
        fail("Mesh dimension must be 1, 2 or 3");
    if (d.x.size() != d.y.size() || d.x.size() != d.z.size())
        fail("Coordinate arrays have different sizes");
    parallel_for(d.x.size(), [&](std::size_t i) {
        if (!std::isfinite(d.x[i]) || !std::isfinite(d.y[i]) || !std::isfinite(d.z[i]))
            fail("Coordinates must be finite");
    });
    validate_csr(d.face_nodes, d.x.size(), "face_nodes");
    validate_csr(d.cell_faces, rows(d.face_nodes), "cell_faces");
    if (d.cell_face_orientation.size() != d.cell_faces.indices.size())
        fail("Cell orientation array size mismatch");
    if (d.boundary_faces.size() != d.boundary_patch.size())
        fail("Boundary arrays have different sizes");
    std::set<std::string> names;
    for (const std::string& name : d.patch_names) {
        if (!mesh_input::valid_name(name))
            fail("Boundary name must be nonempty UTF-8 without control characters");
        if (!names.insert(name).second)
            fail("Duplicate boundary name");
    }
    validate_topology(d);
}

Mesh Mesh::read(const std::string& path, const MeshReadLimits& limits) {
    if (path.find('\0') != std::string::npos)
        fail("NUL in mesh path");
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input)
        fail("Cannot open mesh for reading: " + path);
    return read(input, limits);
}

Mesh Mesh::read(std::istream& input, const MeshReadLimits& limits) try {
    mesh_input::Reader reader(input, limits);
    reader.block("DIMENSION");
    const Index dimension = reader.u64();
    if (dimension < 1 || dimension > 3)
        fail("Mesh dimension must be 1, 2 or 3");
    reader.block("COUNTS");
    std::array<Index, 7> counts = {};
    for (Index& count : counts)
        count = reader.u64();
    if (counts[1] == std::numeric_limits<Index>::max() || counts[3] == std::numeric_limits<Index>::max())
        fail("CSR row count overflows");
    MeshData d;
    d.dimension = static_cast<unsigned>(dimension);
    reader.resize(d.x, counts[0]);
    reader.resize(d.y, counts[0]);
    reader.resize(d.z, counts[0]);
    reader.resize(d.face_nodes.offsets, counts[1] + 1);
    reader.resize(d.face_nodes.indices, counts[2]);
    reader.resize(d.cell_faces.offsets, counts[3] + 1);
    reader.resize(d.cell_faces.indices, counts[4]);
    reader.resize(d.cell_face_orientation, counts[4]);
    reader.resize(d.boundary_faces, counts[5]);
    reader.resize(d.boundary_patch, counts[5]);
    reader.resize(d.patch_names, counts[6]);
    const auto fingerprint_word = reader.read_fingerprint_word();
    transfer_mesh_arrays(reader, d);
    reader.block("PATCH_NAMES");
    for (std::string& name : d.patch_names)
        name = reader.name();
    reader.finish();
    Mesh mesh(std::move(d));
    mesh.fingerprint_word_ = fingerprint_word;
    return mesh;
} catch (const io::CompressionError& error) {
    throw MeshError(error.what());
}

namespace {
void write_encoded(const MeshData& data, std::ostream& output, MeshFormat format) {
    try {
        if (format == MeshFormat::Lzma) {
            io::LzmaOutputBuffer buffer(output);
            std::ostream compressed(&buffer);
            compressed.exceptions(std::ios::badbit | std::ios::failbit);
            write_payload(data, compressed, MeshFormat::Binary);
            buffer.finish();
        } else {
            write_payload(data, output, format);
        }
    } catch (const io::CompressionError& error) {
        throw MeshError(error.what());
    }
}
}

void Mesh::write(const std::string& path, MeshFormat format) const {
    if (format != MeshFormat::Text && format != MeshFormat::Binary && format != MeshFormat::Lzma)
        fail("Unsupported mesh encoding");
    if (path.find('\0') != std::string::npos)
        fail("NUL in mesh path");
    std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!output)
        fail("Cannot open mesh for writing: " + path);
    write_encoded(data_, output, format);
    output.close();
    if (!output)
        fail("Cannot finish writing mesh: " + path);
}

void Mesh::write(std::ostream& output, MeshFormat format) const {
    // Исходная связность проверена в конструкторе и доступна только для чтения.
    // В частности, вычисление SHA-256 не требует новой проверки всех оболочек.
    write_encoded(data_, output, format);
}

void Mesh::write_fingerprint_payload(std::ostream& output) const {
    write_payload(data_, output, MeshFormat::Binary, &fingerprint_word_);
}

} // namespace aero
