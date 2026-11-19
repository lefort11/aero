#include "detail.h"
#ifndef AERO_WITH_CGNS
namespace aero {
namespace converter {
bool cgns_available() noexcept {
    return false;
}
Mesh read_cgns(const std::string&, const Options&) {
    throw MeshError("CGNS support is disabled. Rebuild with WITH_CGNS=1 and the CGNS library.");
}
}
}
#else
#include <cgnslib.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
namespace aero {
namespace converter {
namespace {
void check(int code) {
    if (code != CG_OK)
        throw MeshError(std::string("CGNS: ") + cg_get_error());
}
std::size_t count(cgsize_t n) {
    if (n < 0 || static_cast<std::uint64_t>(n) > 100000000)
        throw MeshError("CGNS: invalid/excessive count");
    return static_cast<std::size_t>(n);
}
struct File {
    int id;
    explicit File(const std::string& path) { check(cg_open(path.c_str(), CG_MODE_READ, &id)); }
    ~File() { cg_close(id); }
};
int kind(CGNS_ENUMT(ElementType_t) type) {
    switch (type) {
    case CGNS_ENUMV(NODE):
        return 15;
    case CGNS_ENUMV(BAR_2):
        return 1;
    case CGNS_ENUMV(TRI_3):
        return 2;
    case CGNS_ENUMV(QUAD_4):
        return 3;
    case CGNS_ENUMV(TETRA_4):
        return 4;
    case CGNS_ENUMV(HEXA_8):
        return 5;
    case CGNS_ENUMV(PENTA_6):
        return 6;
    case CGNS_ENUMV(PYRA_5):
        return 7;
    case CGNS_ENUMV(NGON_n):
        return 100;
    default:
        throw MeshError("CGNS: unsupported/high-order element type " +
                        std::to_string(static_cast<int>(type)));
    }
}
MeshIndex node(cgsize_t id, std::size_t n) {
    if (id < 1 || static_cast<std::uint64_t>(id) > n)
        throw MeshError("CGNS: invalid node reference");
    return static_cast<MeshIndex>(id - 1);
}
struct Record {
    int kind;
    detail::Nodes nodes;
    std::vector<cgsize_t> faces;
};
MeshIndex structured_node(const cgsize_t* ijk, const cgsize_t* shape, int dim) {
    MeshIndex id = 0, stride = 1;
    for (int d = 0; d < dim; ++d) {
        if (ijk[d] < 1 || ijk[d] > shape[d])
            throw MeshError("CGNS: structured point index out of range");
        id += static_cast<MeshIndex>(ijk[d] - 1) * stride;
        stride *= static_cast<MeshIndex>(shape[d]);
    }
    return id;
}
}
bool cgns_available() noexcept {
    return true;
}
namespace {
// Читает выбранную базу и зону CGNS. Идентификаторы CGNS начинаются с единицы.
// Размеры и диапазоны сохраняем один раз для чтения координат, ячеек и меток границы.
class CgnsZone {
    File file;
    int base_id = 0, zone_id = 0; // Номера выбранных базы и зоны внутри файла.
    int dimension = 0, physical_dimension = 0,
        index_dimension = 0;             // Размерности ячеек, координат и индексов CGNS.
    bool structured = false;             // Структурная зона задаёт связность через индексы i, j, k.
    cgsize_t size[9] = {};               // Размеры вершин, ячеек и граничных вершин из заголовка зоны.
    cgsize_t last_vertex[3] = {1, 1, 1}; // Верхний индекс включительно при чтении координат.
    std::size_t node_count = 1,
                cell_count = 1;         // Полные числа узлов и ячеек, проверенные до выделения массивов.
    std::map<cgsize_t, Record> records; // Для неструктурной зоны: метка элемента → вершины или грани.

    // Сначала сверяем размеры. На структурной сетке на каждой активной оси узлов на один больше, чем ячеек.
    void select_zone(const Options& options) {
        int nb = 0;
        check(cg_nbases(file.id, &nb));
        base_id = options.base ? options.base : 1;
        if (base_id < 1 || base_id > nb || (!options.base && nb != 1))
            throw MeshError("CGNS: select a base explicitly with --base");
        char name[33];
        check(cg_base_read(file.id, base_id, name, &dimension, &physical_dimension));
        if (dimension < 1 || dimension > 3 || physical_dimension < dimension || physical_dimension > 3)
            throw MeshError("CGNS: unsupported base dimensions");
        int nz = 0;
        check(cg_nzones(file.id, base_id, &nz));
        zone_id = options.zone ? options.zone : 1;
        if (zone_id < 1 || zone_id > nz || (!options.zone && nz != 1))
            throw MeshError("CGNS: select a zone explicitly with --zone");
        CGNS_ENUMT(ZoneType_t) ztype;
        check(cg_zone_type(file.id, base_id, zone_id, &ztype));
        structured = ztype == CGNS_ENUMV(Structured);
        if (!structured && ztype != CGNS_ENUMV(Unstructured))
            throw MeshError("CGNS: unsupported zone type");
        int connections = 0;
        check(cg_nconns(file.id, base_id, zone_id, &connections));
        int one_to_one = 0;
        check(cg_n1to1(file.id, base_id, zone_id, &one_to_one));
        if (connections || one_to_one)
            throw MeshError("CGNS: inter-zone/periodic connectivity is not yet supported");
        check(cg_zone_read(file.id, base_id, zone_id, name, size));
        index_dimension = structured ? dimension : 1;
        for (int d = 0; d < index_dimension; ++d) {
            const std::size_t n = count(size[d]), c = count(size[index_dimension + d]);
            if (n == 0 || c == 0 || node_count > 100000000 / n || cell_count > 100000000 / c)
                throw MeshError("CGNS: invalid zone size");
            if (structured && c + 1 != n)
                throw MeshError("CGNS: inconsistent structured dimensions");
            node_count *= n;
            cell_count *= c;
            last_vertex[d] = size[d];
        }
    }
    // Названия CoordinateX/Y/Z задают оси независимо от порядка массивов в файле.
    void read_coordinates(detail::Builder& builder) {
        char name[33];
        cgsize_t first_vertex[3] = {1, 1, 1};
        builder.data.x.resize(node_count);
        builder.data.y.resize(node_count);
        builder.data.z.resize(node_count);
        std::vector<double>* coords[] = {&builder.data.x, &builder.data.y, &builder.data.z};
        bool have[3] = {false, false, false};
        int ncoords = 0;
        check(cg_ncoords(file.id, base_id, zone_id, &ncoords));
        for (int c = 1; c <= ncoords; ++c) {
            CGNS_ENUMT(DataType_t) type;
            check(cg_coord_info(file.id, base_id, zone_id, c, &type, name));
            int axis = -1;
            if (std::string(name) == "CoordinateX")
                axis = 0;
            if (std::string(name) == "CoordinateY")
                axis = 1;
            if (std::string(name) == "CoordinateZ")
                axis = 2;
            if (axis >= 0) {
                check(cg_coord_read(file.id, base_id, zone_id, name, CGNS_ENUMV(RealDouble), first_vertex,
                                    last_vertex, coords[axis]->data()));
                have[axis] = true;
            }
        }
        for (int axis = 0; axis < physical_dimension; ++axis)
            if (!have[axis])
                throw MeshError("CGNS: Cartesian CoordinateX/Y/Z arrays required");
    }
    // Восстанавливаем узлы отрезков, четырёхугольников или гексаэдров по шагам строки и слоя.
    void add_structured_cells(detail::Builder& builder) {
        const cgsize_t ni = size[0], nj = dimension >= 2 ? size[1] : 1, nk = dimension >= 3 ? size[2] : 1;
        for (cgsize_t k = 1; k < (dimension >= 3 ? nk : 2); ++k)
            for (cgsize_t j = 1; j < (dimension >= 2 ? nj : 2); ++j)
                for (cgsize_t i = 1; i < ni; ++i) {
                    const cgsize_t p[3] = {i, j, k};
                    const MeshIndex a = structured_node(p, size, dimension), row = static_cast<MeshIndex>(ni),
                                    layer = row * static_cast<MeshIndex>(nj);
                    detail::Nodes nodes;
                    if (dimension == 1)
                        nodes = {a, a + 1};
                    else if (dimension == 2)
                        nodes = {a, a + 1, a + row + 1, a + row};
                    else
                        nodes = {a,         a + 1,         a + row + 1,         a + row,
                                 a + layer, a + layer + 1, a + layer + row + 1, a + layer + row};
                    builder.cell(detail::cell_faces(dimension == 1 ? 1 : (dimension == 2 ? 3 : 5), nodes));
                }
    }
    // Считываем все секции до разрешения ссылок NFACE на многоугольники NGON.
    void read_element_sections() {
        char name[33];
        int ns = 0;
        check(cg_nsections(file.id, base_id, zone_id, &ns));
        for (int s = 1; s <= ns; ++s) {
            CGNS_ENUMT(ElementType_t) type;
            cgsize_t start = 0, end = 0, total = 0;
            int boundary = 0, parent = 0;
            check(
                cg_section_read(file.id, base_id, zone_id, s, name, &type, &start, &end, &boundary, &parent));
            if (start < 1 || end < start)
                throw MeshError("CGNS: invalid section range");
            const std::size_t ne = count(end - start + 1);
            check(cg_ElementDataSize(file.id, base_id, zone_id, s, &total));
            std::vector<cgsize_t> conn(count(total)), offset;
            const bool poly =
                type == CGNS_ENUMV(NGON_n) || type == CGNS_ENUMV(NFACE_n) || type == CGNS_ENUMV(MIXED);
            if (poly) {
                offset.resize(ne + 1);
                check(
                    cg_poly_elements_read(file.id, base_id, zone_id, s, conn.data(), offset.data(), nullptr));
            } else
                check(cg_elements_read(file.id, base_id, zone_id, s, conn.data(), nullptr));
            std::size_t cursor = 0;
            for (std::size_t e = 0; e < ne; ++e) {
                CGNS_ENUMT(ElementType_t) et = type;
                std::size_t last = 0;
                if (poly) {
                    if (offset[e] < 0 || offset[e + 1] < offset[e] ||
                        static_cast<std::uint64_t>(offset[e + 1]) > conn.size() ||
                        static_cast<std::size_t>(offset[e]) != cursor)
                        throw MeshError("CGNS: invalid connectivity offsets");
                    last = static_cast<std::size_t>(offset[e + 1]);
                } else {
                    int npe = 0;
                    check(cg_npe(type, &npe));
                    last = cursor + static_cast<std::size_t>(npe);
                }
                if (last > conn.size() || cursor == last)
                    throw MeshError("CGNS: truncated element connectivity");
                if (type == CGNS_ENUMV(MIXED))
                    et = static_cast<CGNS_ENUMT(ElementType_t)>(conn[cursor++]);
                Record record;
                if (et == CGNS_ENUMV(NFACE_n)) {
                    if (dimension != 3 || type == CGNS_ENUMV(MIXED))
                        throw MeshError("CGNS: unsupported NFACE placement");
                    record.kind = 101;
                    while (cursor < last)
                        record.faces.push_back(conn[cursor++]);
                } else {
                    record.kind = kind(et);
                    while (cursor < last)
                        record.nodes.push_back(node(conn[cursor++], node_count));
                    if (record.kind != 100 && record.nodes.size() != detail::node_count(record.kind))
                        throw MeshError("CGNS: wrong element size");
                }
                if (!records.insert(std::make_pair(start + static_cast<cgsize_t>(e), std::move(record)))
                         .second)
                    throw MeshError("CGNS: overlapping element sections");
            }
            if (cursor != conn.size())
                throw MeshError("CGNS: trailing section connectivity");
        }
    }
    // Знак ссылки NFACE определяет порядок вершин грани в границе данной ячейки.
    void add_unstructured_cells(detail::Builder& builder) {
        std::size_t cells = 0;
        for (const auto& item : records) {
            const Record& rec = item.second;
            if (rec.kind == 101) {
                detail::Faces faces;
                for (cgsize_t signed_id : rec.faces) {
                    if (signed_id == 0 || signed_id == std::numeric_limits<cgsize_t>::min())
                        throw MeshError("CGNS: invalid NFACE reference");
                    auto f = records.find(signed_id < 0 ? -signed_id : signed_id);
                    if (f == records.end() || f->second.kind != 100)
                        throw MeshError("CGNS: NFACE references missing NGON");
                    detail::Nodes nodes = f->second.nodes;
                    if (signed_id < 0)
                        std::reverse(nodes.begin(), nodes.end());
                    faces.push_back(nodes);
                }
                builder.cell(faces);
                ++cells;
            } else if (detail::dimension(rec.kind) == static_cast<unsigned>(dimension)) {
                builder.cell(detail::cell_faces(rec.kind, rec.nodes));
                ++cells;
            }
        }
        if (cells != cell_count)
            throw MeshError("CGNS: zone cell count mismatch; NGON volumes require NFACE connectivity");
    }
    // Граничный участок CGNS может задаваться списком, диапазоном вершин или ссылками на граничные элементы.
    void mark_boundaries(detail::Builder& builder) {
        char name[33];
        int nboco = 0;
        check(cg_nbocos(file.id, base_id, zone_id, &nboco));
        for (int bc = 1; bc <= nboco; ++bc) {
            CGNS_ENUMT(BCType_t) type;
            CGNS_ENUMT(PointSetType_t) settype;
            CGNS_ENUMT(DataType_t) normaltype;
            cgsize_t np = 0, normals = 0;
            int normalindex[3] = {0}, ndataset = 0;
            check(cg_boco_info(file.id, base_id, zone_id, bc, name, &type, &settype, &np, normalindex,
                               &normals, &normaltype, &ndataset));
            const std::string patch = name;
            const std::size_t n = count(np);
            if (n > 100000000 / static_cast<unsigned>(index_dimension))
                throw MeshError("CGNS: boundary too large");
            std::vector<cgsize_t> points(n * static_cast<unsigned>(index_dimension));
            check(cg_boco_read(file.id, base_id, zone_id, bc, points.data(), nullptr));
            CGNS_ENUMT(GridLocation_t) location = CGNS_ENUMV(Vertex);
            check(cg_boco_gridlocation_read(file.id, base_id, zone_id, bc, &location));
            const bool range = settype == CGNS_ENUMV(PointRange) || settype == CGNS_ENUMV(ElementRange);
            const bool list = settype == CGNS_ENUMV(PointList) || settype == CGNS_ENUMV(ElementList);
            if ((!range && !list) || (range && n != 2) || !n)
                throw MeshError("CGNS: unsupported boundary point set");
            const bool vertex = location == CGNS_ENUMV(Vertex) && settype != CGNS_ENUMV(ElementRange) &&
                                settype != CGNS_ENUMV(ElementList);
            if (structured && !vertex)
                throw MeshError("CGNS: structured BC currently requires Vertex location");
            detail::Nodes selected;
            if (range) {
                cgsize_t low[3] = {1, 1, 1}, high[3] = {1, 1, 1};
                std::size_t expanded = 1;
                for (int d = 0; d < index_dimension; ++d) {
                    low[d] = std::min(points[static_cast<std::size_t>(d)],
                                      points[static_cast<std::size_t>(index_dimension + d)]);
                    high[d] = std::max(points[static_cast<std::size_t>(d)],
                                       points[static_cast<std::size_t>(index_dimension + d)]);
                    if (low[d] < 1)
                        throw MeshError("CGNS: invalid boundary range");
                    const std::size_t width = count(high[d] - low[d] + 1);
                    if (width == 0 || expanded > 100000000 / width)
                        throw MeshError("CGNS: boundary range too large");
                    expanded *= width;
                }
                for (cgsize_t k = low[2]; k <= high[2]; ++k)
                    for (cgsize_t j = low[1]; j <= high[1]; ++j)
                        for (cgsize_t i = low[0]; i <= high[0]; ++i) {
                            cgsize_t p[3] = {i, j, k};
                            selected.push_back(
                                structured ? structured_node(p, size, dimension)
                                           : (vertex ? node(i, node_count) : static_cast<MeshIndex>(i)));
                        }
            } else
                for (std::size_t i = 0; i < n; ++i)
                    selected.push_back(
                        structured
                            ? structured_node(&points[i * static_cast<unsigned>(index_dimension)], size,
                                              dimension)
                            : (vertex ? node(points[i], node_count) : static_cast<MeshIndex>(points[i])));
            if (vertex)
                builder.mark_vertices(selected, patch);
            else {
                if (location != CGNS_ENUMV(FaceCenter) && location != CGNS_ENUMV(EdgeCenter) &&
                    settype != CGNS_ENUMV(ElementList) && settype != CGNS_ENUMV(ElementRange))
                    throw MeshError("CGNS: unsupported BC location");
                for (MeshIndex id : selected) {
                    const auto e = records.find(static_cast<cgsize_t>(id));
                    if (e == records.end() || e->second.kind == 101 ||
                        detail::dimension(e->second.kind) + 1 != static_cast<unsigned>(dimension))
                        throw MeshError("CGNS: BC does not reference boundary elements");
                    builder.mark(e->second.nodes, patch);
                }
            }
        }
    }

public:
    CgnsZone(const std::string& path, const Options& options) : file(path) { select_zone(options); }
    Mesh read(const Options& options) {
        detail::Builder builder(static_cast<unsigned>(dimension), options);
        read_coordinates(builder);
        if (structured) {
            add_structured_cells(builder);
        } else {
            read_element_sections();
            add_unstructured_cells(builder);
        }
        mark_boundaries(builder);
        return builder.finish();
    }
};
}
Mesh read_cgns(const std::string& path, const Options& options) {
    return CgnsZone(path, options).read(options);
}
}
}
#endif
