#ifndef AERO_MESH_H
#define AERO_MESH_H

#include "mesh_index.h"
#ifndef __CUDACC_RTC__
#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#endif

#if defined(__CUDACC__)
#define AERO_MESH_HD __host__ __device__
#else
#define AERO_MESH_HD
#endif

namespace aero {

#ifndef __CUDACC_RTC__
enum class MeshFormat { Text, Binary, Lzma };

class MeshError : public std::runtime_error {
public:
    explicit MeshError(const std::string& message) : std::runtime_error(message) {}
};

// Zero-based CSR with half-open [offset[i], offset[i+1]) rows.
struct MeshCsr {
    std::vector<MeshIndex> offsets;
    std::vector<MeshIndex> indices;
    MeshCsr() : offsets(1, 0) {}
    MeshCsr(std::vector<MeshIndex> row_offsets, std::vector<MeshIndex> entries)
        : offsets(std::move(row_offsets)), indices(std::move(entries)) {}
};

struct MeshData {
    // Topological dimension (1, 2 or 3); coordinates always use x, y, z.
    unsigned dimension;
    double length_scale_to_meters;
    std::vector<double> x, y, z;
    // 1D: one vertex; 2D: two directed endpoints; 3D: a cyclic polygon.
    MeshCsr face_nodes;
    // Cells reference shared faces, each stored exactly once.
    MeshCsr cell_faces;
    // Parallel to cell_faces.indices. 1D: -1 at start, +1 at end.
    // 2D: +1 follows the directed edge, -1 reverses it along the cell contour.
    // 3D: +1 uses the face cycle as outward, -1 reverses it.
    std::vector<std::int8_t> cell_face_orientation;
    std::vector<MeshIndex> boundary_faces;
    std::vector<MeshIndex> boundary_patch;
    // Names only; boundary conditions and periodic pairs belong to run setup.
    std::vector<std::string> patch_names;
    MeshData() : dimension(3), length_scale_to_meters(1.0) {}
};

#endif
struct MeshIndexSpan {
    const MeshIndex* data;
    MeshIndex size;
    AERO_MESH_HD MeshIndexSpan(const MeshIndex* values = nullptr, MeshIndex count = 0)
        : data(values), size(count) {}
    AERO_MESH_HD MeshIndex operator[](MeshIndex i) const noexcept { return data[i]; }
    AERO_MESH_HD const MeshIndex* begin() const noexcept { return data; }
    AERO_MESH_HD const MeshIndex* end() const noexcept { return size ? data + size : data; }
};

class MeshNode;
class MeshEdge;
class MeshFace;
class MeshCell;

// Borrowed CPU/CUDA view; value-initialize with MeshView{}. Pointers must
// address memory accessible to the chosen executor. Copying this struct does
// not copy its arrays to a GPU. Access is unchecked; names remain on the host.
struct MeshView {
    unsigned dimension;
    MeshIndex node_count, face_count, cell_count, edge_count, boundary_face_count, patch_count;
    double length_scale_to_meters;
    const double* x;
    const double* y;
    const double* z;
    const MeshIndex* face_offsets;
    const MeshIndex* face_indices;
    const MeshIndex* cell_offsets;
    const MeshIndex* cell_indices;
    const signed char* cell_orientation;
    const MeshIndex* boundary_faces;
    const MeshIndex* boundary_patch;
    const MeshIndex* edge_begin;
    const MeshIndex* edge_end;
    // Separate CSR: 0 edges per face in 1D, 1 in 2D, cyclic edges in 3D.
    const MeshIndex* face_edge_offsets;
    const MeshIndex* face_edge_indices;
    const MeshIndex* cell_node_offsets;
    const MeshIndex* cell_node_indices;
    const MeshIndex* cell_edge_offsets;
    const MeshIndex* cell_edge_indices;

    AERO_MESH_HD MeshNode node(MeshIndex id) const noexcept;
    AERO_MESH_HD MeshEdge edge(MeshIndex id) const noexcept;
    AERO_MESH_HD MeshFace face(MeshIndex id) const noexcept;
    AERO_MESH_HD MeshCell cell(MeshIndex id) const noexcept;

    AERO_MESH_HD MeshIndexSpan face_nodes(MeshIndex face) const noexcept {
        const MeshIndex first = face_offsets[face];
        return MeshIndexSpan(face_indices + first, face_offsets[face + 1] - first);
    }
    AERO_MESH_HD MeshIndexSpan cell_faces(MeshIndex cell) const noexcept {
        const MeshIndex first = cell_offsets[cell];
        return MeshIndexSpan(cell_indices + first, cell_offsets[cell + 1] - first);
    }
    AERO_MESH_HD int face_orientation(MeshIndex cell, MeshIndex local_face) const noexcept {
        return cell_orientation[cell_offsets[cell] + local_face];
    }
    AERO_MESH_HD MeshIndexSpan face_edges(MeshIndex face) const noexcept {
        const MeshIndex first = face_edge_offsets[face];
        const MeshIndex count = face_edge_offsets[face + 1] - first;
        return MeshIndexSpan(count ? face_edge_indices + first : nullptr, count);
    }
    AERO_MESH_HD MeshIndexSpan cell_nodes(MeshIndex cell) const noexcept {
        const MeshIndex first = cell_node_offsets[cell];
        return MeshIndexSpan(cell_node_indices + first, cell_node_offsets[cell + 1] - first);
    }
    AERO_MESH_HD MeshIndexSpan cell_edges(MeshIndex cell) const noexcept {
        const MeshIndex first = cell_edge_offsets[cell];
        return MeshIndexSpan(cell_edge_indices + first, cell_edge_offsets[cell + 1] - first);
    }
};

#ifndef __CUDACC_RTC__
static_assert(std::is_trivially_copyable<MeshView>::value, "MeshView must be trivially copyable");
static_assert(std::is_standard_layout<MeshView>::value, "MeshView must have standard layout");
#endif

// Non-owning, read-only entity handles. They copy the view descriptor, never
// arrays, and are safe to obtain from a temporary mesh.view(). Their lifetime
// is limited by the underlying mesh/device storage, not by the view object.
// All indices below are LOCAL positions; id()/..._id() return GLOBAL IDs.
// Like MeshView, access is unchecked (valid indices required), on CPU and CUDA.
class MeshEntity {
public:
    AERO_MESH_HD MeshIndex id() const noexcept { return id_; }
protected:
    AERO_MESH_HD MeshEntity(MeshView view, MeshIndex id) : view_(view), id_(id) {}
    MeshView view_;
    MeshIndex id_;
};

class MeshNode : public MeshEntity {
public:
    AERO_MESH_HD MeshNode(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    // Stored coordinates, before applying length_scale_to_meters.
    AERO_MESH_HD double x() const noexcept { return view_.x[id_]; }
    AERO_MESH_HD double y() const noexcept { return view_.y[id_]; }
    AERO_MESH_HD double z() const noexcept { return view_.z[id_]; }
};

class MeshEdge : public MeshEntity {
public:
    AERO_MESH_HD MeshEdge(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    AERO_MESH_HD MeshIndex node_count() const noexcept { return 2; }
    // Canonical endpoint order by node ID, not cell/face traversal direction.
    AERO_MESH_HD MeshIndex node_id(MeshIndex local) const noexcept {
        return local == 0 ? view_.edge_begin[id_] : view_.edge_end[id_];
    }
    AERO_MESH_HD MeshNode node(MeshIndex local) const noexcept { return view_.node(node_id(local)); }
};

class MeshFace : public MeshEntity {
public:
    AERO_MESH_HD MeshFace(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    AERO_MESH_HD MeshIndexSpan node_ids() const noexcept { return view_.face_nodes(id_); }
    AERO_MESH_HD MeshIndexSpan edge_ids() const noexcept { return view_.face_edges(id_); }
    AERO_MESH_HD MeshIndex node_count() const noexcept { return node_ids().size; }
    AERO_MESH_HD MeshIndex edge_count() const noexcept { return edge_ids().size; }
    AERO_MESH_HD MeshIndex node_id(MeshIndex local) const noexcept { return node_ids()[local]; }
    AERO_MESH_HD MeshIndex edge_id(MeshIndex local) const noexcept { return edge_ids()[local]; }
    AERO_MESH_HD MeshNode node(MeshIndex local) const noexcept { return view_.node(node_id(local)); }
    AERO_MESH_HD MeshEdge edge(MeshIndex local) const noexcept { return view_.edge(edge_id(local)); }
};

class MeshCell : public MeshEntity {
public:
    AERO_MESH_HD MeshCell(MeshView view, MeshIndex id) : MeshEntity(view, id) {}
    AERO_MESH_HD MeshIndexSpan face_ids() const noexcept { return view_.cell_faces(id_); }
    // Unique nodes/edges sorted by global ID; not a geometric traversal order.
    AERO_MESH_HD MeshIndexSpan node_ids() const noexcept { return view_.cell_nodes(id_); }
    AERO_MESH_HD MeshIndexSpan edge_ids() const noexcept { return view_.cell_edges(id_); }
    AERO_MESH_HD MeshIndex face_count() const noexcept { return face_ids().size; }
    AERO_MESH_HD MeshIndex node_count() const noexcept { return node_ids().size; }
    AERO_MESH_HD MeshIndex edge_count() const noexcept { return edge_ids().size; }
    AERO_MESH_HD MeshIndex face_id(MeshIndex local) const noexcept { return face_ids()[local]; }
    AERO_MESH_HD MeshIndex node_id(MeshIndex local) const noexcept { return node_ids()[local]; }
    AERO_MESH_HD MeshIndex edge_id(MeshIndex local) const noexcept { return edge_ids()[local]; }
    AERO_MESH_HD int face_orientation(MeshIndex local) const noexcept { return view_.face_orientation(id_,local); }
    AERO_MESH_HD MeshFace face(MeshIndex local) const noexcept { return view_.face(face_id(local)); }
    AERO_MESH_HD MeshNode node(MeshIndex local) const noexcept { return view_.node(node_id(local)); }
    AERO_MESH_HD MeshEdge edge(MeshIndex local) const noexcept { return view_.edge(edge_id(local)); }
};

AERO_MESH_HD inline MeshNode MeshView::node(MeshIndex id) const noexcept { return MeshNode(*this,id); }
AERO_MESH_HD inline MeshEdge MeshView::edge(MeshIndex id) const noexcept { return MeshEdge(*this,id); }
AERO_MESH_HD inline MeshFace MeshView::face(MeshIndex id) const noexcept { return MeshFace(*this,id); }
AERO_MESH_HD inline MeshCell MeshView::cell(MeshIndex id) const noexcept { return MeshCell(*this,id); }

#ifndef __CUDACC_RTC__
static_assert(std::is_trivially_copyable<MeshCell>::value, "Entity handles must be trivially copyable");

struct MeshReadLimits {
    // Stored input arrays/strings only; excludes validation and derived connectivity.
    std::uint64_t max_payload_bytes, max_name_bytes;
    MeshReadLimits(std::uint64_t bytes = 1024ULL * 1024ULL * 1024ULL,
                   std::uint64_t name_bytes = 4096)
        : max_payload_bytes(bytes), max_name_bytes(name_bytes) {}
};

// Lzma wraps binary v2 in AEROLZMA blocks; see source/io/README.txt.
// Reading detects all three encodings by signature, independent of extension.
// Native format v2; v1 is explicitly rejected, without implicit conversion.
// TEXT: "AEROMESH 2 TEXT", then DIMENSION, COUNTS, LENGTH_SCALE_TO_METERS,
// X, Y, Z, FACE_OFFSETS, FACE_NODES, CELL_OFFSETS, CELL_FACES, CELL_ORIENTATIONS,
// BOUNDARY_FACES, BOUNDARY_PATCH, PATCH_NAMES, END.
// COUNTS: nodes, faces, face-node entries, cells, cell-face entries, boundary
// faces, patches. Offsets have rows+1 entries. DIMENSION=1,2,3. IDs start at zero.
// All dimensions store all three coordinate arrays. Existing v2 3D files
// remain compatible; no padding or new fields are added to the file layout.
// Whitespace separates tokens; decimal point is always a dot. Names use double
// quotes; a backslash escapes a quote or another backslash.
// BINARY: 8 bytes "AEROMSHB", uint64 version=2, then the same payload without
// labels or END. Integers are little-endian uint64, except orientations: one
// byte 0x01 (+1) or 0xff (-1). Doubles are IEEE-754 binary64, little-endian.
// A name is uint64 byte-length followed by UTF-8 bytes, without a NUL terminator.
// Names are nonempty unique UTF-8 without ASCII control characters. Trailing
// data is rejected in both encodings. Derived edges and cell node/edge lists
// are rebuilt on loading; they do not add fields to the file format.
//
// A 1D cell has two point faces with opposite signs. A 2D cell has a single
// simple oriented contour (at least three edges), including concave polygons.
// A 3D cell boundary is a connected, closed, oriented manifold surface.
// A face has one or two incident cells; the latter use opposite orientations.
// Every exterior face occurs once in the boundary list. Polygon sizes and
// polyhedron face counts have no fixed maximum. Duplicate face cycles/cells,
// repeated vertices, orphan faces, and incomplete boundaries are rejected.
// Geometry validation covers finite coordinates and a positive length scale,
// NOT planarity, zero lengths/areas, signed volumes, intersections or overlap.
// Topological orientation alone does not prove positive volume.
class Mesh {
public:
    Mesh() : face_edge_offsets_(1, 0) {}
    explicit Mesh(MeshData data);
    const MeshData& data() const noexcept { return data_; }
    const std::vector<MeshIndex>& edge_begin() const noexcept { return edge_begin_; }
    const std::vector<MeshIndex>& edge_end() const noexcept { return edge_end_; }
    // Host pointers; cannot outlive or survive replacement of their storage.
    MeshView view() const noexcept;
    // Host entry points check the global ID and throw MeshError on failure.
    // Nested access/local indices remain unchecked, as in MeshView.
    MeshNode node(MeshIndex id) const;
    MeshEdge edge(MeshIndex id) const;
    MeshFace face(MeshIndex id) const;
    MeshCell cell(MeshIndex id) const;
    void validate() const;

    // CPU-only C++11 I/O. Paths use the platform's narrow-string encoding.
    static Mesh read(const std::string& path, const MeshReadLimits& limits = MeshReadLimits());
    static Mesh read(std::istream& input, const MeshReadLimits& limits = MeshReadLimits());
    // Failed I/O can leave a partially written output.
    void write(const std::string& path, MeshFormat format = MeshFormat::Binary) const;
    void write(std::ostream& output, MeshFormat format = MeshFormat::Binary) const;

private:
    void build_edges();
    void build_cell_connectivity();
    MeshData data_;
    std::vector<MeshIndex> edge_begin_, edge_end_, face_edge_offsets_, face_edge_indices_;
    MeshCsr cell_nodes_, cell_edges_;
};

#endif
} // namespace aero

#undef AERO_MESH_HD
#endif
