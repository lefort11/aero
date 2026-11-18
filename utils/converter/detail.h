#ifndef AERO_CONVERTER_DETAIL_H
#define AERO_CONVERTER_DETAIL_H
#include "converter.h"
#include <map>
#include <vector>
namespace aero { namespace converter { namespace detail {
typedef std::vector<MeshIndex> Nodes;
typedef std::vector<Nodes> Faces;
struct Element { int kind; Nodes nodes; std::string patch; };
// Kind numbers match linear Gmsh types: 15 point, 1 line, 2 tri, 3 quad,
// 4 tet, 5 hex, 6 prism, 7 pyramid. Kind 100 denotes a polygon.
unsigned dimension(int kind);
std::size_t node_count(int kind);
Faces cell_faces(int kind, const Nodes& nodes);
Nodes key(Nodes nodes);
class Builder {
public:
    MeshData data;
    explicit Builder(unsigned dim, const Options& options);
    void cell(const Faces& faces);
    void mark(const Nodes& nodes, const std::string& name, bool require_boundary = true);
    void mark_vertices(const Nodes& nodes, const std::string& name);
    Mesh finish();
private:
    Options options_;
    std::map<Nodes, MeshIndex> face_ids_;
    std::vector<Nodes> faces_;
    std::vector<unsigned> uses_;
    std::map<MeshIndex, std::string> labels_;
};
} } }
#endif
