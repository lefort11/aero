#include "detail.h"
#include "../../source/parallel/parallel_for.h"
#include <algorithm>
#include <set>
namespace aero {
namespace converter {
namespace detail {
unsigned dimension(int k) {
    switch (k) {
    case 15:
        return 0;
    case 1:
        return 1;
    case 2:
    case 3:
    case 100:
        return 2;
    case 4:
    case 5:
    case 6:
    case 7:
        return 3;
    default:
        throw MeshError("Unsupported element type: " + std::to_string(k));
    }
}
std::size_t node_count(int k) {
    switch (k) {
    case 15:
        return 1;
    case 1:
        return 2;
    case 2:
        return 3;
    case 3:
    case 4:
        return 4;
    case 5:
        return 8;
    case 6:
        return 6;
    case 7:
        return 5;
    default:
        throw MeshError("Unsupported/high-order element type: " + std::to_string(k));
    }
}
Nodes key(Nodes n) {
    std::sort(n.begin(), n.end());
    if (n.empty() || std::adjacent_find(n.begin(), n.end()) != n.end())
        throw MeshError("Empty or repeated face vertices");
    return n;
}
Faces cell_faces(int kind, const Nodes& n) {
    if (kind != 100 && n.size() != node_count(kind))
        throw MeshError("Wrong element node count");
    Faces f;
    if (kind == 1)
        return {{n[0]}, {n[1]}};
    if (dimension(kind) == 2) {
        if (n.size() < 3)
            throw MeshError("Polygon needs at least three nodes");
        for (std::size_t i = 0; i < n.size(); ++i)
            f.push_back({n[i], n[(i + 1) % n.size()]});
        return f;
    }
    std::vector<std::vector<unsigned>> local;
    if (kind == 4)
        local = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
    else if (kind == 5)
        local = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    else if (kind == 6)
        local = {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
    else if (kind == 7)
        local = {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
    else
        throw MeshError("Unsupported cell type");
    for (const auto& row : local) {
        Nodes face;
        for (unsigned i : row)
            face.push_back(n[i]);
        f.push_back(face);
    }
    return f;
}
static int orientation(const Nodes& stored, const Nodes& current) {
    if (stored.size() == 1)
        return 1;
    if (stored.size() == 2)
        return stored[0] == current[0] ? 1 : -1;
    const std::size_t n = stored.size();
    const std::size_t start =
        static_cast<std::size_t>(std::find(stored.begin(), stored.end(), current[0]) - stored.begin());
    bool forward = true, backward = true;
    for (std::size_t k = 0; k < n; ++k) {
        forward = forward && stored[(start + k) % n] == current[k];
        backward = backward && stored[(start + n - k) % n] == current[k];
    }
    if (!forward && !backward)
        throw MeshError("Incompatible face cycles with identical vertices");
    return forward ? 1 : -1;
}
// Собирает единую сетку AEROMESH из элементов внешнего формата, сохраняя масштаб входных координат.
Builder::Builder(unsigned dim, const Options& options) : options_(options) {
    data.dimension = dim;
    data.length_scale_to_meters = options.length_scale_to_meters;
}
void Builder::cell(const Faces& faces) {
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const Nodes& n = faces[i];
        const Nodes k = key(n);
        for (MeshIndex v : n)
            if (v >= data.x.size())
                throw MeshError("Cell node index out of range");
        auto it = face_ids_.find(k);
        MeshIndex id;
        if (it == face_ids_.end()) {
            id = faces_.size();
            face_ids_[k] = id;
            faces_.push_back(n);
            uses_.push_back(0);
            data.face_nodes.indices.insert(data.face_nodes.indices.end(), n.begin(), n.end());
            data.face_nodes.offsets.push_back(data.face_nodes.indices.size());
        } else
            id = it->second;
        ++uses_[static_cast<std::size_t>(id)];
        int sign = orientation(faces_[static_cast<std::size_t>(id)], n);
        if (data.dimension == 1)
            sign = i == 0 ? -1 : 1;
        data.cell_faces.indices.push_back(id);
        data.cell_face_orientation.push_back(static_cast<std::int8_t>(sign));
    }
    data.cell_faces.offsets.push_back(data.cell_faces.indices.size());
}
void Builder::mark(const Nodes& nodes, const std::string& name, bool required) {
    const auto it = face_ids_.find(key(nodes));
    if (it == face_ids_.end() || uses_[static_cast<std::size_t>(it->second)] != 1) {
        if (required)
            throw MeshError("Named boundary is not an exterior cell face: " + name);
        return;
    }
    orientation(faces_[static_cast<std::size_t>(it->second)], nodes);
    if (name.empty())
        return;
    const auto old = labels_.find(it->second);
    if (old != labels_.end() && old->second != name)
        throw MeshError("Several boundary names assigned to one face");
    labels_[it->second] = name;
}
void Builder::mark_vertices(const Nodes& nodes, const std::string& name) {
    std::set<MeshIndex> selected(nodes.begin(), nodes.end());
    for (MeshIndex n : nodes)
        if (n >= data.x.size())
            throw MeshError("Boundary node index out of range");
    bool matched = false;
    std::vector<unsigned char> covered(faces_.size());
    parallel_for(faces_.size(), [&](std::size_t f) {
        if (uses_[f] == 1)
            covered[f] = std::all_of(faces_[f].begin(), faces_[f].end(),
                                     [&](MeshIndex n) { return selected.count(n) != 0; });
    });
    for (std::size_t f = 0; f < faces_.size(); ++f)
        if (covered[f]) {
            mark(faces_[f], name);
            matched = true;
        }
    if (!matched)
        throw MeshError("Vertex boundary does not cover any complete exterior face: " + name);
}
// Проверяет полноту граничных меток и передаёт готовые таблицы конструктору Mesh.
Mesh Builder::finish() {
    if (data.cell_faces.offsets.size() == 1)
        throw MeshError("Input contains no cells");
    std::map<std::string, MeshIndex> patches;
    for (std::size_t f = 0; f < faces_.size(); ++f)
        if (uses_[f] == 1) {
            const auto it = labels_.find(f);
            const std::string name = it == labels_.end() ? options_.unmarked_name : it->second;
            if (name.empty())
                throw MeshError("Exterior face has no boundary name");
            auto p = patches.find(name);
            if (p == patches.end()) {
                MeshIndex id = data.patch_names.size();
                data.patch_names.push_back(name);
                p = patches.insert(std::make_pair(name, id)).first;
            }
            data.boundary_faces.push_back(f);
            data.boundary_patch.push_back(p->second);
        }
    return Mesh(std::move(data));
}
}
}
}
