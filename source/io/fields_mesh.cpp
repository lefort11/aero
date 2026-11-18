#include "fields.h"
#include "sha256.h"
#include "../mesh/mesh.h"
#include <ostream>

namespace aero {
namespace {
std::array<std::uint64_t, 4> mesh_counts(const Mesh& mesh) {
    const MeshView view = mesh.view();
    return {{view.node_count, view.edge_count, view.face_count, view.cell_count}};
}
} // namespace

FieldSnapshot::FieldSnapshot(const Mesh& mesh, std::uint64_t step_number, double time_seconds)
    : dimension(mesh.data().dimension), counts(mesh_counts(mesh)), mesh_sha256(mesh_fingerprint(mesh)),
      step(step_number), time(time_seconds) {
    validate();
}

std::string FieldSnapshot::mesh_fingerprint(const Mesh& mesh) {
    io_detail::Sha256Buffer buffer;
    std::ostream out(&buffer);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    mesh.write(out, MeshFormat::Binary);
    return buffer.hex_digest();
}

void FieldSnapshot::check_mesh(const Mesh& mesh) const {
    validate();
    if (dimension != mesh.data().dimension || counts != mesh_counts(mesh))
        throw FieldError("Field mesh dimension/counts mismatch");
    if (mesh_sha256 != mesh_fingerprint(mesh))
        throw FieldError("Field mesh SHA-256 mismatch");
}
} // namespace aero
