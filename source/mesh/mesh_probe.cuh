#ifndef AERO_MESH_PROBE_CUH
#define AERO_MESH_PROBE_CUH
#include "mesh.h"
// Compile check only; does not execute on a GPU or transfer host arrays.
extern "C" __global__ void mesh_view_compile_probe(aero::MeshView view, aero::MeshIndex* output) {
    if (view.cell_count && view.face_count) {
        const aero::MeshIndex face = view.cell_faces(0)[0];
        const aero::MeshIndexSpan edges = view.face_edges(face);
        output[0] = view.face_nodes(face)[0];
        if (edges.size) output[0] += view.edge_begin[edges[0]];
        output[1] = view.face_orientation(0, 0) == -1 ? 1 : 0;
        output[2] = view.dimension;
        const aero::MeshCell cell = view.cell(0);
        output[3] = cell.node(0).id() + cell.edge(0).node(1).id();
        output[4] = cell.face(0).node(0).id();
        for (aero::MeshIndex id : cell.edge_ids()) output[3] += view.edge(id).node_count();
        if (cell.face(0).edge_count()) output[4] += cell.face(0).edge(0).id();
    }
}
#endif
