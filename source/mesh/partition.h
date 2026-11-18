#ifndef AERO_MESH_PARTITION_H
#define AERO_MESH_PARTITION_H
#include "mdec.h"
#include <memory>
namespace aero {
class Processes;
// Leaf ranges follow spatial order; ranks on the same node can be interleaved in MPI_COMM_WORLD.
struct LeafLayout {
    std::vector<MeshIndex> cuts;
    std::vector<int> ranks;
    int owner(MeshIndex leaf) const;
};
// Local numbering follows the global IDs. Every owned node has its complete
// incident cell star. Only edges with an owned lower endpoint are published.
struct MeshPartition {
    Mesh mesh;
    std::string mesh_sha256;
    std::array<MeshIndex,3> global_counts{}; // nodes, faces, cells
    std::vector<MeshIndex> nodes,faces,cells;
    std::vector<MeshIndex> boundary_order; // original boundary table index, or absent at a cut
    std::vector<int> node_owners;
    MeshIndex physical_patches=0;
    MeshIndex first_leaf=0,end_leaf=0;
};
// Retain the decoded core only until the reconstruction rows have been built.
class PartitionInput {
public:
    PartitionInput(const std::string& path,const Processes& world,const LeafLayout& layout);
    MeshPartition core(const Processes& world);
    using NodeOwners=std::vector<std::pair<MeshIndex,int>>;
    void prepare_stencil(const std::set<MeshIndex>& periodic_nodes,bool fifth,NodeOwners* near_vertices);
    void add_stars(const std::set<MeshIndex>& vertices);
    void complete_layer_faces();
    NodeOwners boundary_nodes(std::set<MeshIndex>* incomplete_stars);
    MeshPartition stencil();
private:
    mdec::Archive archive_;
    LeafLayout layout_;
    int slot_;
    mdec::Records records_;
    mdec::Addresses owned_,cells_;
    mdec::Addresses cell_faces(const mdec::Addresses& cells);
    mdec::Addresses face_nodes(const mdec::Addresses& faces);
    MeshPartition indexed_mesh(const mdec::Addresses& cells);
};
MeshPartition load_partition(const std::string& path,const Processes& world,
                            std::unique_ptr<PartitionInput>* stencil_input=nullptr,const LeafLayout& layout={});
}
#endif
