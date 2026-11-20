#include "aes.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

namespace aero {
namespace viscosity {
namespace {
Element describe(const MeshPartition& part, const PeriodicPartition* seam, const ResidualFrame& frame,
                 MeshIndex local) {
    Element e{};
    const auto mesh = part.mesh.view();
    const auto vertices = mesh.cell_nodes(local);
    e.id = part.cells.empty() ? local : part.cells[local];
    if (vertices.size > 8) throw MeshError("AES: unsupported cell at " + std::to_string(e.id));
    e.count = static_cast<unsigned>(vertices.size);
    const auto origin = mesh.coordinate(vertices[0]);
    for (unsigned k = 0; k < e.count; ++k) {
        const auto node = vertices[k];
        const auto slot = seam ? seam->node_dof[node] : node;
        e.nodes[k] = seam ? seam->ids[slot] : part.nodes[node];
        e.owners[k] = seam ? seam->owners[slot] : part.node_owners[node];
        const Coordinate dx = mesh.coordinate(node) - origin;
        for (unsigned m = 0; m < mesh.dimension; ++m) {
            const auto a = frame.axis[m];
            e.position[k][m] = (a.x*dx[0] + a.y*dx[1] + a.z*dx[2]) * mesh.length_scale_to_meters;
        }
    }
    for (auto edge : mesh.cell_edges(local)) {
        const auto a = std::find(vertices.begin(), vertices.end(), mesh.edge_begin[edge]) - vertices.begin();
        const auto b = std::find(vertices.begin(), vertices.end(), mesh.edge_end[edge]) - vertices.begin();
        e.neighbours[a] |= 1u << b;
        e.neighbours[b] |= 1u << a;
    }
    return e;
}

}

void Aes::prepare(const MeshPartition& part, const PeriodicPartition* seam, const ResidualFrame& frame,
                  const Processes& world, const ViscositySettings& settings, double gamma, double mach,
                  MeshIndex rows, const Locate& locate) {
    const auto mesh = part.mesh.view();
    op_.dim = mesh.dimension;
    op_.row_count = rows;
    op_.law = {gamma,gamma*mach*mach,1./settings.reynolds,settings.prandtl,
               settings.law == "sutherland" ? settings.sutherland_s/settings.t_amb : 0.};
    std::vector<std::vector<MeshIndex>> by_node;
    world.local([&] { by_node.resize(rows); });
    GradientBuilder gradients;
    const auto add = [&](const Element& e) {
        Cell cell{};
        cell.count = e.count;
        cell.inverse_count = 1. / e.count;
        gradients.prepare(e,mesh.dimension,gamma,op_.law.prandtl,cell,metrics_);
        cell.first_vertex = op_.vertex_count;
        op_.vertex_count += cell.count;
        for (unsigned i = 0; i < e.count; ++i) {
            cell.node[i] = locate(e.nodes[i],e.owners[i]);
            if (e.owners[i] == world.rank) {
                by_node.at(cell.node[i]).push_back(8*cells_.size()+i);
                op_.neighbour_count += e.count-1;
            }
        }
        cells_.push_back(cell);
    };

    if (world.size == 1) {
        for (MeshIndex i = 0; i < mesh.cell_count; ++i) add(describe(part,seam,frame,i));
    } else {
        Processes::Messages send;
        world.local([&] {
            send.resize(world.size);
            for (MeshIndex i = 0; i < mesh.cell_count; ++i) {
                // Единственная исходная вершина назначает отправителя. Это исключает повторное
                // включение приграничного элемента после обмена периодическими представителями.
                if (part.node_owners[mesh.cell_nodes(i)[0]] != world.rank) continue;
                const auto e = describe(part,seam,frame,i);
                const std::set<int> destinations(e.owners,e.owners+e.count);
                for (int p : destinations) send[p].append(reinterpret_cast<const char*>(&e),sizeof(e));
            }
        });
        const auto received = world.exchange(send);
        world.local([&] {
            // Глобальный порядок элементов даёт одинаковый порядок сложения в строке на одном и нескольких процессах.
            std::map<MeshIndex,Element> sorted;
            for (const auto& message : received) {
                if (message.size()%sizeof(Element)) throw MeshError("AES: incomplete MPI element");
                for (size_t k = 0; k < message.size(); k += sizeof(Element)) {
                    Element e;
                    std::memcpy(&e,message.data()+k,sizeof(e));
                    if (!sorted.emplace(e.id,e).second) throw MeshError("AES: duplicate MPI element");
                }
            }
            for (const auto& item : sorted) add(item.second);
        });
    }
    world.local([&] {
        offsets_.push_back(0);
        for (const auto& row : by_node) {
            incidence_.insert(incidence_.end(),row.begin(),row.end());
            offsets_.push_back(incidence_.size());
        }
        op_.cell_count = cells_.size();
        op_.cells = cells_.data();
        op_.stencil.pairs = metrics_.data();
        op_.offsets = offsets_.data();
        op_.incidence = incidence_.data();
        wall.assign(rows,0);
    });
}
void Aes::allocate_host(MeshIndex nodes) {
    gas_.resize(nodes);
    material_.resize(cells_.size());
    rows_.resize(wall.size());
    op_.node_count = nodes;
    op_.gas = gas_.data();
    op_.material = material_.data();
    op_.rows = rows_.data();
}
void Aes::evaluate(const double* state, const EulerState* cached, bool gas_is_current) {
    if (!gas_is_current)
        parallel_for(op_.node_count,[&](MeshIndex node) { op_.prepare_gas(node,state,cached); });
    parallel_for(op_.cell_count,[&](MeshIndex cell) { op_.average_material(cell); });
    parallel_for(op_.row_count,[&](MeshIndex node) { op_.assemble(node); });
}
}
}
