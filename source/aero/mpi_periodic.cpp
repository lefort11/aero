#include "mpi_periodic_detail.h"

namespace aero {
using namespace periodic_mpi;
namespace {
struct EdgeData { Key key; Ref a,b; Point area; };
struct Member { EdgeData edge; Point direction; };
struct Volume { Ref node; Id original; double value; };
struct Piece { Key order; Ref node; Id patch; Point area; };

void put_edge(std::string& s,const EdgeData& e) { put(s,e.key); put(s,e.a); put(s,e.b); put(s,e.area); }
EdgeData read_edge(Reader& r) { return {r.key(),r.ref(),r.ref(),r.point()}; }

std::vector<Ref> node_representatives(const MeshPartition& part,const Labels& labels,const Processes& w) {
    std::vector<Ref> result;
    Messages replies;
    exchange(w,[&](Messages& out) {
        result.resize(part.nodes.size());
        replies.resize(w.size);
        for (Id i=0;i<part.nodes.size();++i) {
            const Ref n=vertex(part,i);
            if (n.rank==w.rank) result[i]=representative(labels,n);
            else { put(out[n.rank],i); put(out[n.rank],n.key); }
        }
    },[&](Reader& r,int peer) {
        const Id i=r.id(); const Key key=r.key();
        put(replies[peer],i); put(replies[peer],representative(labels,{key,w.rank}));
    });
    exchange(w,[&](Messages& out) { out.swap(replies); },[&](Reader& r,int) {
        const Id i=r.id(); result.at(i)=r.ref();
    });
    return result;
}

std::vector<EdgeData> collect_edges(const MeshPartition& part,CvView cv,const Labels& labels,
                                    const std::vector<Ref>& nodes,const Processes& w) {
    const auto m=part.mesh.view();
    std::map<Key,std::vector<Member>> groups;
    std::vector<EdgeData> incident;
    exchange(w,[&](Messages& out) {
        for (Id e=0;e<m.edge_count;++e) {
            const Id a=m.edge_begin[e],b=m.edge_end[e];
            if (part.node_owners[a]!=w.rank) continue;
            const Key key={part.nodes[a],part.nodes[b]};
            const auto it=labels.find(key);
            if (it==labels.end()) continue;
            auto& s=out[it->second.rank];
            put(s,it->second.key); put_edge(s,{key,nodes[a],nodes[b],point(cv.edge_areas[e])});
            put(s,sub(position(m,b),position(m,a)));
        }
    },[&](Reader& r,int) {
        const Key root=r.key();
        groups[root].push_back({read_edge(r),r.point()});
    });
    exchange(w,[&](Messages& out) {
        const auto deliver=[&](const EdgeData& e) {
            if (e.a.rank==w.rank || e.b.rank==w.rank) incident.push_back(e);
            if (e.a.rank!=w.rank) put_edge(out[e.a.rank],e);
            if (e.b.rank!=w.rank && e.a.rank!=e.b.rank) put_edge(out[e.b.rank],e);
        };
        for (Id e=0;e<m.edge_count;++e) {
            const Id a=m.edge_begin[e],b=m.edge_end[e];
            const Key key={part.nodes[a],part.nodes[b]};
            if (part.node_owners[a]==w.rank && !labels.count(key))
                deliver({key,nodes[a],nodes[b],point(cv.edge_areas[e])});
        }
        for (auto& kv:groups) {
            auto& members=kv.second;
            std::sort(members.begin(),members.end(),[](const Member& a,const Member& b) { return a.edge.key<b.edge.key; });
            EdgeData e=members.front().edge;
            if (e.key!=kv.first) throw MeshError("periodic: missing representative edge");
            e.area={{0,0,0}};
            const Point dir=members.front().direction;
            for (const auto& member:members) {
                // Endpoint labels alone cannot orient an edge that wraps to itself.
                const bool forward=norm(sub(member.direction,dir))<=norm(add(member.direction,dir));
                e.area=forward?add(e.area,member.edge.area):sub(e.area,member.edge.area);
            }
            deliver(e);
        }
    },[&](Reader& r,int) { incident.push_back(read_edge(r)); });
    return incident;
}
}

PeriodicPartition partition_periodic(const MeshPartition& part,CvView cv,const std::vector<PeriodicPair>& pairs,const Processes& w) {
    const auto labels=match_periodic(part,cv,pairs,w);
    const auto nodes=node_representatives(part,labels,w);
    auto edges=collect_edges(part,cv,labels,nodes,w);
    std::vector<Volume> volumes;
    exchange(w,[&](Messages& out) {
        for (Id i=0;i<nodes.size();++i) if (part.node_owners[i]==w.rank) {
            const Volume v={nodes[i],part.nodes[i],cv.volumes[i]};
            if (v.node.rank==w.rank) volumes.push_back(v);
            else {
                auto& s=out[v.node.rank]; put(s,v.node); put(s,v.original); put_real(s,v.value);
            }
        }
    },[&](Reader& r,int) { volumes.push_back({r.ref(),r.id(),r.real()}); });

    std::vector<Piece> boundary;
    exchange(w,[&](Messages& out) {
        std::set<std::string> paired;
        for (const auto& p:pairs) { paired.insert(p.patch); paired.insert(p.pair); }
        const auto m=part.mesh.view();
        Id k=0;
        for (Id b=0;b<m.boundary_face_count;++b) {
            const Id patch=m.boundary_patch[b];
            const auto corners=m.face_nodes(m.boundary_faces[b]);
            if (patch<part.physical_patches && !paired.count(part.mesh.data().patch_names[patch]))
                for (Id j=0;j<corners.size;++j) {
                    const Id n=corners[j];
                    if (part.node_owners[n]!=w.rank) continue;
                    const Piece piece={{part.boundary_order[b],j},nodes[n],patch,point(cv.boundary[k+j].area)};
                    if (piece.node.rank==w.rank) boundary.push_back(piece);
                    else {
                        auto& s=out[piece.node.rank];
                        put(s,piece.order); put(s,piece.node); put(s,patch); put(s,piece.area);
                    }
                }
            k+=corners.size;
        }
    },[&](Reader& r,int) { boundary.push_back({r.key(),r.ref(),r.id(),r.point()}); });

    PeriodicPartition grid;
    w.local([&] {
        std::map<Id,int> needed;
        for (Ref n:nodes) needed.emplace(n.key.first,n.rank);
        for (const auto& e:edges) { needed.emplace(e.a.key.first,e.a.rank); needed.emplace(e.b.key.first,e.b.rank); }
        for (const auto& n:needed) { grid.ids.push_back(n.first); grid.owners.push_back(n.second); }
        const auto local=[&](Ref n)->Id {
            const auto it=std::lower_bound(grid.ids.begin(),grid.ids.end(),n.key.first);
            if (it==grid.ids.end() || *it!=n.key.first) throw MeshError("periodic: missing local node");
            return static_cast<Id>(it-grid.ids.begin());
        };
        for (Ref n:nodes) grid.node_dof.push_back(local(n));
        grid.volumes.resize(grid.ids.size());
        // Match the serial accumulation order independently of message arrival.
        std::sort(volumes.begin(),volumes.end(),[](const Volume& a,const Volume& b) { return a.original<b.original; });
        for (const auto& v:volumes) grid.volumes[local(v.node)]+=v.value;
        std::sort(edges.begin(),edges.end(),[](const EdgeData& a,const EdgeData& b) { return a.key<b.key; });
        for (const auto& e:edges) grid.edges.push_back({local(e.a),local(e.b),vector(e.area),e.key});
        std::sort(boundary.begin(),boundary.end(),[](const Piece& a,const Piece& b) { return a.order<b.order; });
        for (const auto& p:boundary) grid.boundary.push_back({local(p.node),p.order.first,p.patch,vector(p.area)});
    });
    return grid;
}
}

