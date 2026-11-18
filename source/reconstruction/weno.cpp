#include "weno.h"
#include "parallel_stencil.h"
#include "reconstruction.h"
#include <cmath>
#include <limits>
#include <map>

namespace aero {
ReconstructionStencil prepare_weno(const Mesh& mesh,const ReconstructionOptions& options) {
    mesh.validate();
    if (options.scheme!=ReconstructionScheme::Weno3 && options.scheme!=ReconstructionScheme::Weno5)
        throw ReconstructionError("prepare_weno requires Weno3 or Weno5");
    if (!(options.weno_epsilon>0) || !std::isfinite(options.weno_epsilon))
        throw ReconstructionError("WENO epsilon must be positive and finite");
    const bool fifth_requested=options.scheme==ReconstructionScheme::Weno5;
    const MeshView m=mesh.view();
    ReconstructionStencil result;
    ReconstructionStencilData& data=result.data_;
    data.scheme=options.scheme;
    data.node_count=m.node_count; data.edge_count=m.edge_count;
    if (data.edge_count>((std::numeric_limits<ReconstructionIndex>::max)()-1)/8)
        throw ReconstructionError("Too many mesh edges");
    const double eps=options.relative_tolerance;
    const detail::ReconstructionGeometry geometry(mesh,eps,false);
    typedef detail::RayPoint Hit;
    typedef detail::SparseRow SparseRow;
    // Paper (2017), section 2: faces whose vertices all belong to N1 or N2.
    // On simplices cell neighbours equal edge neighbours. For general cells
    // include every vertex sharing a cell, so polygon/polyhedron layers close.
    std::vector<std::vector<MeshIndex> > patches(static_cast<std::size_t>(m.node_count));
    std::vector<std::vector<MeshIndex> > second_patches;
    {
        typedef std::map<MeshIndex,bool> NodeSet;
        std::vector<NodeSet> neighbours(static_cast<std::size_t>(m.node_count));
        std::vector<std::vector<MeshIndex> > node_faces(static_cast<std::size_t>(m.node_count));
        for (MeshIndex c=0;c<m.cell_count;++c) {
            const MeshCell cell=m.cell(c);
            for (MeshIndex n:cell.node_ids()) for (MeshIndex q:cell.node_ids())
                if (n!=q) neighbours[static_cast<std::size_t>(n)][q]=true;
        }
        for (MeshIndex f=0;f<m.face_count;++f) for (MeshIndex n:m.face_nodes(f))
            node_faces[static_cast<std::size_t>(n)].push_back(f);
        second_patches.resize(static_cast<std::size_t>(m.node_count));
        parallel_for(m.node_count,[&](MeshIndex n) {
            const NodeSet& first=neighbours[static_cast<std::size_t>(n)];
            NodeSet second;
            if (fifth_requested) for (const auto& q:first)
                for (const auto& r:neighbours[static_cast<std::size_t>(q.first)])
                    if (r.first!=n && first.find(r.first)==first.end()) second[r.first]=true;
            for (unsigned layer=0;layer<(fifth_requested?2u:1u);++layer) {
                const NodeSet& nodes=layer?second:first;
                NodeSet candidates;
                for (const auto& q:nodes) for (MeshIndex f:node_faces[static_cast<std::size_t>(q.first)]) candidates[f]=true;
                for (const auto& candidate:candidates) {
                    bool inside=true;
                    for (MeshIndex q:m.face_nodes(candidate.first)) if (nodes.find(q)==nodes.end()) { inside=false; break; }
                    if (inside) (layer?second_patches:patches)[static_cast<std::size_t>(n)].push_back(candidate.first);
                }
            }
        });
    }
    detail::build_edge_rows(data,[&](MeshIndex e,ReconstructionStencilData& rows) {
        const MeshIndex i=m.edge_begin[e],j=m.edge_end[e];
        const double h=geometry.edge_length(e);
        Hit hits[2];
        SparseRow near_derivative[2],far_derivative[2];
        double far_distance[2]={0,0};
        const SparseRow edge_difference={{i,-1},{j,1}};
        for (unsigned side=0;side<2;++side) {
            const MeshIndex center=side?j:i;
            hits[side]=geometry.layer_hit(e,side,patches[static_cast<std::size_t>(center)]);
            if (!hits[side].weights.empty()) near_derivative[side]=detail::ray_derivative(center,hits[side],h,side);
        }
        if (options.scheme==ReconstructionScheme::Weno5 && !near_derivative[0].empty() && !near_derivative[1].empty()) {
            for (unsigned side=0;side<2;++side) {
                const MeshIndex center=side?j:i;
                const Hit far=geometry.layer_hit(e,side,second_patches[static_cast<std::size_t>(center)]);
                const double step=far.distance-hits[side].distance;
                if (!far.weights.empty() && step>eps*h) {
                    const double factor=(side?1.0:-1.0)*(h/step);
                    detail::add_scaled_row(far_derivative[side],far.weights,factor);
                    detail::add_scaled_row(far_derivative[side],hits[side].weights,-factor);
                    far_distance[side]=far.distance;
                }
            }
        }
        for (unsigned side=0;side<2;++side) {
            const bool near_available=!near_derivative[side].empty();
            const bool fifth=fifth_requested && !far_derivative[side].empty();
            const bool complete=fifth_requested?fifth:near_available;
            if (!complete && options.require_complete_stencil)
                throw ReconstructionError("Incomplete requested EBR stencil at edge "+std::to_string(e)+" side "+std::to_string(side));
            SparseRow row;
            ReconstructionSideScheme actual=ReconstructionSideScheme::EdgeAverage;
            if (fifth) {
                detail::add_scaled_row(row,far_derivative[side],-1.0/15);
                detail::add_scaled_row(row,near_derivative[side],11.0/30);
                detail::add_scaled_row(row,edge_difference,4.0/5);
                detail::add_scaled_row(row,near_derivative[1-side],-1.0/10);
                actual=ReconstructionSideScheme::Weno5;
            } else if (near_available) {
                detail::add_scaled_row(row,edge_difference,2.0/3);
                detail::add_scaled_row(row,near_derivative[side],1.0/3);
                actual=ReconstructionSideScheme::Weno3;
            } else row=edge_difference;
            geometry.append_row(rows,row,e,side);
            rows.complete.push_back(complete?1:0);
            rows.side_scheme.push_back(actual);
            rows.outer_distance.push_back(near_available?hits[side].distance:0);
            rows.far_distance.push_back(far_distance[side]);
            {
                const unsigned char order=fifth?5:(near_available?3:0);
                rows.weno_order.push_back(order);
                const double epsilon_scale=h*std::sqrt(options.weno_epsilon);
                if (!std::isfinite(epsilon_scale)) throw ReconstructionError("WENO epsilon/edge scale overflow");
                rows.weno_epsilon_scale.push_back(epsilon_scale);
                const SparseRow* slopes[4]={&edge_difference,&near_derivative[side],&near_derivative[1-side],&far_derivative[side]};
                for (unsigned k=0;k<4;++k) {
                    if (order && (k<2 || order==5)) for (const auto& w:*slopes[k]) {
                        if (!std::isfinite(w.second)) throw ReconstructionError("Non-finite WENO derivative weight");
                        if (w.second!=0) { rows.slope_nodes.push_back(w.first); rows.slope_weights.push_back(w.second); }
                    }
                    rows.slope_offsets.push_back(static_cast<ReconstructionIndex>(rows.slope_nodes.size()));
                }
            }
        }
    });
    return result;
}

} // namespace aero
