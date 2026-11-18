#include "reconstruction.h"
#include "parallel_stencil.h"
#include <limits>

namespace aero {
namespace {
// Original EBR5: interpolate supporting derivatives through first-face vertices.
detail::SparseRow far_derivative_layer(const detail::ReconstructionGeometry& geometry,
    const detail::RayPoint& near,MeshIndex edge,unsigned side,double h,double& distance) {
    detail::SparseRow row; double step=0; distance=0;
    if (near.weights.empty()) return row;
    for (const auto& w:near.weights) {
        if (w.second==0) continue;
        const detail::RayPoint far=geometry.parallel_star_hit(w.first,edge,side);
        if (far.weights.empty()) return detail::SparseRow();
        detail::add_scaled_row(row,detail::ray_derivative(w.first,far,h,side),w.second);
        step+=w.second*far.distance;
    }
    distance=near.distance+step;
    return row;
}
} // namespace
ReconstructionStencil prepare_ebr(const Mesh& mesh,const ReconstructionOptions& options) {
    mesh.validate();
    const bool fifth_requested=options.scheme==ReconstructionScheme::Ebr5;
    if (options.scheme!=ReconstructionScheme::Ebr3 && options.scheme!=ReconstructionScheme::Ebr5)
        throw ReconstructionError("Invalid reconstruction scheme");
    const MeshView m=mesh.view();
    ReconstructionStencil result;
    ReconstructionStencilData& data=result.data_;
    data.scheme=options.scheme;
    data.node_count=m.node_count; data.edge_count=m.edge_count;
    if (data.edge_count>((std::numeric_limits<ReconstructionIndex>::max)()-1)/2)
        throw ReconstructionError("Too many mesh edges");
    const detail::ReconstructionGeometry geometry(mesh,options.relative_tolerance,true);
    detail::build_edge_rows(data,[&](MeshIndex e,ReconstructionStencilData& rows) {
        const MeshIndex i=m.edge_begin[e],j=m.edge_end[e];
        const double h=geometry.edge_length(e);
        detail::RayPoint hits[2];
        detail::SparseRow near_derivative[2],far_derivative[2];
        double far_distance[2]={0,0};
        const detail::SparseRow edge_difference={{i,-1},{j,1}};
        for (unsigned side=0;side<2;++side) {
            const MeshIndex center=side?j:i;
            hits[side]=geometry.star_hit(e,side);
            if (!hits[side].weights.empty()) near_derivative[side]=detail::ray_derivative(center,hits[side],h,side);
        }
        if (options.scheme==ReconstructionScheme::Ebr5 && !near_derivative[0].empty() && !near_derivative[1].empty()) {
            for (unsigned side=0;side<2;++side)
                far_derivative[side]=far_derivative_layer(geometry,hits[side],e,side,h,far_distance[side]);
        }
        for (unsigned side=0;side<2;++side) {
            const bool near_available=!near_derivative[side].empty();
            const bool fifth=fifth_requested && !far_derivative[side].empty();
            const bool complete=fifth_requested?fifth:near_available;
            if (!complete && options.require_complete_stencil)
                throw ReconstructionError("Incomplete requested EBR stencil at edge "+std::to_string(e)+" side "+std::to_string(side));
            detail::SparseRow row;
            ReconstructionSideScheme actual=ReconstructionSideScheme::EdgeAverage;
            if (fifth) {
                detail::add_scaled_row(row,far_derivative[side],-1.0/15);
                detail::add_scaled_row(row,near_derivative[side],11.0/30);
                detail::add_scaled_row(row,edge_difference,4.0/5);
                detail::add_scaled_row(row,near_derivative[1-side],-1.0/10);
                actual=ReconstructionSideScheme::Ebr5;
            } else if (near_available) {
                detail::add_scaled_row(row,edge_difference,2.0/3);
                detail::add_scaled_row(row,near_derivative[side],1.0/3);
                actual=ReconstructionSideScheme::Ebr3;
            } else row=edge_difference;
            geometry.append_row(rows,row,e,side);
            rows.complete.push_back(complete?1:0);
            rows.side_scheme.push_back(actual);
            rows.outer_distance.push_back(near_available?hits[side].distance:0);
            rows.far_distance.push_back(far_distance[side]);
        }
    });
    return result;
}

} // namespace aero
