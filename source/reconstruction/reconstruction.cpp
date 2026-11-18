#include "reconstruction.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>

namespace aero {
detail::RayPoint::RayPoint()
    : distance((std::numeric_limits<double>::infinity)()),outward(false) {}
namespace {
struct Vec {
    double x,y,z;
    Vec(double a=0,double b=0,double c=0) : x(a),y(b),z(c) {}
};
Vec operator+(Vec a,Vec b) { return Vec(a.x+b.x,a.y+b.y,a.z+b.z); }
Vec operator-(Vec a,Vec b) { return Vec(a.x-b.x,a.y-b.y,a.z-b.z); }
Vec operator*(Vec a,double t) { return Vec(a.x*t,a.y*t,a.z*t); }
Vec operator/(Vec a,double t) { return a*(1.0/t); }
double dot(Vec a,Vec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec cross(Vec a,Vec b) { return Vec(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
double norm(Vec a) { return std::hypot(std::hypot(a.x,a.y),a.z); }
Vec point(MeshView m,MeshIndex n) { return Vec(m.x[n],m.y[n],m.z[n]); }
typedef std::array<MeshIndex,3> Triangle;
struct Surface { Vec normal; std::vector<Triangle> triangles; };

std::size_t checked_size(ReconstructionIndex count,ReconstructionIndex components) {
    const ReconstructionIndex limit=static_cast<ReconstructionIndex>((std::numeric_limits<std::size_t>::max)()/sizeof(double));
    if (!components || (count && components>limit/count))
        throw ReconstructionError("Invalid component count or array size overflow");
    return static_cast<std::size_t>(count*components);
}
double turn(Vec a,Vec b,Vec c,Vec normal) { return dot(cross(b-a,c-a),normal); }

// Ear clipping is a geometric auxiliary construction, not a change of mesh
// connectivity. It works for simple planar concave as well as convex faces.
Surface triangulate(MeshView mesh,const std::vector<MeshIndex>& ids,double eps) {
    Surface surface;
    if (ids.size()<3) throw ReconstructionError("Polygon has fewer than three nodes");
    const Vec origin=point(mesh,ids[0]);
    double scale=0;
    for (MeshIndex id:ids) scale=(std::max)(scale,norm(point(mesh,id)-origin));
    if (!(scale>0) || !std::isfinite(scale)) throw ReconstructionError("Degenerate polygon scale");
    std::vector<Vec> p;
    for (MeshIndex id:ids) p.push_back((point(mesh,id)-origin)/scale);
    Vec area;
    for (std::size_t i=0;i<p.size();++i) area=area+cross(p[i],p[(i+1)%p.size()]);
    if (!(norm(area)>eps)) throw ReconstructionError("Degenerate or self-intersecting polygon");
    surface.normal=area/norm(area);
    for (Vec v:p) if (std::abs(dot(v,surface.normal))>eps)
        throw ReconstructionError("EBR requires planar polygon faces/cells");
    const std::size_t n=p.size();
    for (std::size_t i=0;i<n;++i) {
        if (norm(p[(i+1)%n]-p[i])<=eps) throw ReconstructionError("Zero or unresolved polygon edge");
        for (std::size_t j=i+1;j<n;++j) {
            if (j==i+1 || (i==0 && j==n-1)) continue;
            const Vec a=p[i],b=p[(i+1)%n],c=p[j],d=p[(j+1)%n];
            const double ac=turn(a,b,c,surface.normal),ad=turn(a,b,d,surface.normal);
            const double ca=turn(c,d,a,surface.normal),cb=turn(c,d,b,surface.normal);
            if (((ac>eps && ad< -eps)||(ac< -eps && ad>eps)) &&
                ((ca>eps && cb< -eps)||(ca< -eps && cb>eps)))
                throw ReconstructionError("Self-intersecting polygon");
            const Vec ends[4]={a,b,c,d};
            for (unsigned k=0;k<4;++k) {
                const Vec q=ends[k],r=ends[k<2?2:0],t=ends[k<2?3:1];
                if (std::abs(turn(r,t,q,surface.normal))<=eps &&
                    dot(q-r,q-t)<=eps*eps)
                    throw ReconstructionError("Touching or overlapping polygon edges");
            }
        }
    }
    std::vector<std::size_t> active;
    for (std::size_t i=0;i<n;++i) active.push_back(i);
    while (active.size()>3) {
        bool clipped=false;
        for (std::size_t j=0;j<active.size();++j) {
            const std::size_t a=active[(j+active.size()-1)%active.size()], b=active[j],
                              c=active[(j+1)%active.size()];
            if (turn(p[a],p[b],p[c],surface.normal)<=eps) continue;
            bool contains=false;
            for (std::size_t k:active) if (k!=a && k!=b && k!=c) {
                if (turn(p[a],p[b],p[k],surface.normal)>=-eps &&
                    turn(p[b],p[c],p[k],surface.normal)>=-eps &&
                    turn(p[c],p[a],p[k],surface.normal)>=-eps) { contains=true; break; }
            }
            if (contains) continue;
            surface.triangles.push_back(Triangle{{ids[a],ids[b],ids[c]}});
            active.erase(active.begin()+static_cast<std::ptrdiff_t>(j));
            clipped=true; break;
        }
        if (!clipped) throw ReconstructionError("Cannot triangulate polygon at selected tolerance");
    }
    if (turn(p[active[0]],p[active[1]],p[active[2]],surface.normal)<=eps)
        throw ReconstructionError("Degenerate final polygon triangle");
    surface.triangles.push_back(Triangle{{ids[active[0]],ids[active[1]],ids[active[2]]}});
    return surface;
}

std::vector<MeshIndex> cell_cycle(MeshCell cell) {
    std::map<MeshIndex,MeshIndex> next;
    for (MeshIndex i=0;i<cell.face_count();++i) {
        const MeshFace face=cell.face(i);
        const bool forward=cell.face_orientation(i)>0;
        next[face.node_id(forward?0:1)]=face.node_id(forward?1:0);
    }
    std::vector<MeshIndex> result;
    MeshIndex id=next.begin()->first;
    for (std::size_t i=0;i<next.size();++i) { result.push_back(id); id=next.at(id); }
    return result;
}

struct PatchFace { MeshIndex face,cell; int orientation; unsigned count; };
typedef detail::RayPoint Hit;
void consider(Hit& best,double distance,bool outward,const std::map<ReconstructionIndex,double>& weights,
              double tolerance,bool farthest=false) {
    if (!(distance>tolerance) || !std::isfinite(distance)) return;
    if ((farthest ? distance>best.distance+tolerance : distance<best.distance-tolerance) ||
        (std::abs(distance-best.distance)<=tolerance && outward && !best.outward)) {
        best.distance=distance; best.outward=outward; best.weights=weights;
    }
}

// Intersect a ray with the boundary of the union of cells incident to its
// origin. The nearest crossing must be outward: never jump through a hole or
// outside a concave domain to manufacture an outer point.
Hit outer_point(MeshView m,MeshIndex node,Vec direction,double h,
                const std::vector<PatchFace>& boundary,const std::vector<Surface>& faces,
                const std::vector<Vec>& cell_normal,double eps,bool farthest=false) {
    const Vec origin=point(m,node);
    const double tolerance=eps*h;
    Hit hit;
    if (farthest) hit.distance=0;
    for (const PatchFace& entry:boundary) {
        if (entry.count!=1) continue;
        const MeshFace face=m.face(entry.face);
        if (m.dimension==1) {
            const MeshIndex id=face.node_id(0);
            const Vec v=point(m,id)-origin;
            const double t=dot(v,direction);
            if (norm(v-direction*t)<=eps*(std::max)(h,norm(v)))
                consider(hit,t,true,{{static_cast<ReconstructionIndex>(id),1}},tolerance,farthest);
        } else if (m.dimension==2) {
            const MeshIndex a=face.node_id(0),b=face.node_id(1);
            const Vec edge=point(m,b)-point(m,a),from=point(m,a)-origin;
            const Vec normal=cell_normal[static_cast<std::size_t>(entry.cell)];
            const Vec cross_de=cross(direction,edge);
            const double det=dot(cross_de,normal);
            if (std::abs(det)<=eps*norm(edge)) continue;
            const double t=dot(cross(from,edge),normal)/det;
            double u=dot(cross(from,direction),normal)/det;
            if (u< -eps || u>1+eps) continue;
            if (norm(direction*t-(from+edge*u))>eps*(std::max)(h,norm(from)+norm(edge))) continue;
            u=(std::max)(0.0,(std::min)(1.0,u));
            const Vec outward=cross(edge,normal)*static_cast<double>(entry.orientation);
            consider(hit,t,dot(direction,outward)>0,
                     {{static_cast<ReconstructionIndex>(a),1-u},{static_cast<ReconstructionIndex>(b),u}},tolerance,farthest);
        } else {
            const Surface& surface=faces[static_cast<std::size_t>(entry.face)];
            for (const Triangle& tri:surface.triangles) {
                // Normalize locally to avoid squared/cubed mesh units in tests.
                const Vec a=point(m,tri[0])-origin;
                const Vec b=point(m,tri[1])-origin,c=point(m,tri[2])-origin;
                const double scale=(std::max)(norm(b-a),norm(c-a));
                const Vec e1=(b-a)/scale,e2=(c-a)/scale,r=a/scale;
                const Vec p=cross(direction,e2);
                const double det=dot(e1,p);
                if (std::abs(det)<=eps*norm(e1)*norm(e2)) continue;
                double u=dot(r*(-1),p)/det;
                const Vec q=cross(r*(-1),e1);
                double v=dot(direction,q)/det;
                const double t=dot(e2,q)/det*scale;
                if (u< -eps || v< -eps || u+v>1+eps) continue;
                u=(std::max)(0.0,u); v=(std::max)(0.0,v);
                if (u+v>1) { const double sum=u+v; u/=sum; v/=sum; }
                consider(hit,t,dot(direction,surface.normal)*entry.orientation>0,
                         {{static_cast<ReconstructionIndex>(tri[0]),1-u-v},
                          {static_cast<ReconstructionIndex>(tri[1]),u},{static_cast<ReconstructionIndex>(tri[2]),v}},tolerance,farthest);
            }
        }
    }
    if (!farthest && !hit.outward) hit.weights.clear();
    return hit;
}

typedef std::map<ReconstructionIndex,double> SparseRow;
void add_row(SparseRow& destination,const SparseRow& source,double factor) {
    for (const auto& w:source) destination[w.first]+=factor*w.second;
}
// h times the directional derivative in the fixed i -> j direction.
// On simplices this is the projected P1 gradient of the supporting element.
SparseRow forward_derivative(MeshIndex node,const Hit& hit,double h,unsigned side) {
    SparseRow row;
    const double a=(side?1.0:-1.0)*(h/hit.distance);
    row[node]=-a;
    for (const auto& w:hit.weights) row[w.first]+=a*w.second;
    return row;
}

} // namespace

ReconstructionStencilView ReconstructionStencil::view() const {
    const ReconstructionStencilView result={data_.node_count,data_.edge_count,data_.offsets.data(),
        data_.nodes.data(),data_.row_owner.data(),data_.weights.data(),data_.slope_offsets.data(),data_.slope_nodes.data(),
        data_.slope_weights.data(),data_.weno_order.empty()?nullptr:data_.weno_order.data(),
        data_.weno_epsilon_scale.data()};
    return result;
}
ReconstructionIndex ReconstructionStencil::incomplete_sides() const {
    return static_cast<ReconstructionIndex>(std::count(data_.complete.begin(),data_.complete.end(),0));
}

struct detail::ReconstructionGeometry::Impl {
    MeshView m;
    double eps;
    std::vector<Surface> faces;
    std::vector<Vec> normals;
    std::vector<std::vector<PatchFace> > patches;
    std::vector<PatchFace> owners;
    Impl(const Mesh& mesh,double tolerance,bool build_stars)
        : m(mesh.view()),eps(tolerance),faces(static_cast<std::size_t>(m.face_count)),
          normals(static_cast<std::size_t>(m.cell_count)),patches(static_cast<std::size_t>(m.node_count)),
          owners(static_cast<std::size_t>(m.face_count)) {
        if (!std::isfinite(eps) || eps<=0 || eps>=0.01) throw ReconstructionError("Invalid geometry tolerance");
        std::vector<std::vector<MeshIndex> > incident(static_cast<std::size_t>(m.node_count));
        if (m.dimension==3) parallel_for(m.face_count,[&](MeshIndex f) {
            const MeshIndexSpan ids=m.face_nodes(f);
            faces[static_cast<std::size_t>(f)]=triangulate(m,std::vector<MeshIndex>(ids.begin(),ids.end()),eps);
        });
        for (MeshIndex c=0;c<m.cell_count;++c) {
            const MeshCell cell=m.cell(c);
            for (MeshIndex k=0;k<cell.face_count();++k)
                owners[static_cast<std::size_t>(cell.face_id(k))]=PatchFace{cell.face_id(k),c,cell.face_orientation(k),1};
            for (MeshIndex n:cell.node_ids()) incident[static_cast<std::size_t>(n)].push_back(c);
            if (m.dimension==2) normals[static_cast<std::size_t>(c)]=triangulate(m,cell_cycle(cell),eps).normal;
            if (m.dimension==3) {
                const Vec origin=point(m,cell.node_id(0));
                double scale=0;
                for (MeshIndex n:cell.node_ids()) scale=(std::max)(scale,norm(point(m,n)-origin));
                if (!(scale>0) || !std::isfinite(scale)) throw ReconstructionError("Degenerate cell scale");
                double volume6=0;
                for (MeshIndex k=0;k<cell.face_count();++k)
                    for (const Triangle& t:faces[static_cast<std::size_t>(cell.face_id(k))].triangles)
                        volume6+=cell.face_orientation(k)*dot((point(m,t[0])-origin)/scale,
                             cross((point(m,t[1])-origin)/scale,(point(m,t[2])-origin)/scale));
                if (!(volume6>eps)) throw ReconstructionError("Inverted or degenerate 3D cell");
            }
        }
        // In 2D a straight reconstruction is defined in one common plane.
        if (m.dimension==2 && m.cell_count) {
            const Vec origin=point(m,m.cell(0).node_id(0)),normal=normals[0];
            for (MeshIndex n=0;n<m.node_count;++n) {
                const Vec v=point(m,n)-origin;
                if (std::abs(dot(v,normal))>eps*norm(v)) throw ReconstructionError("2D EBR requires a planar mesh");
            }
        }
        if (build_stars) parallel_for(m.node_count,[&](MeshIndex n) {
            std::map<MeshIndex,PatchFace> boundary;
            for (MeshIndex c:incident[static_cast<std::size_t>(n)]) {
                const MeshCell cell=m.cell(c);
                for (MeshIndex k=0;k<cell.face_count();++k) {
                    const MeshIndex f=cell.face_id(k);
                    const auto found=boundary.find(f);
                    if (found==boundary.end()) boundary[f]=PatchFace{f,c,cell.face_orientation(k),1};
                    else ++found->second.count;
                }
            }
            for (const auto& f:boundary) if (f.second.count==1) patches[static_cast<std::size_t>(n)].push_back(f.second);
        });
    }
    Vec edge(MeshIndex e) const { return point(m,m.edge_end[e])-point(m,m.edge_begin[e]); }
};
detail::ReconstructionGeometry::ReconstructionGeometry(const Mesh& mesh,double tolerance,bool build_stars)
    : impl_(new Impl(mesh,tolerance,build_stars)) {}
detail::ReconstructionGeometry::~ReconstructionGeometry() {}
double detail::ReconstructionGeometry::edge_length(MeshIndex e) const {
    const double h=norm(impl_->edge(e));
    if (!(h>0) || !std::isfinite(h)) throw ReconstructionError("Zero or non-finite edge length");
    return h;
}
detail::RayPoint detail::ReconstructionGeometry::star_hit(MeshIndex e,unsigned side) const {
    const Impl& g=*impl_; const double h=edge_length(e);
    const MeshIndex node=side?g.m.edge_end[e]:g.m.edge_begin[e];
    return outer_point(g.m,node,g.edge(e)*((side?1.0:-1.0)/h),h,
                       g.patches[static_cast<std::size_t>(node)],g.faces,g.normals,g.eps);
}
detail::RayPoint detail::ReconstructionGeometry::layer_hit(MeshIndex e,unsigned side,const std::vector<MeshIndex>& ids) const {
    const Impl& g=*impl_; const double h=edge_length(e);
    const MeshIndex node=side?g.m.edge_end[e]:g.m.edge_begin[e];
    std::vector<PatchFace> layer; layer.reserve(ids.size());
    for (MeshIndex f:ids) layer.push_back(g.owners[static_cast<std::size_t>(f)]);
    return outer_point(g.m,node,g.edge(e)*((side?1.0:-1.0)/h),h,layer,g.faces,g.normals,g.eps,true);
}
detail::RayPoint detail::ReconstructionGeometry::parallel_star_hit(MeshIndex node,MeshIndex e,unsigned side) const {
    const Impl& g=*impl_; const double h=edge_length(e);
    return outer_point(g.m,node,g.edge(e)*((side?1.0:-1.0)/h),h,
                       g.patches[static_cast<std::size_t>(node)],g.faces,g.normals,g.eps);
}
void detail::ReconstructionGeometry::append_row(ReconstructionStencilData& data,const SparseRow& row,MeshIndex e,unsigned side) const {
    const Impl& g=*impl_; const double h=edge_length(e);
    const MeshIndex center=side?g.m.edge_end[e]:g.m.edge_begin[e];
    double sum=0; Vec moment;
    for (const auto& w:row) {
        if (!std::isfinite(w.second)) throw ReconstructionError("Non-finite reconstruction weight");
        sum+=w.second;
        moment=moment+((point(g.m,w.first)-point(g.m,center))/h)*w.second;
        if (w.second!=0) { data.nodes.push_back(w.first); data.weights.push_back(w.second); }
    }
    if (std::abs(sum)>100*g.eps || norm(moment-g.edge(e)/h)>100*g.eps)
        throw ReconstructionError("EBR stencil failed constant/linear consistency check");
    data.offsets.push_back(static_cast<ReconstructionIndex>(data.nodes.size()));
    data.row_owner.push_back(center);
}
void detail::add_scaled_row(SparseRow& destination,const SparseRow& source,double factor) {
    add_row(destination,source,factor);
}
detail::SparseRow detail::ray_derivative(MeshIndex node,const RayPoint& hit,double h,unsigned side) {
    return forward_derivative(node,hit,h,side);
}

ReconstructionStencil prepare_reconstruction(const Mesh& mesh,const ReconstructionOptions& options) {
    if (options.scheme==ReconstructionScheme::Weno3 || options.scheme==ReconstructionScheme::Weno5)
        return prepare_weno(mesh,options);
    if (options.scheme==ReconstructionScheme::Ebr3 || options.scheme==ReconstructionScheme::Ebr5)
        return prepare_ebr(mesh,options);
    if (options.scheme!=ReconstructionScheme::FirstOrder) throw ReconstructionError("Invalid reconstruction scheme");
    mesh.validate();
    ReconstructionStencil result;
    ReconstructionStencilData& data=result.data_;
    const MeshView m=mesh.view();
    data.scheme=options.scheme; data.node_count=m.node_count; data.edge_count=m.edge_count;
    if (data.edge_count>((std::numeric_limits<ReconstructionIndex>::max)()-1)/2)
        throw ReconstructionError("Too many mesh edges");
    const std::size_t rows=checked_size(2*data.edge_count,1);
    data.complete.assign(rows,1); data.outer_distance.assign(rows,0.0);
    data.side_scheme.assign(rows,ReconstructionSideScheme::FirstOrder); data.far_distance.assign(rows,0.0);
    data.offsets.assign(rows+1,0);
    for (MeshIndex e=0;e<m.edge_count;++e) {
        data.row_owner.push_back(m.edge_begin[e]); data.row_owner.push_back(m.edge_end[e]);
    }
    return result;
}

struct ReconstructionContext::Impl {
    ReconstructionStencil stencil;
    ReconstructionBackend backend;
    void* gpu;
    int device;
    explicit Impl(ReconstructionStencil s) : stencil(std::move(s)),backend(ReconstructionBackend::Cpu),gpu(nullptr),device(-1) {}
    ~Impl() {
#ifdef AERO_WITH_CUDA
        aero_reconstruction_cuda_destroy(gpu);
#endif
    }
};
bool ReconstructionContext::cuda_compiled() {
#ifdef AERO_WITH_CUDA
    return true;
#else
    return false;
#endif
}
int ReconstructionContext::cuda_device_count() {
#ifdef AERO_WITH_CUDA
    int count=0; char error[512]={};
    if (aero_reconstruction_cuda_count(&count,error,sizeof(error))) throw ReconstructionError(error);
    return count;
#else
    return 0;
#endif
}
ReconstructionContext::ReconstructionContext(ReconstructionStencil stencil,ReconstructionBackend backend,int device)
    : impl_(new Impl(std::move(stencil))) {
    if (device<0) throw ReconstructionError("Negative CUDA device index");
    if (backend!=ReconstructionBackend::Cpu && backend!=ReconstructionBackend::Cuda && backend!=ReconstructionBackend::Auto)
        throw ReconstructionError("Invalid EBR backend");
    if (backend==ReconstructionBackend::Cpu) return;
    const int count=cuda_device_count();
    if (!count && backend==ReconstructionBackend::Auto) return;
    if (device>=count) throw ReconstructionError("Requested CUDA device is unavailable (or CUDA module was not built)");
#ifdef AERO_WITH_CUDA
    char error[512]={};
    if (aero_reconstruction_cuda_create(impl_->stencil.view(),device,&impl_->gpu,error,sizeof(error))) throw ReconstructionError(error);
    impl_->backend=ReconstructionBackend::Cuda;
    impl_->device=device;
#endif
}
ReconstructionContext::~ReconstructionContext() {}
ReconstructionBackend ReconstructionContext::backend() const { return impl_->backend; }
int ReconstructionContext::device() const { return impl_->device; }
ReconstructionStencilView ReconstructionContext::host_view() const { return impl_->stencil.view(); }
ReconstructionStencilView ReconstructionContext::device_view() const {
    if (impl_->backend!=ReconstructionBackend::Cuda) throw ReconstructionError("Device view requires CUDA backend");
    ReconstructionStencilView view={};
#ifdef AERO_WITH_CUDA
    aero_reconstruction_cuda_view(impl_->gpu,&view);
#endif
    return view;
}
} // namespace aero
