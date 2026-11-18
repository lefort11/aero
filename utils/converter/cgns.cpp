#include "detail.h"
#ifndef AERO_WITH_CGNS
namespace aero { namespace converter {
bool cgns_available() noexcept { return false; }
Mesh read_cgns(const std::string&,const Options&) {
    throw MeshError("CGNS support is disabled. Rebuild with WITH_CGNS=1 and the CGNS library.");
}
} }
#else
#include <cgnslib.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
namespace aero { namespace converter {
namespace {
void check(int code) { if(code!=CG_OK) throw MeshError(std::string("CGNS: ")+cg_get_error()); }
std::size_t count(cgsize_t n) {
    if(n<0 || static_cast<std::uint64_t>(n)>100000000) throw MeshError("CGNS: invalid/excessive count");
    return static_cast<std::size_t>(n);
}
struct File {
    int id;
    explicit File(const std::string& path) {check(cg_open(path.c_str(),CG_MODE_READ,&id));}
    ~File() {cg_close(id);}
};
int kind(CGNS_ENUMT(ElementType_t) type) {
    switch(type) {
    case CGNS_ENUMV(NODE):return 15; case CGNS_ENUMV(BAR_2):return 1;
    case CGNS_ENUMV(TRI_3):return 2; case CGNS_ENUMV(QUAD_4):return 3;
    case CGNS_ENUMV(TETRA_4):return 4; case CGNS_ENUMV(HEXA_8):return 5;
    case CGNS_ENUMV(PENTA_6):return 6; case CGNS_ENUMV(PYRA_5):return 7;
    case CGNS_ENUMV(NGON_n):return 100;
    default:throw MeshError("CGNS: unsupported/high-order element type "+std::to_string(static_cast<int>(type)));
    }
}
MeshIndex node(cgsize_t id,std::size_t n) {
    if(id<1||static_cast<std::uint64_t>(id)>n) throw MeshError("CGNS: invalid node reference");
    return static_cast<MeshIndex>(id-1);
}
struct Record { int kind; detail::Nodes nodes; std::vector<cgsize_t> faces; };
MeshIndex structured_node(const cgsize_t* ijk,const cgsize_t* shape,int dim) {
    MeshIndex id=0,stride=1;
    for(int d=0;d<dim;++d) {
        if(ijk[d]<1||ijk[d]>shape[d]) throw MeshError("CGNS: structured point index out of range");
        id+=static_cast<MeshIndex>(ijk[d]-1)*stride; stride*=static_cast<MeshIndex>(shape[d]);
    }
    return id;
}
}
bool cgns_available() noexcept { return true; }
Mesh read_cgns(const std::string& path,const Options& options) {
    File file(path); const int fn=file.id;
    int nb=0; check(cg_nbases(fn,&nb));
    const int b=options.base?options.base:1;
    if(b<1||b>nb||(!options.base&&nb!=1)) throw MeshError("CGNS: select a base explicitly with --base");
    char name[33];int dim=0,physical=0;
    check(cg_base_read(fn,b,name,&dim,&physical));
    if(dim<1||dim>3||physical<dim||physical>3) throw MeshError("CGNS: unsupported base dimensions");
    int nz=0;check(cg_nzones(fn,b,&nz));const int z=options.zone?options.zone:1;
    if(z<1||z>nz||(!options.zone&&nz!=1)) throw MeshError("CGNS: select a zone explicitly with --zone");
    CGNS_ENUMT(ZoneType_t) ztype;check(cg_zone_type(fn,b,z,&ztype));
    const bool structured=ztype==CGNS_ENUMV(Structured);
    if(!structured&&ztype!=CGNS_ENUMV(Unstructured)) throw MeshError("CGNS: unsupported zone type");
    int connections=0;check(cg_nconns(fn,b,z,&connections));
    int one_to_one=0;check(cg_n1to1(fn,b,z,&one_to_one));
    if(connections||one_to_one) throw MeshError("CGNS: inter-zone/periodic connectivity is not yet supported");
    cgsize_t size[9]={0};check(cg_zone_read(fn,b,z,name,size));
    const int indexdim=structured?dim:1;
    cgsize_t lo[3]={1,1,1},hi[3]={1,1,1};
    std::size_t nn=1,nc=1;
    for(int d=0;d<indexdim;++d) {
        const std::size_t n=count(size[d]),c=count(size[indexdim+d]);
        if(n==0||c==0||nn>100000000/n||nc>100000000/c) throw MeshError("CGNS: invalid zone size");
        if(structured&&c+1!=n) throw MeshError("CGNS: inconsistent structured dimensions");
        nn*=n;nc*=c;hi[d]=size[d];
    }
    detail::Builder builder(static_cast<unsigned>(dim),options);
    builder.data.x.resize(nn);builder.data.y.resize(nn);builder.data.z.resize(nn);
    std::vector<double>* coords[]={&builder.data.x,&builder.data.y,&builder.data.z};
    bool have[3]={false,false,false};
    int ncoords=0;check(cg_ncoords(fn,b,z,&ncoords));
    for(int c=1;c<=ncoords;++c) {
        CGNS_ENUMT(DataType_t) type;check(cg_coord_info(fn,b,z,c,&type,name));
        int axis=-1;
        if(std::string(name)=="CoordinateX")axis=0;
        if(std::string(name)=="CoordinateY")axis=1;
        if(std::string(name)=="CoordinateZ")axis=2;
        if(axis>=0) {check(cg_coord_read(fn,b,z,name,CGNS_ENUMV(RealDouble),lo,hi,coords[axis]->data()));have[axis]=true;}
    }
    for(int axis=0;axis<physical;++axis) if(!have[axis]) throw MeshError("CGNS: Cartesian CoordinateX/Y/Z arrays required");
    std::map<cgsize_t,Record> records;
    if(structured) {
        const cgsize_t ni=size[0],nj=dim>=2?size[1]:1,nk=dim>=3?size[2]:1;
        for(cgsize_t k=1;k<(dim>=3?nk:2);++k)
            for(cgsize_t j=1;j<(dim>=2?nj:2);++j)
                for(cgsize_t i=1;i<ni;++i) {
                    const cgsize_t p[3]={i,j,k};
                    const MeshIndex a=structured_node(p,size,dim), row=static_cast<MeshIndex>(ni), layer=row*static_cast<MeshIndex>(nj);
                    detail::Nodes nodes;
                    if(dim==1)nodes={a,a+1};
                    else if(dim==2)nodes={a,a+1,a+row+1,a+row};
                    else nodes={a,a+1,a+row+1,a+row,a+layer,a+layer+1,a+layer+row+1,a+layer+row};
                    builder.cell(detail::cell_faces(dim==1?1:(dim==2?3:5),nodes));
                }
    } else {
        int ns=0;check(cg_nsections(fn,b,z,&ns));
        for(int s=1;s<=ns;++s) {
            CGNS_ENUMT(ElementType_t) type;cgsize_t start=0,end=0,total=0;int boundary=0,parent=0;
            check(cg_section_read(fn,b,z,s,name,&type,&start,&end,&boundary,&parent));
            if(start<1||end<start)throw MeshError("CGNS: invalid section range");
            const std::size_t ne=count(end-start+1);
            check(cg_ElementDataSize(fn,b,z,s,&total));
            std::vector<cgsize_t> conn(count(total)),offset;
            const bool poly=type==CGNS_ENUMV(NGON_n)||type==CGNS_ENUMV(NFACE_n)||type==CGNS_ENUMV(MIXED);
            if(poly) {offset.resize(ne+1);check(cg_poly_elements_read(fn,b,z,s,conn.data(),offset.data(),nullptr));}
            else check(cg_elements_read(fn,b,z,s,conn.data(),nullptr));
            std::size_t cursor=0;
            for(std::size_t e=0;e<ne;++e) {
                CGNS_ENUMT(ElementType_t) et=type;
                std::size_t last=0;
                if(poly) {
                    if(offset[e]<0||offset[e+1]<offset[e]||static_cast<std::uint64_t>(offset[e+1])>conn.size()||
                       static_cast<std::size_t>(offset[e])!=cursor)throw MeshError("CGNS: invalid connectivity offsets");
                    last=static_cast<std::size_t>(offset[e+1]);
                } else {int npe=0;check(cg_npe(type,&npe));last=cursor+static_cast<std::size_t>(npe);}
                if(last>conn.size()||cursor==last)throw MeshError("CGNS: truncated element connectivity");
                if(type==CGNS_ENUMV(MIXED))et=static_cast<CGNS_ENUMT(ElementType_t)>(conn[cursor++]);
                Record record;
                if(et==CGNS_ENUMV(NFACE_n)) {
                    if(dim!=3||type==CGNS_ENUMV(MIXED))throw MeshError("CGNS: unsupported NFACE placement");
                    record.kind=101;
                    while(cursor<last)record.faces.push_back(conn[cursor++]);
                } else {
                    record.kind=kind(et);
                    while(cursor<last)record.nodes.push_back(node(conn[cursor++],nn));
                    if(record.kind!=100&&record.nodes.size()!=detail::node_count(record.kind))throw MeshError("CGNS: wrong element size");
                }
                if(!records.insert(std::make_pair(start+static_cast<cgsize_t>(e),record)).second)
                    throw MeshError("CGNS: overlapping element sections");
            }
            if(cursor!=conn.size())throw MeshError("CGNS: trailing section connectivity");
        }
        std::size_t cells=0;
        for(const auto& item:records) {
            const Record& rec=item.second;
            if(rec.kind==101) {
                detail::Faces faces;
                for(cgsize_t signed_id:rec.faces) {
                    if(signed_id==0||signed_id==std::numeric_limits<cgsize_t>::min())throw MeshError("CGNS: invalid NFACE reference");
                    auto f=records.find(signed_id<0?-signed_id:signed_id);
                    if(f==records.end()||f->second.kind!=100)throw MeshError("CGNS: NFACE references missing NGON");
                    detail::Nodes nodes=f->second.nodes;
                    if(signed_id<0)std::reverse(nodes.begin(),nodes.end());
                    faces.push_back(nodes);
                }
                builder.cell(faces);++cells;
            } else if(detail::dimension(rec.kind)==static_cast<unsigned>(dim)) {
                builder.cell(detail::cell_faces(rec.kind,rec.nodes));++cells;
            }
        }
        if(cells!=nc)throw MeshError("CGNS: zone cell count mismatch; NGON volumes require NFACE connectivity");
    }
    int nboco=0;check(cg_nbocos(fn,b,z,&nboco));
    for(int bc=1;bc<=nboco;++bc) {
        CGNS_ENUMT(BCType_t) type;CGNS_ENUMT(PointSetType_t) settype;
        CGNS_ENUMT(DataType_t) normaltype;cgsize_t np=0,normals=0;int normalindex[3]={0},ndataset=0;
        check(cg_boco_info(fn,b,z,bc,name,&type,&settype,&np,normalindex,&normals,&normaltype,&ndataset));
        const std::string patch=name;const std::size_t n=count(np);
        if(n>100000000/static_cast<unsigned>(indexdim))throw MeshError("CGNS: boundary too large");
        std::vector<cgsize_t> points(n*static_cast<unsigned>(indexdim));
        check(cg_boco_read(fn,b,z,bc,points.data(),nullptr));
        CGNS_ENUMT(GridLocation_t) location=CGNS_ENUMV(Vertex);
        check(cg_boco_gridlocation_read(fn,b,z,bc,&location));
        const bool range=settype==CGNS_ENUMV(PointRange)||settype==CGNS_ENUMV(ElementRange);
        const bool list=settype==CGNS_ENUMV(PointList)||settype==CGNS_ENUMV(ElementList);
        if((!range&&!list)||(range&&n!=2)||!n)throw MeshError("CGNS: unsupported boundary point set");
        const bool vertex=location==CGNS_ENUMV(Vertex)&&settype!=CGNS_ENUMV(ElementRange)&&settype!=CGNS_ENUMV(ElementList);
        if(structured&&!vertex)throw MeshError("CGNS: structured BC currently requires Vertex location");
        detail::Nodes selected;
        if(range) {
            cgsize_t low[3]={1,1,1},high[3]={1,1,1};
            std::size_t expanded=1;
            for(int d=0;d<indexdim;++d) {
                low[d]=std::min(points[static_cast<std::size_t>(d)],points[static_cast<std::size_t>(indexdim+d)]);
                high[d]=std::max(points[static_cast<std::size_t>(d)],points[static_cast<std::size_t>(indexdim+d)]);
                if(low[d]<1)throw MeshError("CGNS: invalid boundary range");
                const std::size_t width=count(high[d]-low[d]+1);
                if(width==0||expanded>100000000/width)throw MeshError("CGNS: boundary range too large");
                expanded*=width;
            }
            for(cgsize_t k=low[2];k<=high[2];++k)for(cgsize_t j=low[1];j<=high[1];++j)for(cgsize_t i=low[0];i<=high[0];++i) {
                cgsize_t p[3]={i,j,k};
                selected.push_back(structured?structured_node(p,size,dim):(vertex?node(i,nn):static_cast<MeshIndex>(i)));
            }
        } else for(std::size_t i=0;i<n;++i)
            selected.push_back(structured?structured_node(&points[i*static_cast<unsigned>(indexdim)],size,dim):
                               (vertex?node(points[i],nn):static_cast<MeshIndex>(points[i])));
        if(vertex) builder.mark_vertices(selected,patch);
        else {
            if(location!=CGNS_ENUMV(FaceCenter)&&location!=CGNS_ENUMV(EdgeCenter)&&
               settype!=CGNS_ENUMV(ElementList)&&settype!=CGNS_ENUMV(ElementRange))
                throw MeshError("CGNS: unsupported BC location");
            for(MeshIndex id:selected) {
                const auto e=records.find(static_cast<cgsize_t>(id));
                if(e==records.end()||e->second.kind==101||detail::dimension(e->second.kind)+1!=static_cast<unsigned>(dim))
                    throw MeshError("CGNS: BC does not reference boundary elements");
                builder.mark(e->second.nodes,patch);
            }
        }
    }
    return builder.finish();
}
} }
#endif
