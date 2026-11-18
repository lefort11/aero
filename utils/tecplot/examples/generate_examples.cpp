// Synthetic visualization fixtures; these are analytic samples, not CFD solutions.
#include "../../../source/mesh/mesh.h"
#include "../../../source/io/fields.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
using namespace aero;
typedef std::vector<MeshIndex> Polygon;
const double pi=std::acos(-1.0);
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}

struct Builder {
    MeshData data;
    std::map<Polygon,MeshIndex> faces;
    std::vector<unsigned> incidence;
    void cell(const std::vector<Polygon>& shell) {
        for(const Polygon& nodes:shell) {
            Polygon key=nodes;std::sort(key.begin(),key.end());
            const auto found=faces.find(key);MeshIndex id;int orientation=1;
            if(found==faces.end()) {
                id=data.face_nodes.offsets.size()-1;faces[key]=id;incidence.push_back(0);
                data.face_nodes.indices.insert(data.face_nodes.indices.end(),nodes.begin(),nodes.end());
                data.face_nodes.offsets.push_back(data.face_nodes.indices.size());
            } else {
                id=found->second;
                const std::size_t first=static_cast<std::size_t>(data.face_nodes.offsets[static_cast<std::size_t>(id)]);
                const std::size_t n=nodes.size();
                bool forward=false,backward=false;
                for(std::size_t shift=0;shift<n;++shift) {
                    bool f=true,b=true;
                    for(std::size_t k=0;k<n;++k) {
                        f=f&&nodes[k]==data.face_nodes.indices[first+(shift+k)%n];
                        b=b&&nodes[k]==data.face_nodes.indices[first+(shift+n-k)%n];
                    }
                    forward=forward||f;backward=backward||b;
                }
                // Directed edges have no cyclic rotation ambiguity in 2D.
                if(data.dimension==2)forward=nodes[0]==data.face_nodes.indices[first];
                require(forward||backward,"Inconsistent face cycle");orientation=forward?1:-1;
            }
            ++incidence[static_cast<std::size_t>(id)];
            data.cell_faces.indices.push_back(id);data.cell_face_orientation.push_back(static_cast<std::int8_t>(orientation));
        }
        data.cell_faces.offsets.push_back(data.cell_faces.indices.size());
    }
    Mesh finish() {
        data.patch_names={"xmin","xmax","ymin","ymax"};
        if(data.dimension==3) {data.patch_names.push_back("zmin");data.patch_names.push_back("zmax");}
        for(std::size_t f=0;f<incidence.size();++f)if(incidence[f]==1) {
            const std::size_t begin=static_cast<std::size_t>(data.face_nodes.offsets[f]);
            const std::size_t end=static_cast<std::size_t>(data.face_nodes.offsets[f+1]);
            double x=0,y=0,z=0;
            for(std::size_t k=begin;k<end;++k) {
                const std::size_t n=static_cast<std::size_t>(data.face_nodes.indices[k]);x+=data.x[n];y+=data.y[n];z+=data.z[n];
            }
            const double count=static_cast<double>(end-begin);x/=count;y/=count;z/=count;
            MeshIndex patch=0;
            if(std::abs(x)<1e-12)patch=0;else if(std::abs(x-1)<1e-12)patch=1;
            else if(std::abs(y)<1e-12)patch=2;else if(std::abs(y-1)<1e-12)patch=3;
            else if(data.dimension==3&&std::abs(z)<1e-12)patch=4;
            else if(data.dimension==3&&std::abs(z-1)<1e-12)patch=5;
            else throw std::runtime_error("Unexpected exterior face");
            data.boundary_faces.push_back(f);data.boundary_patch.push_back(patch);
        }
        return Mesh(data);
    }
};

Mesh line_mesh() {
    MeshData d;d.dimension=1;d.patch_names={"inlet","outlet"};const MeshIndex n=80;
    for(MeshIndex i=0;i<=n;++i) {
        d.x.push_back(std::pow(static_cast<double>(i)/static_cast<double>(n),1.6));d.y.push_back(0);d.z.push_back(0);
        d.face_nodes.indices.push_back(i);d.face_nodes.offsets.push_back(i+1);
    }
    for(MeshIndex i=0;i<n;++i) {
        d.cell_faces.indices.insert(d.cell_faces.indices.end(),{i,i+1});d.cell_faces.offsets.push_back(2*(i+1));
        d.cell_face_orientation.insert(d.cell_face_orientation.end(),{-1,1});
    }
    d.boundary_faces={0,n};d.boundary_patch={0,1};return Mesh(d);
}
struct Plane {
    std::vector<double> x,y;
    std::vector<Polygon> polygons;
    Plane() {
        const unsigned nx=16,ny=12;
        const auto node=[](unsigned i,unsigned j)->MeshIndex {return j*17+i;};
        for(unsigned j=0;j<=ny;++j)for(unsigned i=0;i<=nx;++i) {
            const bool inside=i>0&&i<nx&&j>0&&j<ny;
            x.push_back((i+(inside?0.14*std::sin(0.7*i+0.4*j):0))/nx);
            y.push_back((j+(inside?0.14*std::cos(0.3*i-0.8*j):0))/ny);
        }
        for(unsigned j=0;j<ny;++j)for(unsigned i=0;i<nx;) {
            if(i%4==0&&i+1<nx) {
                polygons.push_back({node(i,j),node(i+1,j),node(i+2,j),node(i+2,j+1),node(i+1,j+1),node(i,j+1)});i+=2;
            } else {
                const MeshIndex a=node(i,j),b=node(i+1,j),c=node(i+1,j+1),d=node(i,j+1);
                if((i+j)%3==0) {polygons.push_back({a,b,c});polygons.push_back({a,c,d});}
                else polygons.push_back({a,b,c,d});
                ++i;
            }
        }
    }
};
Mesh planar_mesh(const Plane& plane) {
    Builder b;b.data.dimension=2;b.data.x=plane.x;b.data.y=plane.y;b.data.z.assign(plane.x.size(),0);
    for(const Polygon& polygon:plane.polygons) {
        std::vector<Polygon> edges;
        for(std::size_t i=0;i<polygon.size();++i)edges.push_back({polygon[i],polygon[(i+1)%polygon.size()]});
        b.cell(edges);
    }
    return b.finish();
}
Mesh volume_mesh(const Plane& plane) {
    Builder b;const MeshIndex stride=plane.x.size();const unsigned layers=6;
    for(unsigned k=0;k<=layers;++k)for(std::size_t i=0;i<plane.x.size();++i) {
        b.data.x.push_back(plane.x[i]);b.data.y.push_back(plane.y[i]);b.data.z.push_back(static_cast<double>(k)/layers);
    }
    for(unsigned k=0;k<layers;++k)for(const Polygon& polygon:plane.polygons) {
        Polygon bottom,top;
        for(auto it=polygon.rbegin();it!=polygon.rend();++it)bottom.push_back(*it+k*stride);
        for(MeshIndex n:polygon)top.push_back(n+(k+1)*stride);
        std::vector<Polygon> shell={bottom,top};
        for(std::size_t i=0;i<polygon.size();++i) {
            const MeshIndex a=polygon[i],c=polygon[(i+1)%polygon.size()];
            shell.push_back({a+k*stride,c+k*stride,c+(k+1)*stride,a+(k+1)*stride});
        }
        b.cell(shell);
    }
    return b.finish();
}
struct Point {double x,y,z;};
Point position(const Mesh& mesh,FieldLocation loc,MeshIndex id) {
    if(loc==FieldLocation::Node) {const MeshNode n=mesh.node(id);return {n.x(),n.y(),n.z()};}
    Polygon ids;
    if(loc==FieldLocation::Edge)ids={mesh.edge(id).node_id(0),mesh.edge(id).node_id(1)};
    else {
        const MeshIndexSpan span=loc==FieldLocation::Face?mesh.face(id).node_ids():mesh.cell(id).node_ids();
        ids.assign(span.begin(),span.end());
    }
    Point p={0,0,0};
    for(MeshIndex n:ids) {p.x+=mesh.node(n).x();p.y+=mesh.node(n).y();p.z+=mesh.node(n).z();}
    const double size=static_cast<double>(ids.size());p.x/=size;p.y/=size;p.z/=size;return p;
}
double pulse(Point p,unsigned dim) {
    double r=(p.x-0.45)*(p.x-0.45);
    if(dim>=2)r+=(p.y-0.55)*(p.y-0.55);
    if(dim==3)r+=(p.z-0.5)*(p.z-0.5);
    return std::exp(-r/0.035);
}
Point velocity(Point p,unsigned dim) {
    if(dim==1)return {1+0.3*std::sin(2*pi*p.x),0,0};
    return {-(p.y-0.5),p.x-0.5,dim==3?0.2*std::sin(pi*p.z):0};
}
Field make_field(const Mesh& mesh,FieldLocation loc,const std::string& name,const std::string& unit,const std::vector<std::string>& components) {
    Field f;f.name=name;f.location=loc;f.units=unit;f.components=components;
    const MeshView v=mesh.view();
    const MeshIndex count=loc==FieldLocation::Node?v.node_count:loc==FieldLocation::Cell?v.cell_count:loc==FieldLocation::Face?v.face_count:v.edge_count;
    for(std::size_t c=0;c<components.size();++c)for(MeshIndex i=0;i<count;++i) {
        const Point p=position(mesh,loc,i);double value=0;
        if(name=="temperature")value=300+80*pulse(p,v.dimension);
        else if(name=="pressure")value=100000+20000*pulse(p,v.dimension);
        else if(name=="density")value=1.2+0.1*std::sin(2*pi*p.x)*(v.dimension>=2?std::cos(2*pi*p.y):1);
        else {
            const Point u=velocity(p,v.dimension);
            value=name=="speed"?std::sqrt(u.x*u.x+u.y*u.y+u.z*u.z):(c==0?u.x:c==1?u.y:u.z);
        }
        f.values.push_back(value);
    }
    return f;
}
FieldSnapshot sample(const Mesh& mesh) {
    FieldSnapshot s(mesh,0,0);
    s.fields.push_back(make_field(mesh,FieldLocation::Node,"temperature","K",{"value"}));
    s.fields.push_back(make_field(mesh,FieldLocation::Node,"velocity","m/s",{"x","y","z"}));
    s.fields.push_back(make_field(mesh,FieldLocation::Cell,"pressure","Pa",{"value"}));
    s.fields.push_back(make_field(mesh,FieldLocation::Cell,"density","kg/m^3",{"value"}));
    s.fields.push_back(make_field(mesh,FieldLocation::Edge,"speed","m/s",{"value"}));
    s.fields.push_back(make_field(mesh,FieldLocation::Face,"pressure","Pa",{"value"}));
    return s;
}
void geometry_check(const Mesh& mesh) {
    double total=0;
    for(MeshIndex c=0;c<mesh.view().cell_count;++c) {
        const MeshCell cell=mesh.cell(c);double measure=0;
        if(mesh.data().dimension==1)measure=cell.node(1).x()-cell.node(0).x();
        else for(MeshIndex i=0;i<cell.face_count();++i) {
            const MeshFace f=cell.face(i);const double sign=cell.face_orientation(i);
            const MeshNode a=f.node(0);
            if(mesh.data().dimension==2) {const MeshNode b=f.node(1);measure+=sign*(a.x()*b.y()-a.y()*b.x())/2;}
            else for(MeshIndex j=1;j+1<f.node_count();++j) {
                const MeshNode b=f.node(j),d=f.node(j+1);
                measure+=sign*(a.x()*(b.y()*d.z()-b.z()*d.y())+a.y()*(b.z()*d.x()-b.x()*d.z())+a.z()*(b.x()*d.y()-b.y()*d.x()))/6;
            }
        }
        require(measure>0,"Nonpositive cell measure");total+=measure;
    }
    require(std::abs(total-1)<1e-10,"Expected unit domain measure");
}
void save(const std::string& prefix,const Mesh& mesh,MeshFormat mf,FieldFormat ff) {
    const FieldSnapshot original=sample(mesh);geometry_check(mesh);
    for(const std::string& extension:{std::string(".mesh"),std::string(".fields")}) {
        std::ifstream existing((prefix+extension).c_str());require(!existing,"Example already exists");
    }
    mesh.write(prefix+".mesh",mf);original.write(prefix+".fields",ff);
    const Mesh restored=Mesh::read(prefix+".mesh");const FieldSnapshot read=FieldSnapshot::read(prefix+".fields");read.check_mesh(restored);
    std::ostringstream a(std::ios::binary),b(std::ios::binary);original.write(a);read.write(b);
    require(a.str()==b.str(),"Fields changed during round trip");
    geometry_check(restored);
    std::cout<<prefix<<": nodes="<<mesh.view().node_count<<", cells="<<mesh.view().cell_count<<", faces="<<mesh.view().face_count<<", fields="<<read.fields.size()<<'\n';
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::runtime_error("Usage: generate_tecplot_examples EXISTING_OUTPUT_DIRECTORY");
        const std::string dir=argv[1];const Plane plane;
        save(dir+"/demo_1d",line_mesh(),MeshFormat::Text,FieldFormat::Text);
        save(dir+"/demo_2d",planar_mesh(plane),MeshFormat::Binary,FieldFormat::Binary);
        save(dir+"/demo_3d",volume_mesh(plane),MeshFormat::Lzma,FieldFormat::Lzma);
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    return 0;
}
