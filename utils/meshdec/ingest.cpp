#include "meshdec.h"
#include "../../source/mesh/mesh_reader.h"
#include <cmath>

namespace aero { namespace meshdec {
namespace {
using mesh_input::Reader;
void indices(Reader& r,const std::string& path,const char* name,Id count,Id limit) {
    r.block(name);
    auto out=output(path);
    for (Id i=0;i<count;++i) {
        const Id v=r.u64();
        require(v<limit,std::string(name)+" index out of range");
        put(out,v);
    }
    out.close();
}
void offsets(Reader& r,const std::string& path,const char* name,Id rows,Id entries,
             Id min_width,Id max_width) {
    r.block(name);
    auto out=output(path);
    Id prev=r.u64();
    require(prev==0,"CSR must start at zero");
    put(out,prev);
    for (Id i=0;i<rows;++i) {
        const Id end=r.u64();
        require(end>=prev && end<=entries && end-prev>=min_width && end-prev<=max_width,
                std::string(name)+" invalid row");
        put(out,end);
        prev=end;
    }
    require(prev==entries,"CSR length mismatch");
    out.close();
}
struct Link {
    Id target,source,entry;
    bool operator<(const Link& b) const {
        return target==b.target ? source<b.source : target<b.target;
    }
};
void reverse(const std::string& dir,const std::string& forward,const std::string& back,
             Id rows,Id targets,Id memory,Table<std::int8_t>* signs=nullptr) {
    Table<Id> starts(dir+"/"+forward+"-start"), references(dir+"/"+forward);
    Sorter<Link> links(dir+"/"+back+".sort",memory);
    for (Id row=0;row<rows;++row) {
        const Id first=starts.at(row),end=starts.at(row+1);
        for (Id k=first;k<end;++k) links.add({references.at(k),row,k});
    }
    Table<Link> sorted(links.finish());
    auto off=output(dir+"/"+back+"-start"),ids=output(dir+"/"+back);
    Id k=0;
    for (Id i=0;i<targets;++i) {
        put(off,k);
        Id previous=none;
        int first_sign=0;
        while (k<sorted.count && sorted.at(k).target==i) {
            const auto v=sorted.at(k++);
            require(v.source!=previous,"Repeated entity within a face or cell");
            if (signs) {
                const int sign=signs->at(v.entry);
                if (previous==none) first_sign=sign;
                else require(first_sign+sign==0,"Shared face orientations must be opposite");
            }
            put(ids,v.source);
            previous=v.source;
        }
    }
    put(off,k);
    off.close();
    ids.close();
}
unsigned coordinate(double x,double lo,double hi) {
    if (lo==hi) return 0;
    // Halving prevents overflow for an axis spanning [-DBL_MAX, DBL_MAX].
    const double span=hi-lo;
    const double t=std::isfinite(span) ? (x-lo)/span : (x*0.5-lo*0.5)/(hi*0.5-lo*0.5);
    return static_cast<unsigned>(std::min(1.0,std::max(0.0,t))*2097151.0);
}
Id morton(const unsigned p[3]) {
    Id key=0;
    for (unsigned b=0;b<21;++b)
        for (unsigned axis=0;axis<3;++axis)
            key|=static_cast<Id>((p[axis]>>b)&1u)<<(3*b+axis);
    return key;
}
}
void unpack(const Options& options,const std::string& dir) {
    std::ifstream source(options.source.c_str(),std::ios::binary);
    require(source.is_open(),"Cannot open mesh: "+options.source);
    Reader r(source,MeshReadLimits(none));
    r.block("DIMENSION");
    Header h{};
    h.dimension=r.u64();
    require(h.dimension>=1 && h.dimension<=3,"Invalid mesh dimension");
    r.block("COUNTS");
    h.nodes=r.u64(); h.faces=r.u64(); h.face_entries=r.u64();
    h.cells=r.u64(); h.cell_entries=r.u64(); h.boundaries=r.u64(); h.patches=r.u64();
    r.block("LENGTH_SCALE_TO_METERS");
    h.scale=r.real();
    require(std::isfinite(h.scale) && h.scale>0,"Invalid mesh length scale");
    const char* axes[]={"X","Y","Z"};
    for (unsigned axis=0;axis<3;++axis) {
        r.block(axes[axis]);
        auto out=output(dir+"/"+axes[axis]);
        for (Id i=0;i<h.nodes;++i) {
            const auto v=r.real();
            require(std::isfinite(v),"Invalid coordinate");
            if (!i || v<h.low[axis]) h.low[axis]=v;
            if (!i || v>h.high[axis]) h.high[axis]=v;
            put(out,v);
        }
        out.close();
    }
    offsets(r,dir+"/fn-start","FACE_OFFSETS",h.faces,h.face_entries,h.dimension,h.dimension==3?none:h.dimension);
    indices(r,dir+"/fn","FACE_NODES",h.face_entries,h.nodes);
    offsets(r,dir+"/cf-start","CELL_OFFSETS",h.cells,h.cell_entries,h.dimension+1,h.dimension==1?2:none);
    indices(r,dir+"/cf","CELL_FACES",h.cell_entries,h.faces);
    r.block("CELL_ORIENTATIONS");
    auto signs=output(dir+"/signs");
    int first_sign=0;
    for (Id k=0;k<h.cell_entries;++k) {
        const auto sign=r.orientation();
        if (h.dimension==1) {
            if (k%2==0) first_sign=sign;
            else require(first_sign+sign==0,"Segment endpoints must have opposite orientations");
        }
        put(signs,sign);
    }
    signs.close();
    indices(r,dir+"/boundary","BOUNDARY_FACES",h.boundaries,h.faces);
    indices(r,dir+"/patch","BOUNDARY_PATCH",h.boundaries,h.patches);
    r.block("PATCH_NAMES");
    auto names=output(dir+"/names");
    // Native v2 limits names to 4096 bytes. Disk sorting also bounds memory
    // when the number of patches itself is large. Valid names contain no NUL.
    using Name=std::array<char,4096>;
    Sorter<Name> name_order(dir+"/name-order",options.memory);
    for (Id i=0;i<h.patches;++i) {
        const auto name=r.name();
        require(mesh_input::valid_name(name),"Invalid boundary name");
        word(names,name.size());
        names.write(name.data(),position(name.size()));
        if (h.patches>1) {
            Name key{};
            require(name.size()<=key.size(),"Boundary name too long");
            std::copy(name.begin(),name.end(),key.begin());
            name_order.add(key);
        }
    }
    r.finish();
    names.close();
    Table<Name> sorted_names(name_order.finish());
    for (Id i=1;i<sorted_names.count;++i)
        require(sorted_names.at(i-1)!=sorted_names.at(i),"Duplicate boundary name");
    auto meta=output(dir+"/header");
    put(meta,h);
    meta.close();
}
void index_topology(const std::string& dir,Id memory) {
    const auto h=Table<Header>(dir+"/header").at(0);
    reverse(dir,"fn","nf",h.faces,h.nodes,memory);
    Table<std::int8_t> signs(dir+"/signs");
    reverse(dir,"cf","fc",h.cells,h.faces,memory,&signs);
    Table<double> x(dir+"/X"),y(dir+"/Y"),z(dir+"/Z");
    auto keys=output(dir+"/node-key");
    for (Id i=0;i<h.nodes;++i) {
        const unsigned p[]={coordinate(x.at(i),h.low[0],h.high[0]),coordinate(y.at(i),h.low[1],h.high[1]),
                            coordinate(z.at(i),h.low[2],h.high[2])};
        put(keys,morton(p));
    }
    keys.close();
    Table<Id> boundary(dir+"/boundary"),patch(dir+"/patch");
    Sorter<Link> labels(dir+"/labels.sort",memory);
    for (Id i=0;i<h.boundaries;++i) labels.add({boundary.at(i),i,patch.at(i)});
    Table<Link> sorted(labels.finish());
    Table<Id> fn(dir+"/fn"),off(dir+"/fn-start"),node_key(dir+"/node-key"),fc(dir+"/fc-start");
    auto faces=output(dir+"/face-key"),bc=output(dir+"/face-bc"),boxes=output(dir+"/face-box");
    Id b=0;
    for (Id f=0;f<h.faces;++f) {
        Id key=none;
        Box box{};
        const Id first=off.at(f),end=off.at(f+1);
        for (Id k=first;k<end;++k) {
            const Id n=fn.at(k);
            key=std::min(key,node_key.at(n));
            const auto point=Box::point(x.at(n),y.at(n),z.at(n));
            if (k==first) box=point;
            else box.include(point);
        }
        put(boxes,box);
        put(faces,key);
        const Id n=fc.at(f+1)-fc.at(f);
        require(n==1 || n==2,"Face must have one or two incident cells");
        Boundary label={none,none};
        if (b<sorted.count && sorted.at(b).target==f) {
            const auto link=sorted.at(b++);
            label={link.source,link.entry};
        }
        require((label.id!=none)==(n==1),"Boundary label and face incidence disagree");
        put(bc,label);
    }
    require(b==h.boundaries,"Repeated boundary face");
    faces.close();
    bc.close();
    boxes.close();
}
} }
