#include "meshdec.h"

namespace aero { namespace meshdec {
namespace {
struct Pair {
    Id row,value;
    bool operator<(const Pair& b) const { return row==b.row ? value<b.value : row<b.row; }
};

// Store each cell once in a row, even when reached through several faces.
// The temporary ID table also supports the join with cross-leaf edge ends.
void star_rows(const std::string& base,Sorter<Pair>& pairs,Id count) {
    Table<Pair> sorted(pairs.finish());
    auto starts=output(base+"-start"),cells=output(base+"-cells");
    Id pos=0,entries=0;
    for (Id row=0;row<count;++row) {
        put(starts,entries);
        Id previous=none;
        while (pos<sorted.count && sorted.at(pos).row==row) {
            const Id cell=sorted.at(pos++).value;
            if (cell==previous) continue;
            put(cells,cell);
            previous=cell;
            ++entries;
        }
    }
    put(starts,entries);
    require(pos==sorted.count,"Halo row out of range");
    starts.close(); cells.close();
}

// A periodic representative can acquire an edge from the opposite patch.
// Only the endpoints of edges touching a physical patch can then be required.
void mark_boundary(const std::string& dir,Id count) {
    Table<Id> starts(dir+"/nf-start"),faces(dir+"/nf");
    Table<Boundary> labels(dir+"/face-bc");
    auto out=output(dir+"/boundary-nodes");
    for (Id n=0;n<count;++n) {
        std::uint8_t marked=0;
        for (Id j=starts.at(n);j<starts.at(n+1);++j)
            if (labels.at(faces.at(j)).id!=none) { marked=1; break; }
        put(out,marked);
    }
    out.close();
}
// Emit vertices individually so even a large polygon or polyhedron can be
// visited within the external sort budget.
struct CellNodes {
    Table<Id> cf,cf_start,fn,fn_start;
    explicit CellNodes(const std::string& dir)
        : cf(dir+"/cf"),cf_start(dir+"/cf-start"),fn(dir+"/fn"),fn_start(dir+"/fn-start") {}
    template<class Visit> void each(Id c,Visit visit) {
        for (Id j=cf_start.at(c);j<cf_start.at(c+1);++j) {
            const Id f=cf.at(j);
            for (Id k=fn_start.at(f);k<fn_start.at(f+1);++k) visit(fn.at(k));
        }
    }
};
void boundary_stars(const std::string& dir,Table<Id>& needed,Table<Id>& starts,Table<Id>& incidence,Table<Address>& cells) {
    auto offsets=output(dir+"/stars-start"),refs=output(dir+"/stars-refs");
    Id i=0,total=0;
    for (Id n=0;n<starts.count-1;++n) {
        put(offsets,total);
        if (i==needed.count || needed.at(i)!=n) continue;
        while (i<needed.count && needed.at(i)==n) ++i;
        for (Id j=starts.at(n);j<starts.at(n+1);++j) {
            const auto a=cells.at(incidence.at(j));
            word(refs,a.leaf); word(refs,a.offset); ++total;
        }
    }
    put(offsets,total);
    offsets.close(); refs.close();
}

// Each leaf has two disjoint rows. The core covers every owned vertex;
// the second row contains just the additional cells required by EBR3.
template<class Visit> void leaf_rows(const std::string& dir,Table<Pair>& core,Table<Pair>& outer,
                                     Table<Address>& cells,Id count,Visit visit) {
    auto offsets=output(dir+"/halo-start"),refs=output(dir+"/halo-refs");
    Id i=0,j=0,total=0;
    const auto emit=[&](Id leaf,Id c) {
        visit(leaf,c);
        const auto a=cells.at(c);
        word(refs,a.leaf); word(refs,a.offset);
        ++total;
    };
    for (Id leaf=0;leaf<count;++leaf) {
        Id k=i,previous=none;
        put(offsets,total);
        while (i<core.count && core.at(i).row==leaf) {
            const Id c=core.at(i++).value;
            if (c!=previous) emit(leaf,c);
            previous=c;
        }
        put(offsets,total); previous=none;
        while (j<outer.count && outer.at(j).row==leaf) {
            const Id c=outer.at(j++).value;
            if (c==previous) continue;
            while (k<i && core.at(k).value<c) ++k;
            if (k==i || core.at(k).value!=c) emit(leaf,c);
            previous=c;
        }
    }
    put(offsets,total);
    require(i==core.count && j==outer.count,"Halo leaf out of range");
    offsets.close(); refs.close();
}

// Look up the stars requested by each leaf once per distinct vertex.
void expand_stars(Table<Pair>& centers,Table<Id>& starts,Table<Id>& incidence,Sorter<Pair>& output) {
    Pair previous={none,none};
    for (Id i=0;i<centers.count;++i) {
        const auto p=centers.at(i);
        if (p.row==previous.row && p.value==previous.value) continue;
        for (Id j=starts.at(p.value);j<starts.at(p.value+1);++j) output.add({p.row,incidence.at(j)});
        previous=p;
    }
}
void far_rows(const std::string& dir,Table<Pair>& core,Table<Pair>& outer,Table<Pair>& far,Table<Address>& cells,Id leaves) {
    auto offsets=output(dir+"/far-start"),refs=output(dir+"/far-refs");
    Id i=0,j=0,k=0,total=0;
    for (Id leaf=0;leaf<leaves;++leaf) {
        put(offsets,total);
        Id previous=none;
        while (k<far.count && far.at(k).row==leaf) {
            const auto p=far.at(k++);
            if (previous==p.value) continue;
            previous=p.value;
            while (i<core.count && core.at(i)<p) ++i;
            while (j<outer.count && outer.at(j)<p) ++j;
            if ((i<core.count && !(p<core.at(i))) || (j<outer.count && !(p<outer.at(j)))) continue;
            const auto a=cells.at(p.value);
            word(refs,a.leaf); word(refs,a.offset); ++total;
        }
    }
    put(offsets,total);
    offsets.close(); refs.close();
}
}

void plan_halo(const std::string& dir,Id memory) {
    const auto h=Table<Header>(dir+"/header").at(0);
    CellNodes topology(dir);
    auto& cf=topology.cf; auto& cf_start=topology.cf_start;
    auto& fn=topology.fn; auto& fn_start=topology.fn_start;
    Table<Address> nodes(dir+"/address-0"),cells(dir+"/address-2");
    Sorter<Pair> stars(dir+"/stars.sort",memory);
    for (Id c=0;c<h.cells;++c) topology.each(c,[&](Id n) { stars.add({n,c}); });
    star_rows(dir+"/incidence",stars,h.nodes);
    Table<Id> start(dir+"/incidence-start"),incidence(dir+"/incidence-cells");
    Sorter<Pair> core(dir+"/core.sort",memory);
    for (Id n=0;n<h.nodes;++n) {
        const Id leaf=nodes.at(n).leaf;
        for (Id j=start.at(n);j<start.at(n+1);++j) core.add({leaf,incidence.at(j)});
    }
    Table<Pair> core_cells(core.finish());

    // EBR3 needs the complete star at the other end of each edge.
    // Vertices of the same leaf already contribute their stars to the core.
    mark_boundary(dir,h.nodes);
    Table<std::uint8_t> boundary(dir+"/boundary-nodes");
    Sorter<Pair> neighbors(dir+"/cross-edges.sort",memory/2);
    Sorter<Id> near_patch(dir+"/near-patch.sort",memory/2);
    const auto edge=[&](Id a,Id b) {
        const Id left=nodes.at(a).leaf,right=nodes.at(b).leaf;
        if (left!=right) { neighbors.add({left,b}); neighbors.add({right,a}); }
        if (boundary.at(a) || boundary.at(b)) { near_patch.add(a); near_patch.add(b); }
    };
    if (h.dimension==1) {
        for (Id c=0;c<h.cells;++c) {
            const Id j=cf_start.at(c);
            edge(fn.at(fn_start.at(cf.at(j))),fn.at(fn_start.at(cf.at(j+1))));
        }
    } else {
        for (Id f=0;f<h.faces;++f) {
            const Id first=fn_start.at(f),end=fn_start.at(f+1);
            for (Id j=first+1;j<end;++j) edge(fn.at(j-1),fn.at(j));
            if (h.dimension==3) edge(fn.at(end-1),fn.at(first));
        }
    }
    Table<Id> needed(near_patch.finish());
    Table<Pair> sorted(neighbors.finish());
    // Periodic pairs are chosen at run time. Save vertex stars only in the
    // boundary band that a first hit across a possible seam can reach.
    Sorter<Id> band_cells(dir+"/band-cells.sort",memory);
    for (Id i=0;i<needed.count;++i) {
        const Id n=needed.at(i);
        if (i && n==needed.at(i-1)) continue;
        for (Id j=start.at(n);j<start.at(n+1);++j) band_cells.add(incidence.at(j));
    }
    Table<Id> band(band_cells.finish());
    Sorter<Id> band_nodes(dir+"/band-nodes.sort",memory);
    for (Id i=0;i<band.count;++i) if (!i || band.at(i)!=band.at(i-1))
        topology.each(band.at(i),[&](Id n) { band_nodes.add(n); });
    Table<Id> boundary_support(band_nodes.finish());
    boundary_stars(dir,boundary_support,start,incidence,cells);
    Sorter<Pair> halo(dir+"/halo.sort",memory);
    expand_stars(sorted,start,incidence,halo);
    Table<Pair> outer(halo.finish());
    const Id leaves=Table<Leaf>(dir+"/leaves").count;
    Sorter<Pair> far_nodes(dir+"/far-nodes.sort",memory);
    leaf_rows(dir,core_cells,outer,cells,leaves,[&](Id leaf,Id c) {
        // Any vertex of this cell (including diagonally across a face) can
        // support the first ray hit. Owned vertices already have full stars.
        topology.each(c,[&](Id n) { if (nodes.at(n).leaf!=leaf) far_nodes.add({leaf,n}); });
    });
    Table<Pair> centers(far_nodes.finish());
    Sorter<Pair> far_cells(dir+"/far-cells.sort",memory);
    expand_stars(centers,start,incidence,far_cells);
    Table<Pair> far(far_cells.finish());
    far_rows(dir,core_cells,outer,far,cells,leaves);
}
} }
