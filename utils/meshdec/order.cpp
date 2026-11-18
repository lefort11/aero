#include "meshdec.h"

namespace aero { namespace meshdec {
namespace {
Id lower(Table<Entity>& table,const Entity& value) {
    Id a=0,b=table.count;
    while (a<b) {
        const Id mid=a+(b-a)/2;
        if (table.at(mid)<value) a=mid+1;
        else b=mid;
    }
    return a;
}
Id width(Table<Id>& offsets,Id i) { return offsets.at(i+1)-offsets.at(i); }
struct Location {
    Id kind,id;
    Address address;
    bool operator<(const Location& v) const { return kind==v.kind ? id<v.id : kind<v.kind; }
};
}

void order_entities(const std::string& dir,Id memory,int rank,int size) {
    const auto h=Table<Header>(dir+"/header").at(0);
    Table<Id> node_key(dir+"/node-key"),face_key(dir+"/face-key");
    Table<Id> nf(dir+"/nf-start"),fn(dir+"/fn-start"),fc(dir+"/fc-start"),cf(dir+"/cf-start"),refs(dir+"/cf");
    Sorter<Entity> sorted(part(dir+"/local",rank),memory);
    const auto nodes=divide(h.nodes,rank,size),faces=divide(h.faces,rank,size),cells=divide(h.cells,rank,size);
    for (Id i=nodes.begin;i<nodes.end;++i)
        sorted.add({node_key.at(i),0,i,plus(48,times(16,width(nf,i)))});
    for (Id i=faces.begin;i<faces.end;++i)
        sorted.add({face_key.at(i),1,i,plus(48,times(16,plus(width(fn,i),width(fc,i))))});
    for (Id i=cells.begin;i<cells.end;++i) {
        Id key=none;
        for (Id k=cf.at(i);k<cf.at(i+1);++k) key=std::min(key,face_key.at(refs.at(k)));
        sorted.add({key,2,i,plus(24,times(17,width(cf,i)))});
    }
    sorted.finish();
}

// Regular samples choose spatial ranges. Equal keys are ordered by kind/ID.
// Boundaries affect who sorts the records, never the final archive layout.
void choose_splitters(const std::string& dir,Id memory,int size) {
    Sorter<Entity> samples(dir+"/samples",memory);
    for (int r=0;r<size;++r) {
        Table<Entity> data(part(dir+"/local",r));
        const int n=static_cast<int>(std::min<Id>(data.count,64));
        for (int i=0;i<n;++i) samples.add(data.at(divide(data.count,i,n).begin));
    }
    Table<Entity> sorted(samples.finish());
    auto split=output(dir+"/split");
    for (int r=1;r<size;++r) {
        Entity pivot{};
        if (sorted.count) pivot=sorted.at(std::min(sorted.count-1,divide(sorted.count,r,size).begin));
        put(split,pivot);
    }
    split.close();
}
void sort_range(const std::string& dir,Id memory,int rank,int size) {
    if (size==1) {
        require(std::rename(part(dir+"/local",0).c_str(),part(dir+"/range",0).c_str())==0,"Cannot rename sorted run");
        return;
    }
    Table<Entity> split(dir+"/split");
    Sorter<Entity> result(part(dir+"/range",rank),memory);
    for (int r=0;r<size;++r) {
        Table<Entity> data(part(dir+"/local",r));
        const Id begin=rank ? lower(data,split.at(static_cast<unsigned>(rank-1))) : 0;
        const Id end=rank+1<size ? lower(data,split.at(static_cast<unsigned>(rank))) : data.count;
        for (Id k=begin;k<end;++k) result.add(data.at(k));
    }
    result.finish();
}

void plan_leaves(const std::string& dir,const Options& options,int size) {
    auto ordered=output(dir+"/ordered"), leaves=output(dir+"/leaves");
    Sorter<Location> locations(dir+"/locations",options.memory);
    const Id target=options.leaf_bytes;
    const Id nodes=Table<Header>(dir+"/header").at(0).nodes;
    Id node_limit=none;
    if (nodes && options.target_ranks) {
        const Id desired=options.target_ranks>nodes/4 ? nodes : 4*options.target_ranks;
        node_limit=nodes/desired+(nodes%desired!=0 ? 1u : 0u);
    }
    Leaf leaf{};
    Id number=0;
    for (int r=0;r<size;++r) {
        Table<Entity> range(part(dir+"/range",r));
        for (Id k=0;k<range.count;++k) {
            const auto e=range.at(k);
            const bool full=leaf.raw_bytes>=target || e.bytes>target-leaf.raw_bytes;
            if (leaf.records && (full || (e.kind==0 && leaf.nodes>=node_limit))) {
                put(leaves,leaf);
                leaf={plus(leaf.first,leaf.records),0,0,0,0,0};
                ++number;
            }
            locations.add({e.kind,e.id,{number,leaf.raw_bytes}});
            leaf.raw_bytes=plus(leaf.raw_bytes,e.bytes);
            ++leaf.records;
            if (e.kind==0) ++leaf.nodes;
            else if (e.kind==1) ++leaf.faces;
            else ++leaf.cells;
            put(ordered,e);
        }
    }
    if (leaf.records) put(leaves,leaf);
    leaves.close();
    ordered.close();
    Table<Location> sorted(locations.finish());
    Id i=0;
    for (Id kind=0;kind<3;++kind) {
        auto addresses=output(dir+"/address-"+std::to_string(kind));
        Id next=0;
        while (i<sorted.count && sorted.at(i).kind==kind) {
            const auto loc=sorted.at(i++);
            require(loc.id==next++,"Incomplete entity address map");
            put(addresses,loc.address);
        }
        addresses.close();
    }
    plan_halo(dir,options.memory);
}
} }
