#ifndef AERO_MDEC_H
#define AERO_MDEC_H
#include "mesh.h"
#include <array>
#include <fstream>
#include <map>
#include <set>

namespace aero { namespace mdec {
using Id=MeshIndex;
using Address=std::pair<Id,Id>; // leaf, byte position in its decoded payload
const Id absent=~Id(0);
enum Kind { Node,Face,Cell };
void check(bool condition,const char* message);
Id read_word(const std::string& data,std::size_t offset);
void append_word(std::string& data,Id value);

// Keep the wire representation in memory. A reply for a halo request can
// copy a record directly without serializing its connectivity again.
struct Record {
    std::string bytes;
    Id word(std::size_t offset) const { return read_word(bytes,offset); }
    Id kind() const { return word(0); }
    Id id() const { return word(8); }
    double coordinate(unsigned axis) const;
    Id primary_count() const;
    Id secondary_count() const;
    Address link(Id i,bool secondary=false) const;
    std::int8_t sign(Id i) const;
};
using Records=std::map<Address,Record>;
using Addresses=std::set<Address>;

class Archive {
public:
    explicit Archive(const std::string& path);
    static bool matches(const std::string& path);
    std::vector<Id> partition(int processes);
    struct Prefix { Id raw=0,nodes=0,halo=0; };
    Prefix prefix(Id end_leaf,bool fifth);
    void read_leaf(Id leaf,Records& records);
    bool indexed_halo() const { return version_>=2; }
    bool far_halo() const { return version_>=3; }
    void cell_star(Id node,Addresses& cells);
    void leaf_cells(Id leaf,bool outer,Addresses& cells);
    void far_cells(Id leaf,Addresses& cells);
    void read_records(const Addresses& needed,Kind kind,Records& records);
    std::string mesh_sha256;
    unsigned dimension;
    double scale;
    std::array<Id,3> counts;
    Id boundaries,leaves;
    std::vector<std::string> patches;
private:
    std::ifstream file_;
    struct Rows { Id offsets,refs,count,entries; };
    using Branch=std::array<Id,8>;
    Branch branch(Id entry);
    Branch root();
    void children(const Branch& parent,Branch& left,Branch& right);
    Id row_offset(const Rows& rows,Id row);
    void read_row(const Rows& rows,Id row,Addresses& result);
    Rows stars_{},halo_{},far_{};
    Id version_,length_,index_,tree_,payload_,entries_,root_;
    bool compressed_;
};
} }
#endif
