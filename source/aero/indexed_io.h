#ifndef AERO_INDEXED_IO_H
#define AERO_INDEXED_IO_H
#include "../parallel/processes.h"
#include <functional>

namespace aero {
struct OwnedNode {
    std::uint64_t id,representative,slot; // original ID, periodic representative, local state slot
};
namespace indexed_io {
enum class Access { Read, Write };
// The j-th field of a node starts at data+j*pitch. Encode/decode the node once.
using Row=std::function<void(const OwnedNode&,char* data,std::size_t pitch)>;

// The caller has validated the file extent. Nodes are sorted by original ID.
void transfer(const Processes& world,const std::string& file,Access access,unsigned stride,
              const std::vector<OwnedNode>& nodes,const std::vector<std::uint64_t>& offsets,const Row& row);
void put_real(char* dst,double value);
} }
#endif
