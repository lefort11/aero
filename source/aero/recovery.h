#ifndef AERO_RECOVERY_H
#define AERO_RECOVERY_H
#include "../io/settings.h"
#include "indexed_io.h"
#include <memory>

namespace aero {
// Rows are ordered by original node ID. Each rank supplies only its owned
// original nodes (including its periodic copies). Halo storage is reconstructed.
class Recovery {
public:
    Recovery(const RunSettings& run,const Processes& world,std::uint64_t count,unsigned dimension,
             std::string mesh_sha256,const std::vector<OwnedNode>& nodes);
    ~Recovery();
    void restore(std::vector<double>& state,std::vector<double>& initial);
    void save(std::uint64_t step,double time,const std::vector<double>& state,const std::vector<double>& initial);
    bool due(std::uint64_t step,double time) const;
    const std::string& fields_directory() const { return fields_; }
    std::uint64_t step=0;
    double time=0;
private:
    struct Lock;
    std::unique_ptr<Lock> lock_;
    const RunSettings& run_;
    const Processes& world_;
    std::uint64_t count_;
    unsigned width_;
    std::string hash_,fields_;
    const std::vector<OwnedNode>& nodes_;
    int last_=-1;
    std::string path(int slot) const;
};
}
#endif
